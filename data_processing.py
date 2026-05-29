import os
import random
import numpy as np
from pathlib import Path
from collections import defaultdict
from Bio import SeqIO
import torch
from sklearn.model_selection import train_test_split


class BinDatasetProcessor:
    def __init__(self, bin_size=1024, seq_length=196608, split_ratios=(0.7, 0.15, 0.15),
                 split_method='random', seed=42):
        """
        Args:
            bin_size: Size of each bin in bp (default: 1024)
            seq_length: Total sequence length for model input (default: 196608)
            split_ratios: Tuple of (train, val, test) ratios
            split_method: 'random' or 'chrom_end'
            seed: Random seed for reproducibility
        """
        self.bin_size = bin_size
        self.seq_length = seq_length
        self.bins_per_seq = seq_length // bin_size  # 192
        self.split_ratios = split_ratios
        self.split_method = split_method
        self.seed = seed
        random.seed(seed)
        np.random.seed(seed)

    def process_species(self, fasta_path, species_name):
        """Process a single species FASTA file"""
        records = list(SeqIO.parse(fasta_path, "fasta"))

        # Organize bins by chromosome
        chr_bins = defaultdict(list)
        for record in records:
            header_parts = record.description.split()
            chr_loc = header_parts[0].split(':')
            chr_name = chr_loc[0]
            start, end = map(int, chr_loc[1].split('-'))
            labels = list(map(int, header_parts[1].strip('()').split(',')))

            chr_bins[chr_name].append({
                'start': start,
                'end': end,
                'seq': str(record.seq),
                'labels': labels
            })

        # Sort bins by chromosome and position
        for chr_name in chr_bins:
            chr_bins[chr_name].sort(key=lambda x: x['start'])

        # Create sequences of 192 consecutive bins
        sequences = []
        for chr_name, bins in chr_bins.items():
            num_bins = len(bins)
            num_sequences = num_bins // self.bins_per_seq

            # Create complete sequences
            for i in range(num_sequences):
                start_idx = i * self.bins_per_seq
                end_idx = start_idx + self.bins_per_seq
                seq_bins = bins[start_idx:end_idx]
                sequences.append(self._create_sequence(seq_bins, chr_name, species_name))

            # Handle remaining bins: take last 192 bins (may overlap)
            if num_bins % self.bins_per_seq != 0:
                seq_bins = bins[-self.bins_per_seq:]
                sequences.append(self._create_sequence(seq_bins, chr_name, species_name))

        return sequences

    def _create_sequence(self, bins, chr_name, species_name):
        """Helper to create a sequence entry"""
        combined_seq = ''.join([b['seq'] for b in bins])
        combined_labels = [b['labels'] for b in bins]
        start_pos = bins[0]['start']
        end_pos = bins[-1]['end']

        return {
            'species': species_name,
            'chr': chr_name,
            'start': start_pos,
            'end': end_pos,
            'seq': combined_seq,
            'labels': combined_labels
        }

    def split_dataset(self, sequences):
        """Split dataset into train/val/test based on specified method"""
        if self.split_method == 'random':
            train, test = train_test_split(
                sequences,
                test_size=self.split_ratios[2],
                random_state=self.seed
            )
            train, val = train_test_split(
                train,
                test_size=self.split_ratios[1] / (self.split_ratios[0] + self.split_ratios[1]),
                random_state=self.seed
            )
        elif self.split_method == 'chrom_end':
            sequences_sorted = sorted(sequences, key=lambda x: (x['chr'], x['start']))
            total = len(sequences_sorted)
            test_size = int(total * self.split_ratios[2])
            val_size = int(total * self.split_ratios[1])

            test = sequences_sorted[-test_size:]
            val = sequences_sorted[-(test_size + val_size):-test_size]
            train = sequences_sorted[:-(test_size + val_size)]
        else:
            raise ValueError(f"Unknown split method: {self.split_method}")

        return train, val, test

    def save_dataset(self, dataset, output_path):
        """Save dataset to FASTA file with modified headers"""
        with open(output_path, 'w') as f:
            for item in dataset:
                labels_str = ';'.join([','.join(map(str, label)) for label in item['labels']])
                header = f">{item['species']}|{item['chr']}|{item['start']}-{item['end']}|{labels_str}"
                f.write(f"{header}\n{item['seq']}\n")

    def print_label_stats(self, sequences, label_names=None):
        """Print per-channel label statistics"""
        if not sequences:
            print("  No sequences to compute statistics.")
            return

        labels_list = [item['labels'] for item in sequences]
        all_labels = np.array(labels_list, dtype=np.float32)

        if all_labels.ndim == 1:
            all_labels = all_labels.reshape(-1, 1)

        num_channels = all_labels.shape[-1] if all_labels.ndim > 1 else 1

        if label_names is None:
            label_names = [f"ch_{i}" for i in range(num_channels)]

        print(f"\n  Label statistics ({len(sequences)} sequences, {num_channels} channels):")
        print(f"  {'Channel':<12} {'Min':<14} {'Max':<14} {'Mean':<14} {'Std':<14} {'Variance':<14} {'Median':<14} {'Zero%':<10}")
        print(f"  {'-'*106}")

        for i in range(num_channels):
            if all_labels.ndim > 1:
                ch = all_labels[:, i].flatten()
            else:
                ch = all_labels.flatten()
            non_nan = ch[~np.isnan(ch)]
            if len(non_nan) > 0:
                ch_min = non_nan.min()
                ch_max = non_nan.max()
                ch_mean = non_nan.mean()
                ch_std = non_nan.std()
                ch_var = non_nan.var()
                ch_median = np.median(non_nan)
                zero_pct = (non_nan == 0).sum() / len(non_nan) * 100
            else:
                ch_min = ch_max = ch_mean = ch_std = ch_var = ch_median = float('nan')
                zero_pct = float('nan')
            name = label_names[i] if i < len(label_names) else f"ch_{i}"
            print(f"  {name:<12} {ch_min:<14.6g} {ch_max:<14.6g} {ch_mean:<14.6g} {ch_std:<14.6g} {ch_var:<14.6g} {ch_median:<14.6g} {zero_pct:<10.2f}")

    def process_all_species(self, input_files, output_dir):
        """Process all species and save train/val/test splits"""
        os.makedirs(output_dir, exist_ok=True)

        all_train, all_val, all_test = [], [], []

        for species_name, fasta_path in input_files.items():
            print(f"\nProcessing {species_name}...")
            sequences = self.process_species(fasta_path, species_name)
            train, val, test = self.split_dataset(sequences)

            self.print_label_stats(sequences)

            # Save species-specific files
            species_dir = os.path.join(output_dir, species_name)
            os.makedirs(species_dir, exist_ok=True)

            self.save_dataset(train, os.path.join(species_dir, f"{species_name}_train.fa"))
            self.save_dataset(val, os.path.join(species_dir, f"{species_name}_val.fa"))
            self.save_dataset(test, os.path.join(species_dir, f"{species_name}_test.fa"))

            # Combine for mixed training
            all_train.extend(train)
            all_val.extend(val)
            all_test.extend(test)

        # Save mixed datasets (shuffled)
        random.shuffle(all_train)
        random.shuffle(all_val)
        random.shuffle(all_test)

        self.save_dataset(all_train, os.path.join(output_dir, "mixed_train.fa"))
        self.save_dataset(all_val, os.path.join(output_dir, "mixed_val.fa"))
        self.save_dataset(all_test, os.path.join(output_dir, "mixed_test.fa"))

        print(f"\n{'='*60}")
        print("Mixed dataset label statistics:")
        self.print_label_stats(all_train)

        print("\nData processing complete!")


# Example usage:
if __name__ == "__main__":
    # Configure paths to your input files
    input_files = {
        "ath": "/mnt/c/Users/user/OneDrive/课题组/Enformer/enformer-pytorch-main/enformer-pytorch-main/data plants/ath/input_data.fa",
        "osa": "/mnt/c/Users/user/OneDrive/课题组/Enformer/enformer-pytorch-main/enformer-pytorch-main/data plants/osa/input_data.fa"
    }

    # Output directory
    output_dir = "processed_data"

    # Create processor and run
    processor = BinDatasetProcessor(
        split_ratios=(0.7, 0.15, 0.15),
        split_method='random'  # or 'chrom_end'
    )
    processor.process_all_species(input_files, output_dir)