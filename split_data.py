"""
split_data.py

将统一标准格式的数据（FASTA + .npy）划分为训练集、验证集、测试集。

输入：
  - FASTA 文件：每个条目是一条序列
  - .npy 标签文件：形状 [N, target_length, num_channels]

输出：
  - {prefix}_train.fa + {prefix}_train_labels.npy
  - {prefix}_val.fa + {prefix}_val_labels.npy
  - {prefix}_test.fa + {prefix}_test_labels.npy

用法：
  python split_data.py \
      --fasta_file standard_data/oryza_sativa.fa \
      --labels_file standard_data/oryza_sativa_labels.npy \
      --output_dir processed_data \
      --output_prefix oryza_sativa \
      --split_ratios 0.7 0.15 0.15 \
      --split_method random \
      --seed 42
"""

import argparse
import numpy as np
import os
import random
from Bio import SeqIO


def split_data(fasta_path, labels_path, output_dir, output_prefix,
               split_ratios, split_method, seed):
    os.makedirs(output_dir, exist_ok=True)

    random.seed(seed)
    np.random.seed(seed)

    print(f"Loading FASTA: {fasta_path}")
    records = list(SeqIO.parse(fasta_path, "fasta"))
    n = len(records)
    print(f"  Total sequences: {n}")

    print(f"Loading labels: {labels_path}")
    labels = np.load(labels_path)
    print(f"  Labels shape: {labels.shape}")

    if len(records) != labels.shape[0]:
        raise ValueError(
            f"FASTA records ({len(records)}) and labels ({labels.shape[0]}) count mismatch"
        )

    indices = list(range(n))

    if split_method == 'random':
        random.shuffle(indices)
    elif split_method == 'sequential':
        pass
    else:
        raise ValueError(f"Unknown split method: {split_method}")

    train_ratio, val_ratio, test_ratio = split_ratios
    total_ratio = train_ratio + val_ratio + test_ratio
    train_ratio /= total_ratio
    val_ratio /= total_ratio
    test_ratio /= total_ratio

    train_end = int(n * train_ratio)
    val_end = train_end + int(n * val_ratio)

    train_idx = indices[:train_end]
    val_idx = indices[train_end:val_end]
    test_idx = indices[val_end:]

    splits = {
        'train': train_idx,
        'val': val_idx,
        'test': test_idx,
    }

    for split_name, split_indices in splits.items():
        fasta_out = os.path.join(output_dir, f"{output_prefix}_{split_name}.fa")
        labels_out = os.path.join(output_dir, f"{output_prefix}_{split_name}_labels.npy")

        split_records = [records[i] for i in split_indices]
        with open(fasta_out, 'w') as fh:
            SeqIO.write(split_records, fh, 'fasta')

        split_labels = labels[split_indices]
        np.save(labels_out, split_labels.astype(np.float32))

        print(f"\n{split_name}: {len(split_indices)} samples")
        print(f"  FASTA:  {fasta_out}")
        print(f"  Labels: {labels_out} (shape: {split_labels.shape})")

    print(f"\nDone! All files saved to: {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Split standard format data into train/val/test sets"
    )
    parser.add_argument("--fasta_file", type=str, required=True,
                        help="Path to input FASTA file")
    parser.add_argument("--labels_file", type=str, required=True,
                        help="Path to input .npy labels file")
    parser.add_argument("--output_dir", type=str, default="processed_data",
                        help="Output directory (default: processed_data)")
    parser.add_argument("--output_prefix", type=str, default="data",
                        help="Output file prefix (default: data)")
    parser.add_argument("--split_ratios", type=float, nargs=3,
                        default=[0.7, 0.15, 0.15],
                        help="Train/val/test ratios (default: 0.7 0.15 0.15)")
    parser.add_argument("--split_method", type=str, default="random",
                        choices=["random", "sequential"],
                        help="Split method: random or sequential (default: random)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (default: 42)")
    args = parser.parse_args()

    split_data(
        args.fasta_file,
        args.labels_file,
        args.output_dir,
        args.output_prefix,
        tuple(args.split_ratios),
        args.split_method,
        args.seed
    )


if __name__ == "__main__":
    main()