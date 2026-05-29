"""
convert_h5_to_standard.py

将项目内部的 .h5 数据文件转换为统一标准格式：
  - 序列 → FASTA 文件（每个条目一条序列）
  - 标签 → .npy 文件（形状 [N, target_length, num_channels]）

用法：
  python convert_h5_to_standard.py \
      --h5_file dataset_oryza_sativa_TEST_only1.h5 \
      --output_dir standard_data \
      --output_prefix oryza_sativa
"""

import argparse
import h5py
import numpy as np
import os
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord


def convert_h5_to_standard(h5_path, output_dir, output_prefix):
    os.makedirs(output_dir, exist_ok=True)

    fasta_path = os.path.join(output_dir, f"{output_prefix}.fa")
    labels_path = os.path.join(output_dir, f"{output_prefix}_labels.npy")

    print(f"Reading H5 file: {h5_path}")

    with h5py.File(h5_path, 'r') as f:
        sequence_strings = f['inputs/sequence_string'][:]
        n_samples = len(sequence_strings)
        print(f"  Total samples: {n_samples}")

        label_datasets = []
        label_names = []
        for key in f['labels'].keys():
            label_datasets.append(f[f'labels/{key}'][:])
            label_names.append(key)
            print(f"  Label: {key} -> shape {f[f'labels/{key}'].shape}")

        labels = np.stack(label_datasets, axis=-1)
        print(f"  Stacked labels shape: {labels.shape}")

    print(f"\nWriting FASTA: {fasta_path}")
    records = []
    for i in range(n_samples):
        seq_str = sequence_strings[i]
        if isinstance(seq_str, bytes):
            seq_str = seq_str.decode('ascii')
        record = SeqRecord(
            Seq(seq_str),
            id=f"seq_{i}",
            description=f"length={len(seq_str)}"
        )
        records.append(record)

    with open(fasta_path, 'w') as fh:
        SeqIO.write(records, fh, 'fasta')
    print(f"  Written {len(records)} records")

    print(f"\nWriting labels: {labels_path}")
    np.save(labels_path, labels.astype(np.float32))
    print(f"  Labels shape: {labels.shape}, dtype: {labels.dtype}")

    print(f"\nDone! Output files:")
    print(f"  FASTA:  {fasta_path}")
    print(f"  Labels: {labels_path}")
    num_channels = labels.shape[-1]
    print(f"\nLabel channels ({num_channels}):")
    print(f"  {'Channel':<12} {'Min':<14} {'Max':<14} {'Mean':<14} {'Std':<14} {'Variance':<14} {'Median':<14} {'Zero%':<10}")
    print(f"  {'-'*106}")
    for i, name in enumerate(label_names):
        ch = labels[:, :, i]
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
        print(f"  {name:<12} {ch_min:<14.6g} {ch_max:<14.6g} {ch_mean:<14.6g} {ch_std:<14.6g} {ch_var:<14.6g} {ch_median:<14.6g} {zero_pct:<10.2f}")


def main():
    parser = argparse.ArgumentParser(
        description="Convert H5 data to standard FASTA + .npy format"
    )
    parser.add_argument("--h5_file", type=str, required=True,
                        help="Path to input .h5 file")
    parser.add_argument("--output_dir", type=str, default="standard_data",
                        help="Output directory (default: standard_data)")
    parser.add_argument("--output_prefix", type=str, default="data",
                        help="Output file prefix (default: data)")
    args = parser.parse_args()

    convert_h5_to_standard(args.h5_file, args.output_dir, args.output_prefix)


if __name__ == "__main__":
    main()