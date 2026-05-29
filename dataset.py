import torch
from torch.utils.data import Dataset
from Bio import SeqIO
import numpy as np


class GenomeDataset(Dataset):
    def __init__(self, fasta_file, labels_file, label_type='peak', log1p_transform=True):
        """
        Args:
            fasta_file: FASTA文件路径，每个条目是一条输入序列（支持变长）
            labels_file: .npy标签文件路径，形状 [N, max_target_length, num_channels]
                         短序列的标签需zero-padding到max_target_length
            label_type: 'peak' 或 'coverage'
            log1p_transform: 是否对coverage标签做log1p变换（默认True）
        """
        self.records = list(SeqIO.parse(fasta_file, "fasta"))
        self.labels = np.load(labels_file, allow_pickle=True)
        self.label_type = label_type
        self.log1p_transform = log1p_transform

        if len(self.records) != self.labels.shape[0]:
            raise ValueError(
                f"FASTA records ({len(self.records)}) and labels ({self.labels.shape[0]}) count mismatch"
            )

        if self.label_type == 'coverage' and self.log1p_transform:
            if self.labels.ndim == 3:
                self.labels = np.log1p(self.labels).astype(np.float32)
            else:
                self.labels = [np.log1p(l).astype(np.float32) for l in self.labels]

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        torch.backends.cudnn.benchmark = False
        record = self.records[idx]

        seq = str(record.seq).upper()
        seq_len = len(seq)
        seq_tensor = torch.zeros(seq_len, 4)

        for i, base in enumerate(seq):
            if base == 'A':
                seq_tensor[i, 0] = 1
            elif base == 'C':
                seq_tensor[i, 1] = 1
            elif base == 'G':
                seq_tensor[i, 2] = 1
            elif base == 'T':
                seq_tensor[i, 3] = 1
            else:
                seq_tensor[i] = 0.25

        label = self.labels[idx]
        if isinstance(label, np.ndarray):
            label_tensor = torch.FloatTensor(label)
        else:
            label_tensor = torch.FloatTensor(np.array(label))

        return seq_tensor, label_tensor, seq_len


def get_collate_fn(pool_size):
    def collate_fn(batch):
        seqs, labels, seq_lens = zip(*batch)

        seq_lens = tuple(int(s) for s in seq_lens)
        max_seq_len = max(seq_lens)
        if max_seq_len % pool_size != 0:
            max_seq_len = ((max_seq_len + pool_size - 1) // pool_size) * pool_size

        padded_seqs = torch.zeros(len(seqs), max_seq_len, 4)
        for i, seq in enumerate(seqs):
            padded_seqs[i, :seq.shape[0]] = seq

        expected_target_len = max_seq_len // pool_size
        num_channels = labels[0].shape[-1]
        padded_labels = torch.zeros(len(labels), expected_target_len, num_channels)
        for i, label in enumerate(labels):
            t_len = min(label.shape[0], expected_target_len)
            padded_labels[i, :t_len] = label[:t_len]

        seq_lens = torch.LongTensor(seq_lens)
        return padded_seqs, padded_labels, seq_lens
    return collate_fn
