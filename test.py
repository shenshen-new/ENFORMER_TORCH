import torch
import argparse
import numpy as np
from torch.utils.data import DataLoader
from dataset import GenomeDataset, get_collate_fn
from model import Enformer
from config import EnformerConfig
from sklearn.metrics import roc_auc_score, average_precision_score, r2_score
from scipy.stats import pearsonr, spearmanr
import os
import json


def compute_peak_metrics(pred, target):
    pred_flat = pred.cpu().numpy().flatten()
    target_flat = target.cpu().numpy().flatten()

    if len(np.unique(target_flat)) < 2:
        return {'auroc': 0.5, 'auprc': 0.0}

    try:
        auroc = roc_auc_score(target_flat, pred_flat)
        auprc = average_precision_score(target_flat, pred_flat)
    except Exception as e:
        print(f"Warning: Failed to compute metrics: {e}")
        auroc = 0.5
        auprc = 0.0

    return {'auroc': auroc, 'auprc': auprc}


def compute_coverage_metrics(pred, target):
    pred_flat = pred.cpu().numpy().flatten()
    target_flat = target.cpu().numpy().flatten()

    mse = np.mean((pred_flat - target_flat) ** 2)

    pearson_r = 0.0
    if np.std(pred_flat) > 0 and np.std(target_flat) > 0:
        try:
            pearson_r = pearsonr(pred_flat, target_flat)[0]
            if np.isnan(pearson_r):
                pearson_r = 0.0
        except Exception:
            pearson_r = 0.0

    spearman_r = 0.0
    if np.std(pred_flat) > 0 and np.std(target_flat) > 0:
        try:
            spearman_r = spearmanr(pred_flat, target_flat)[0]
            if np.isnan(spearman_r):
                spearman_r = 0.0
        except Exception:
            spearman_r = 0.0

    r2 = r2_score(target_flat, pred_flat)

    return {
        'pearson_r': pearson_r,
        'spearman_r': spearman_r,
        'mse': mse,
        'r2': r2
    }


def load_model(model_path, device):
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)

    if 'config' in checkpoint:
        config = checkpoint['config']
    else:
        print("Warning: No config found in checkpoint, using default config")
        config = EnformerConfig()

    model = Enformer(config).to(device)

    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)

    model.eval()

    label_type = getattr(config, 'label_type', 'peak')

    return model, config, label_type


def test_epoch(model, dataloader, device, label_type='peak', metrics_space='log1p', log1p_transform=True, pool_size=5):
    model.eval()
    total_loss = 0

    all_preds = []
    all_targets = []
    all_masks = []

    with torch.no_grad():
        for seqs, labels, seq_lens in dataloader:
            seqs = seqs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            seq_lens = seq_lens.to(device)

            with torch.amp.autocast(device_type=device.type, enabled=(device.type == 'cuda')):
                loss, pred_dict = model(seqs, target=labels, seq_lens=seq_lens)

            total_loss += loss.item()

            head_name = list(pred_dict.keys())[0]
            all_preds.append(pred_dict[head_name].cpu())
            all_targets.append(labels.cpu())

            batch_size = labels.shape[0]
            target_length = labels.shape[1]
            for i in range(batch_size):
                valid_len = seq_lens[i].item() // pool_size
                mask = torch.zeros(target_length, dtype=torch.float32)
                mask[:valid_len] = 1.0
                all_masks.append(mask)

    avg_loss = total_loss / len(dataloader)

    all_preds = torch.cat(all_preds, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    all_masks = torch.stack(all_masks, dim=0).unsqueeze(-1)

    if label_type == 'coverage' and log1p_transform and metrics_space == 'original':
        all_preds = torch.expm1(all_preds)
        all_targets = torch.expm1(all_targets)

    valid_mask = all_masks.expand_as(all_preds).bool()
    valid_preds = all_preds[valid_mask]
    valid_targets = all_targets[valid_mask]

    if label_type == 'peak':
        metrics = compute_peak_metrics(valid_preds, valid_targets)
    else:
        metrics = compute_coverage_metrics(valid_preds, valid_targets)

    return avg_loss, metrics


def main():
    parser = argparse.ArgumentParser(description="Test Enformer model")
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to trained model checkpoint (.pt file)")
    parser.add_argument("--test_fasta", type=str, required=True,
                        help="Test FASTA file")
    parser.add_argument("--test_labels", type=str, required=True,
                        help="Test labels .npy file")
    parser.add_argument("--batch_size", type=int, default=8,
                        help="Batch size")
    parser.add_argument("--num_workers", type=int, default=4,
                        help="Number of data loader workers")
    parser.add_argument("--label_type", type=str, default=None,
                        choices=["peak", "coverage"],
                        help="Label type (if not specified, inferred from model config)")
    parser.add_argument("--metrics_space", type=str, default="log1p",
                        choices=["log1p", "original"],
                        help="Space for computing metrics: 'log1p' (transformed) or 'original' (back-transformed). Default: log1p")
    parser.add_argument("--output_file", type=str, default=None,
                        help="Path to save results JSON (optional)")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Loading model from: {args.model_path}")

    model, config, model_label_type = load_model(args.model_path, device)

    label_type = args.label_type if args.label_type else model_label_type
    log1p_transform = getattr(config, 'log1p_transform', True)
    metrics_space = args.metrics_space
    print(f"Label type: {label_type} (from {'command line' if args.label_type else 'model config'})")
    print(f"Log1p transform: {log1p_transform}")
    print(f"Metrics space: {metrics_space}")

    output_heads = config.output_heads
    print(f"Output heads: {output_heads}")

    print(f"\nLoading test data...")
    print(f"  FASTA: {args.test_fasta}")
    print(f"  Labels: {args.test_labels}")
    test_dataset = GenomeDataset(args.test_fasta, args.test_labels, label_type=label_type, log1p_transform=log1p_transform)
    pool_size = getattr(config, 'pool_size', 5)
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=get_collate_fn(pool_size),
        num_workers=args.num_workers,
        pin_memory=True
    )

    print(f"Test samples: {len(test_dataset)}")
    print(f"Labels shape: {test_dataset.labels.shape}")

    print(f"\n{'='*60}")
    print(f"Starting evaluation on test set")
    print(f"{'='*60}")

    test_loss, test_metrics = test_epoch(model, test_loader, device, label_type, metrics_space, log1p_transform, pool_size)

    print(f"\n{'='*60}")
    print(f"Test Results (label_type={label_type}, metrics_space={metrics_space})")
    print(f"{'='*60}")
    print(f"Test Loss: {test_loss:.6f}")

    if label_type == 'peak':
        print(f"AUROC: {test_metrics.get('auroc', 0):.6f}")
        print(f"AUPRC: {test_metrics.get('auprc', 0):.6f}")
    else:
        print(f"Pearson r:  {test_metrics.get('pearson_r', 0):.6f}")
        print(f"Spearman r: {test_metrics.get('spearman_r', 0):.6f}")
        print(f"MSE:        {test_metrics.get('mse', 0):.6f}")
        print(f"R²:         {test_metrics.get('r2', 0):.6f}")

    results = {
        'label_type': label_type,
        'metrics_space': metrics_space,
        'test_loss': float(test_loss),
        'metrics': {k: float(v) for k, v in test_metrics.items()}
    }

    if args.output_file:
        with open(args.output_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to: {args.output_file}")

    return results


if __name__ == "__main__":
    main()
