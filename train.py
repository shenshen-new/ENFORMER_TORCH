import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from dataset import GenomeDataset, get_collate_fn
from model import Enformer
from config import EnformerConfig
import argparse
import os
from tqdm import tqdm
import numpy as np
import random
import time
import json
import sys
from sklearn.metrics import roc_auc_score, average_precision_score, r2_score
from scipy.stats import pearsonr, spearmanr


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


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


def train_epoch(model, dataloader, optimizer, device, scaler):
    model.train()
    total_loss = 0
    progress = tqdm(dataloader, desc="Training")

    for seqs, labels, seq_lens in progress:
        torch.cuda.empty_cache()

        seqs = seqs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        seq_lens = seq_lens.to(device)

        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast(device_type=device.type, enabled=(device.type == 'cuda')):
            loss, _ = model(seqs, target=labels, seq_lens=seq_lens)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        progress.set_postfix(loss=loss.item())

    return total_loss / len(dataloader)


def eval_epoch(model, dataloader, device, label_type='peak', metrics_space='log1p', log1p_transform=True, pool_size=5):
    model.eval()
    total_loss = 0

    all_preds = []
    all_targets = []
    all_masks = []

    with torch.no_grad():
        for seqs, labels, seq_lens in tqdm(dataloader, desc="Evaluating"):
            torch.cuda.empty_cache()

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


def main(args):
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    os.makedirs(args.output_dir, exist_ok=True)
    log_file = os.path.join(args.output_dir, "training_log.txt")

    original_stdout = sys.stdout
    class TeeOutput:
        def __init__(self, file_path):
            self.file = open(file_path, 'a')
            self.stdout = original_stdout
        def write(self, data):
            self.stdout.write(data)
            self.file.write(data)
            self.file.flush()
        def flush(self):
            self.stdout.flush()
            self.file.flush()

    sys.stdout = TeeOutput(log_file)

    def log_print(message):
        print(message)

    log_print("=" * 80)
    log_print(f"Training started at: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    log_print(f"Command line arguments: {args}")
    log_print(f"Device: {device}")
    log_print(f"Label type: {args.label_type}")
    log_print(f"Output directory: {args.output_dir}")
    log_print("=" * 80)

    log_print("\nLoading datasets...")
    train_dataset = GenomeDataset(args.train_fasta, args.train_labels, label_type=args.label_type, log1p_transform=args.log1p_transform)
    val_dataset = GenomeDataset(args.val_fasta, args.val_labels, label_type=args.label_type, log1p_transform=args.log1p_transform)
    test_dataset = GenomeDataset(args.test_fasta, args.test_labels, label_type=args.label_type, log1p_transform=args.log1p_transform)

    log_print(f"Train samples: {len(train_dataset)}")
    log_print(f"Val samples: {len(val_dataset)}")
    log_print(f"Test samples: {len(test_dataset)}")

    labels_shape = train_dataset.labels.shape
    num_channels = labels_shape[-1]
    target_length = labels_shape[1]

    seq_lengths = [len(rec.seq) for rec in train_dataset.records]
    min_seq_len = min(seq_lengths)
    max_seq_len = max(seq_lengths)
    log_print(f"Sequence lengths: min={min_seq_len}, max={max_seq_len}")
    log_print(f"Labels shape: {labels_shape} (target_length={target_length}, channels={num_channels})")

    if min_seq_len == max_seq_len:
        log_print(f"All sequences have the same length ({max_seq_len}), fixed-length mode")
    else:
        log_print(f"Variable-length sequences detected, dynamic padding enabled")

    collate_fn = get_collate_fn(args.pool_size)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=args.num_workers,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=args.num_workers,
        pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=args.num_workers,
        pin_memory=True
    )

    log_print("\nInitializing model...")
    config = EnformerConfig(
        output_heads={args.species_name: num_channels},
        target_length=target_length,
        input_length=args.input_length,
        pool_size=args.pool_size,
        num_conv_layers=args.num_conv_layers,
        dropout_rate=args.dropout,
        use_checkpointing=args.use_checkpointing,
        label_type=args.label_type,
        coverage_loss=args.coverage_loss,
        coverage_activation=args.coverage_activation,
        log1p_transform=args.log1p_transform
    )
    model = Enformer(config).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log_print(f"Total parameters: {total_params:,}")
    log_print(f"Trainable parameters: {trainable_params:,}")
    log_print(f"Input length: {args.input_length}, Target length: {target_length}")
    log_print(f"Pool size: {args.pool_size}, Conv layers: {args.num_conv_layers}")
    if args.label_type == 'coverage':
        log_print(f"Coverage loss: {args.coverage_loss}")
        log_print(f"Coverage activation: {args.coverage_activation}")
        log_print(f"Log1p transform: {args.log1p_transform}")
        log_print(f"Metrics space: {args.metrics_space}")

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.5, verbose=False)


    log_print(f"\n{'='*80}")
    log_print(f"Starting training for {args.epochs} epochs")
    log_print(f"Batch size: {args.batch_size}")
    log_print(f"Learning rate: {args.lr}")
    log_print(f"Weight decay: {args.weight_decay}")
    log_print(f"{'='*80}\n")

    sample = next(iter(train_loader))
    seqs, labels, seq_lens = sample
    log_print(f"Batch shapes - seqs: {seqs.shape}, labels: {labels.shape}, seq_lens: {seq_lens}")
    log_print(f"Label value range: [{labels.min():.4f}, {labels.max():.4f}]")

    best_val_loss = float('inf')
    best_epoch = 0
    scaler = torch.amp.GradScaler('cuda', enabled=True)

    history = {
        'epoch': [],
        'train_loss': [],
        'val_loss': [],
        'lr': []
    }

    if args.label_type == 'peak':
        history['train_auroc'] = []
        history['train_auprc'] = []
        history['val_auroc'] = []
        history['val_auprc'] = []
    else:
        history['train_pearson_r'] = []
        history['train_spearman_r'] = []
        history['train_mse'] = []
        history['train_r2'] = []
        history['val_pearson_r'] = []
        history['val_spearman_r'] = []
        history['val_mse'] = []
        history['val_r2'] = []

    for epoch in range(args.epochs):
        epoch_start_time = time.time()
        current_lr = optimizer.param_groups[0]['lr']

        log_print(f"{'─'*60}")
        log_print(f"Epoch {epoch + 1}/{args.epochs} (LR: {current_lr:.2e})")
        log_print(f"{'─'*60}")

        train_loss = train_epoch(model, train_loader, optimizer, device, scaler)
        train_eval_loss, train_metrics = eval_epoch(
            model, train_loader, device, args.label_type,
            metrics_space=args.metrics_space, log1p_transform=args.log1p_transform,
            pool_size=args.pool_size
        )
        val_loss, val_metrics = eval_epoch(
            model, val_loader, device, args.label_type,
            metrics_space=args.metrics_space, log1p_transform=args.log1p_transform,
            pool_size=args.pool_size
        )

        epoch_time = time.time() - epoch_start_time

        history['epoch'].append(epoch + 1)
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['lr'].append(current_lr)

        if args.label_type == 'peak':
            history['train_auroc'].append(train_metrics.get('auroc', 0))
            history['train_auprc'].append(train_metrics.get('auprc', 0))
            history['val_auroc'].append(val_metrics.get('auroc', 0))
            history['val_auprc'].append(val_metrics.get('auprc', 0))
        else:
            history['train_pearson_r'].append(train_metrics.get('pearson_r', 0))
            history['train_spearman_r'].append(train_metrics.get('spearman_r', 0))
            history['train_mse'].append(train_metrics.get('mse', 0))
            history['train_r2'].append(train_metrics.get('r2', 0))
            history['val_pearson_r'].append(val_metrics.get('pearson_r', 0))
            history['val_spearman_r'].append(val_metrics.get('spearman_r', 0))
            history['val_mse'].append(val_metrics.get('mse', 0))
            history['val_r2'].append(val_metrics.get('r2', 0))

        log_print(f"\nResults after Epoch {epoch + 1}:")
        log_print(f"  Train Loss:     {train_loss:.6f}")
        log_print(f"  Val Loss:       {val_loss:.6f}")

        if args.label_type == 'peak':
            log_print(f"  Train AUROC:    {train_metrics.get('auroc', 0):.6f}")
            log_print(f"  Train AUPRC:    {train_metrics.get('auprc', 0):.6f}")
            log_print(f"  Val AUROC:      {val_metrics.get('auroc', 0):.6f}")
            log_print(f"  Val AUPRC:      {val_metrics.get('auprc', 0):.6f}")
        else:
            log_print(f"  Train Pearson r:  {train_metrics.get('pearson_r', 0):.6f}")
            log_print(f"  Train Spearman r: {train_metrics.get('spearman_r', 0):.6f}")
            log_print(f"  Train MSE:        {train_metrics.get('mse', 0):.6f}")
            log_print(f"  Train R²:         {train_metrics.get('r2', 0):.6f}")
            log_print(f"  Val Pearson r:    {val_metrics.get('pearson_r', 0):.6f}")
            log_print(f"  Val Spearman r:   {val_metrics.get('spearman_r', 0):.6f}")
            log_print(f"  Val MSE:          {val_metrics.get('mse', 0):.6f}")
            log_print(f"  Val R²:           {val_metrics.get('r2', 0):.6f}")

        log_print(f"  Time:           {epoch_time:.2f}s ({epoch_time/60:.2f} min)")

        scheduler.step(val_loss)

        is_best = val_loss < best_val_loss
        if is_best:
            best_val_loss = val_loss
            best_epoch = epoch + 1

            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_val_loss': best_val_loss,
                'config': config,
                'history': history
            }

            model_path = os.path.join(args.output_dir, "best_model.pt")
            torch.save(checkpoint, model_path)

            log_print(f"\n  New best model saved! (Val Loss: {best_val_loss:.6f})")
        else:
            log_print(f"\n  Val loss did not improve from {best_val_loss:.6f} (epoch {best_epoch})")

        if (epoch + 1) % 5 == 0:
            checkpoint_path = os.path.join(args.output_dir, f"checkpoint_epoch_{epoch+1}.pt")
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'history': history
            }, checkpoint_path)
            log_print(f"  Checkpoint saved at epoch {epoch + 1}")

        log_print("")

    log_print(f"\n{'='*80}")
    log_print(f"Training completed at: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    log_print(f"Best model was from Epoch {best_epoch} with Val Loss: {best_val_loss:.6f}")
    log_print(f"{'='*80}\n")

    log_print("Loading best model for test evaluation...")
    best_model_path = os.path.join(args.output_dir, "best_model.pt")
    checkpoint = torch.load(best_model_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    log_print(f"\n{'='*80}")
    log_print(f"Test Set Evaluation (best model from epoch {best_epoch})")
    log_print(f"{'='*80}")

    test_loss, test_metrics = eval_epoch(
        model, test_loader, device, args.label_type,
        metrics_space=args.metrics_space, log1p_transform=args.log1p_transform,
        pool_size=args.pool_size
    )

    log_print(f"\nTest Results:")
    log_print(f"  Test Loss:       {test_loss:.6f}")

    if args.label_type == 'peak':
        log_print(f"  Test AUROC:      {test_metrics.get('auroc', 0):.6f}")
        log_print(f"  Test AUPRC:      {test_metrics.get('auprc', 0):.6f}")
        history['test_auroc'] = test_metrics.get('auroc', 0)
        history['test_auprc'] = test_metrics.get('auprc', 0)
    else:
        log_print(f"  Test Pearson r:  {test_metrics.get('pearson_r', 0):.6f}")
        log_print(f"  Test Spearman r: {test_metrics.get('spearman_r', 0):.6f}")
        log_print(f"  Test MSE:        {test_metrics.get('mse', 0):.6f}")
        log_print(f"  Test R²:         {test_metrics.get('r2', 0):.6f}")
        history['test_pearson_r'] = test_metrics.get('pearson_r', 0)
        history['test_spearman_r'] = test_metrics.get('spearman_r', 0)
        history['test_mse'] = test_metrics.get('mse', 0)
        history['test_r2'] = test_metrics.get('r2', 0)

    history['test_loss'] = test_loss
    history['best_epoch'] = best_epoch
    history['best_val_loss'] = best_val_loss

    log_print(f"\n{'='*80}")
    log_print("Training History Summary:")
    log_print("─" * 120)

    if args.label_type == 'peak':
        header = (f"{'Epoch':<8} {'Train Loss':<12} {'Val Loss':<12} "
                  f"{'Tr AUROC':<10} {'Tr AUPRC':<10} "
                  f"{'Va AUROC':<10} {'Va AUPRC':<10}")
    else:
        header = (f"{'Epoch':<8} {'Train Loss':<12} {'Val Loss':<12} "
                  f"{'Tr Pr':<8} {'Tr Sr':<8} {'Tr MSE':<10} {'Tr R²':<8} "
                  f"{'Va Pr':<8} {'Va Sr':<8} {'Va MSE':<10} {'Va R²':<8}")

    log_print(header)
    log_print("─" * 120)

    for i in range(len(history['epoch'])):
        if args.label_type == 'peak':
            log_print(
                f"{history['epoch'][i]:<8} "
                f"{history['train_loss'][i]:<12.6f} "
                f"{history['val_loss'][i]:<12.6f} "
                f"{history['train_auroc'][i]:<10.6f} "
                f"{history['train_auprc'][i]:<10.6f} "
                f"{history['val_auroc'][i]:<10.6f} "
                f"{history['val_auprc'][i]:<10.6f}"
            )
        else:
            log_print(
                f"{history['epoch'][i]:<8} "
                f"{history['train_loss'][i]:<12.6f} "
                f"{history['val_loss'][i]:<12.6f} "
                f"{history['train_pearson_r'][i]:<8.6f} "
                f"{history['train_spearman_r'][i]:<8.6f} "
                f"{history['train_mse'][i]:<10.6f} "
                f"{history['train_r2'][i]:<8.6f} "
                f"{history['val_pearson_r'][i]:<8.6f} "
                f"{history['val_spearman_r'][i]:<8.6f} "
                f"{history['val_mse'][i]:<10.6f} "
                f"{history['val_r2'][i]:<8.6f}"
            )

    log_print("─" * 120)
    log_print(f"Best Epoch: {best_epoch} | Best Val Loss: {best_val_loss:.6f}")
    log_print(f"Test Loss:  {test_loss:.6f}", end="")
    if args.label_type == 'peak':
        log_print(f" | Test AUROC: {test_metrics.get('auroc', 0):.6f} | Test AUPRC: {test_metrics.get('auprc', 0):.6f}")
    else:
        log_print(
            f" | Test Pearson r: {test_metrics.get('pearson_r', 0):.6f} "
            f"| Test Spearman r: {test_metrics.get('spearman_r', 0):.6f} "
            f"| Test MSE: {test_metrics.get('mse', 0):.6f} "
            f"| Test R²: {test_metrics.get('r2', 0):.6f}"
        )

    history_file = os.path.join(args.output_dir, "training_history.json")
    with open(history_file, 'w') as f:
        json.dump(history, f, indent=2)
    log_print(f"\nTraining history saved to: {history_file}")

    sys.stdout = original_stdout


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Enformer model")
    parser.add_argument("--train_fasta", type=str, required=True,
                        help="Path to training FASTA file")
    parser.add_argument("--train_labels", type=str, required=True,
                        help="Path to training labels .npy file")
    parser.add_argument("--val_fasta", type=str, required=True,
                        help="Path to validation FASTA file")
    parser.add_argument("--val_labels", type=str, required=True,
                        help="Path to validation labels .npy file")
    parser.add_argument("--test_fasta", type=str, required=True,
                        help="Path to test FASTA file")
    parser.add_argument("--test_labels", type=str, required=True,
                        help="Path to test labels .npy file")
    parser.add_argument("--output_dir", type=str, default="output",
                        help="Directory to save outputs")
    parser.add_argument("--species_name", type=str, default="osa",
                        help="Species name for output head (default: osa)")
    parser.add_argument("--input_length", type=int, default=10000,
                        help="Input sequence length (default: 10000)")
    parser.add_argument("--pool_size", type=int, default=5,
                        help="Downsampling pool size (default: 5)")
    parser.add_argument("--num_conv_layers", type=int, default=7,
                        help="Number of conv tower layers (default: 7)")
    parser.add_argument("--batch_size", type=int, default=8,
                        help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4,
                        help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.01,
                        help="Weight decay")
    parser.add_argument("--dropout", type=float, default=0.4,
                        help="Dropout rate")
    parser.add_argument("--epochs", type=int, default=50,
                        help="Number of epochs")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--num_workers", type=int, default=4,
                        help="Number of data loader workers")
    parser.add_argument("--use_checkpointing", action="store_true",
                        help="Use gradient checkpointing")
    parser.add_argument("--label_type", type=str, default="peak",
                        choices=["peak", "coverage"],
                        help="Label type: 'peak' or 'coverage'")
    parser.add_argument("--coverage_loss", type=str, default="mse",
                        choices=["mse", "mae", "poisson"],
                        help="Loss function for coverage mode")
    parser.add_argument("--coverage_activation", type=str, default="relu",
                        choices=["relu", "softplus", "linear"],
                        help="Activation function for coverage mode")
    parser.add_argument("--log1p_transform", action="store_true", default=True,
                        help="Apply log1p transform to coverage labels (default: True)")
    parser.add_argument("--no_log1p_transform", dest="log1p_transform", action="store_false",
                        help="Disable log1p transform for coverage labels")
    parser.add_argument("--metrics_space", type=str, default="log1p",
                        choices=["log1p", "original"],
                        help="Space for computing metrics: 'log1p' (transformed) or 'original' (back-transformed). Default: log1p")

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    main(args)
