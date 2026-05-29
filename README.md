# Enformer — PyTorch Implementation

## 1. Project Overview

This project is a PyTorch reimplementation of the Enformer model.

Enformer is a deep learning model capable of predicting gene expression and other regulatory functions from DNA sequences, capturing long-range interactions via attention mechanisms. This project introduces several custom modifications to the original Enformer architecture, including: shortened input sequence length, simplified downsampling mechanism, single-species training support, and flexible output channel configuration.

**Core capability: given a DNA sequence, output predicted signals across multiple regulatory tracks (e.g., binary Peak classification probabilities or continuous Coverage values).**

**This project supports two label types:**

| Label Type | Task Type | Label Form | Loss Function | Evaluation Metrics |
|------------|-----------|------------|---------------|-------------------|
| `peak` | Binary classification | 0/1 | BCE + Focal Loss | AUROC, AUPRC |
| `coverage` | Regression | Continuous values | MSE/MAE/Poisson | Pearson r, Spearman r, MSE, R² |

---

## 2. Model Architecture Overview

```
Input sequence: DNA sequence string or one-hot encoding [B, input_length, 4]
    ↓
[Stem]      Conv1d(4→768, k=15) + ConvBlock + AttentionPool(pool_size)
            → Downsampling: input_length → input_length/pool_size
    ↓
[Conv Tower] N layers of ConvBlock + Residual + AttentionPool(pool_size=1)
            → Channel transformation: 768 → 1536 (length unchanged)
    ↓
[Transformer] 11 layers of Multi-Head Attention + FFN
            → Sequence modeling (length unchanged)
    ↓
[Final Pointwise] ConvBlock(1536→3072)
    ↓
[Output Head] Linear(3072 → num_channels)
    ↓
Output: [B, target_length, num_channels]
```

**Key design:** Downsampling is performed only once in the Stem via AttentionPool (`pool_size` is configurable). The subsequent Conv Tower and Transformer do not perform any spatial downsampling, only channel-wise feature extraction.

---

## 3. Data Format (Unified Standard Format)

### 3.1 Format Requirements

This project uses a unified standard format. Two types of files are required:

**Sequence file (FASTA format)**
- Each entry (starting with `>`) corresponds to one complete input sequence
- **Variable-length sequences are supported**: different entries may have different sequence lengths; the model automatically applies dynamic padding to the maximum length within each batch
- Sequence length must be divisible by `pool_size` (default `pool_size=5`, i.e., sequence length should be a multiple of 5)
- Recommended sequence length range: 1000–100000 bp; sequences that are too short may lack sufficient information, while sequences that are too long will consume more GPU memory

**Label file (.npy format, NumPy archive)**
- Shape: `[N, target_length, num_channels]`
  - `N`: number of sequences, matching the number of entries in the FASTA file
  - `target_length`: label length after downsampling; should be ≥ `max_seq_len / pool_size` across all sequences (label positions corresponding to shorter sequences should be padded with 0)
  - `num_channels`: number of signal tracks
- Data type: `float32`
- For variable-length sequences: the `target_length` of labels should equal `max_seq_len / pool_size` for the longest sequence; labels for shorter sequences should be padded with 0 beyond their effective length (the model automatically generates masks based on sequence lengths to exclude padded regions from loss and metric computation)

**Naming example:**
```
sequences.fa          # FASTA sequence file
labels.npy           # Label file, shape [N, 2000, 2]
```

### 3.2 Data Preparation Pipeline

#### Step 1: Convert raw data to standard format (optional, only needed if you have .h5 files)

If you already have `.h5` format data, use `convert_h5_to_standard.py` to convert it to standard format:

```bash
python convert_h5_to_standard.py \
    --h5_file dataset_oryza_sativa_TEST_only1.h5 \
    --output_dir standard_data \
    --output_prefix oryza_sativa
```

This script outputs:
- `standard_data/oryza_sativa.fa`: FASTA sequence file
- `standard_data/oryza_sativa_labels.npy`: label file, shape `[N, target_length, num_channels]`

#### Step 2: Split into training / validation / test sets

```bash
python split_data.py \
    --fasta_file standard_data/oryza_sativa.fa \
    --labels_file standard_data/oryza_sativa_labels.npy \
    --output_dir processed_data \
    --output_prefix oryza_sativa \
    --split_ratios 0.7 0.15 0.15 \
    --split_method random \
    --seed 42
```

This script outputs 6 files:
```
processed_data/oryza_sativa_train.fa
processed_data/oryza_sativa_train_labels.npy
processed_data/oryza_sativa_val.fa
processed_data/oryza_sativa_val_labels.npy
processed_data/oryza_sativa_test.fa
processed_data/oryza_sativa_test_labels.npy
```

If you already have FASTA and .npy files in the required format, proceed directly to Step 2 for splitting.

---

## 4. Detailed Usage Instructions

### 4.1 Install Dependencies

```bash
pip install -r requirements.txt
```

### 4.2 Training the Model

```bash
python train.py \
    --train_fasta processed_data/oryza_sativa_train.fa \
    --train_labels processed_data/oryza_sativa_train_labels.npy \
    --val_fasta processed_data/oryza_sativa_val.fa \
    --val_labels processed_data/oryza_sativa_val_labels.npy \
    --test_fasta processed_data/oryza_sativa_test.fa \
    --test_labels processed_data/oryza_sativa_test_labels.npy \
    --species_name osa \
    --input_length 10000 \
    --pool_size 5 \
    --num_conv_layers 7 \
    --label_type peak \
    --batch_size 8 \
    --epochs 50 \
    --lr 1e-4 \
    --output_dir output_peak
```

**Parameter descriptions:**

| Parameter | Description |
|-----------|-------------|
| `--train_fasta` / `--train_labels` | Training set FASTA sequence file and label .npy file (required) |
| `--val_fasta` / `--val_labels` | Validation set FASTA sequence file and label .npy file (required) |
| `--test_fasta` / `--test_labels` | Test set FASTA sequence file and label .npy file (required) |
| `--species_name` | Species name, used for output head naming (default: osa) |
| `--input_length` | Input DNA sequence length (for reference; variable-length input is actually supported, default: 10000) |
| `--pool_size` | Downsampling ratio; final output length = `input_length / pool_size` (default: 5) |
| `--num_conv_layers` | Number of Conv Tower convolutional layers (default: 7) |
| `--batch_size` | Batch size (default: 8) |
| `--lr` | Initial learning rate (default: 1e-4) |
| `--weight_decay` | Weight decay (default: 0.01) |
| `--dropout` | Dropout rate (default: 0.4) |
| `--epochs` | Number of training epochs (default: 50) |
| `--seed` | Random seed (default: 42) |
| `--num_workers` | Number of DataLoader worker processes (default: 4) |
| `--use_checkpointing` | Enable gradient checkpointing to save GPU memory |
| `--label_type` | Label type: `peak` (binary classification) or `coverage` (regression) (default: peak) |
| `--coverage_loss` | Coverage mode loss function: `mse` / `mae` / `poisson` (default: mse) |
| `--coverage_activation` | Coverage mode activation function: `relu` / `softplus` / `linear` (default: relu) |
| `--log1p_transform` | Apply log1p transform to Coverage labels (default: enabled; strongly recommended to keep enabled) |
| `--no_log1p_transform` | Disable log1p transform (not recommended unless labels are preprocessed) |
| `--metrics_space` | Metrics computation space: `log1p` (in transformed space, default) or `original` (back-transformed to original space) |
| `--output_dir` | Output directory (default: output) |

**Training outputs:**
- `output_dir/best_model.pt`: best model checkpoint on the validation set
- `output_dir/checkpoint_epoch_N.pt`: checkpoint saved every 5 epochs
- `output_dir/training_log.txt`: complete training log
- `output_dir/training_history.json`: training history (loss, metrics, etc.)

### 4.3 Testing the Model

```bash
python test.py \
    --model_path output_peak/best_model.pt \
    --test_fasta processed_data/oryza_sativa_test.fa \
    --test_labels processed_data/oryza_sativa_test_labels.npy \
    --batch_size 8 \
    --output_file results.json
```

**Parameter descriptions:**

| Parameter | Description |
|-----------|-------------|
| `--model_path` | Path to model checkpoint (required) |
| `--test_fasta` | Test set FASTA sequence file (required) |
| `--test_labels` | Test set label .npy file (required) |
| `--batch_size` | Batch size (default: 8) |
| `--num_workers` | Number of DataLoader worker processes (default: 4) |
| `--label_type` | Label type (inferred from model config if not specified) |
| `--metrics_space` | Metrics computation space: `log1p` (in transformed space, default) or `original` (back-transformed to original space) |
| `--output_file` | Path to save results JSON (optional) |

**Test output:** test loss + AUROC/AUPRC (peak mode) or Pearson r/Spearman r/MSE/R² (coverage mode).

---

## 5. Parameter Configuration (config.py)

All model parameters are centrally defined in the `EnformerConfig` class within `config.py`. They can also be specified via command-line arguments:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `dim` | 1536 | Transformer feature dimension (channel dimension) |
| `depth` | 11 | Number of Transformer layers |
| `heads` | 8 | Number of multi-head attention heads |
| `output_heads` | `{'osa': 2}` | Output head configuration: `{species_name: num_channels}`; **the number of channels is automatically determined by the label .npy file; this serves only as a fallback** |
| `target_length` | 2000 | Label length (= `input_length / pool_size`), overwritten by the label file during training |
| `input_length` | 10000 | Input DNA sequence length (reference value; variable-length input is actually supported) |
| `attn_dim_key` | 64 | Dimension of attention keys |
| `dropout_rate` | 0.4 | Dropout rate |
| `attn_dropout` | 0.05 | Dropout rate for attention weights |
| `pos_dropout` | 0.01 | Dropout rate for positional encodings |
| `use_checkpointing` | False | Whether to use gradient checkpointing in Transformer layers |
| `pool_size` | 5 | **Downsampling ratio**: every `pool_size` positions are aggregated into 1; output length = `input_length / pool_size` |
| `num_conv_layers` | 7 | Number of Conv Tower convolutional blocks |
| `dim_divisible_by` | 128 | Channel dimension alignment constraint |
| `use_tf_gamma` | False | Whether to use TensorFlow Gamma positional encoding |
| `tf_gamma_path` | None | Path to TensorFlow Gamma weight file |
| `label_type` | 'peak' | Label type: `'peak'` (binary classification) or `'coverage'` (regression) |
| `coverage_loss` | 'mse' | Coverage mode loss function: `'mse'` / `'mae'` / `'poisson'` |
| `coverage_activation` | 'relu' | Coverage mode activation function: `'relu'` / `'softplus'` / `'linear'` |
| `log1p_transform` | True | Whether to apply log1p transform to Coverage labels (strongly recommended to keep enabled) |

**How to modify the downsampling ratio:**
Modify `pool_size` in `config.py`. For example, changing `pool_size=5` to `pool_size=2` changes the output length from 2000 to 5000 (provided that `input_length` is divisible by the new value).

**How to modify the number of output channels:**
The number of output channels is automatically determined from the last dimension of the label `.npy` file during training. The `output_heads` in `config.py` serves only as a fallback configuration; in practice, it is overridden by the label file in the training script.

---

## 6. Detailed Model Architecture

### 6.1 Overall Data Flow and Tensor Shapes

Using `input_length=10000`, `pool_size=5`, `dim=1536`, `num_conv_layers=7`, batch_size=2 as an example:

```
Input:                    [B, 10000, 4]
    ↓ Rearrange('b n d -> b d n')
Stem-Conv1d:             [B, 768, 10000]
Stem-ConvBlock:           [B, 768, 10000]
Stem-AttentionPool(5):    [B, 768, 2000]       ← spatial downsampling, 10000→2000

ConvTower-Layer1 (pool_size=1): [B, c1, 2000]
ConvTower-Layer2 (pool_size=1): [B, c2, 2000]
... (7 layers total)
ConvTower-Layer7 (pool_size=1): [B, 1536, 2000]  ← length unchanged, channels increased to 1536

    ↓ Rearrange('b d n -> b n d')
Transformer × 11:         [B, 2000, 1536]      ← length unchanged

    ↓ Rearrange('b n d -> b d n')
FinalPointwise:           [B, 3072, 2000]
    ↓ Rearrange('b d n -> b n d')
OutputHead:               [B, 2000, num_channels]
```

### 6.2 Layer Details

#### Input Encoding

Input is one-hot encoded as `[B, input_length, 4]`, with channel order: A(0), C(1), G(2), T(3). Non-standard characters (e.g., N) are filled with 0.25.

#### Stem

```python
self.stem = nn.Sequential(
    nn.Conv1d(4, 768, kernel_size=15, padding=7),   # local feature extraction
    Residual(ConvBlock(768)),                          # residual convolution block
    AttentionPool(768, pool_size=config.pool_size)    # attention-based pooling downsampling
)
```

| Operation | Input Shape | Output Shape |
|-----------|-------------|--------------|
| Conv1d(4→768, k=15) | `[B, 4, 10000]` | `[B, 768, 10000]` |
| ConvBlock (BN→GELU→Conv1d) | `[B, 768, 10000]` | `[B, 768, 10000]` |
| AttentionPool(pool_size=5) | `[B, 768, 10000]` | `[B, 768, 2000]` |

**How AttentionPool works:**
1. Groups the sequence by `pool_size` (e.g., every 5 consecutive positions form one group)
2. Learns attention weights for the positions within each group
3. Outputs one value per group via weighted summation

This is the downsampling mechanism: rather than simple average pooling, it is **learnable attention-weighted aggregation**. `pool_size=5` means every 5 bp are aggregated into 1 output position.

#### Conv Tower

```python
filter_list = exponential_linspace_int(768, 1536, num=7)
# Results approximately: [768, 896, 1024, 1152, 1280, 1408, 1536]

for dim_in, dim_out in zip(filter_list[:-1], filter_list[1:]):
    conv_layers.append(nn.Sequential(
        ConvBlock(dim_in, dim_out, kernel_size=5),  # channel transformation
        Residual(ConvBlock(dim_out, dim_out, 1)),   # residual connection
        AttentionPool(dim_out, pool_size=1)          # no downsampling
    ))
```

**Key point: `pool_size=1` for all layers, meaning the sequence length remains 2000 throughout**, performing only channel-wise feature extraction.

#### Transformer Encoder

11 layers of standard Transformer blocks:
- Multi-Head Self-Attention (8 heads, key_dim=64)
- FFN: `Linear(1536 → 3072 → 1536)`
- Relative positional encoding (exponential decay + center mask + Gamma distribution)

The sequence length remains fixed at 2000 throughout.

#### Final Output Head

```python
self._heads = nn.ModuleDict({
    name: nn.Linear(3072, num_tracks)
    for name, num_tracks in config.output_heads.items()
})
```

- Input: `[B, 2000, 3072]` (high-dimensional features from Transformer output)
- Output: `[B, 2000, num_channels]` (num_channels is automatically determined by the label file)

---

## 7. Complete Pipeline Example

```bash
# Step 1: Convert H5 data to standard format
python convert_h5_to_standard.py \
    --h5_file dataset_oryza_sativa_TEST_only1.h5 \
    --output_dir standard_data \
    --output_prefix oryza_sativa

# Step 2: Split into train/val/test sets
python split_data.py \
    --fasta_file standard_data/oryza_sativa.fa \
    --labels_file standard_data/oryza_sativa_labels.npy \
    --output_dir processed_data \
    --output_prefix oryza_sativa \
    --split_ratios 0.7 0.15 0.15 \
    --split_method random \
    --seed 42

# Step 3: Training (Peak label type)
python train.py \
    --train_fasta processed_data/oryza_sativa_train.fa \
    --train_labels processed_data/oryza_sativa_train_labels.npy \
    --val_fasta processed_data/oryza_sativa_val.fa \
    --val_labels processed_data/oryza_sativa_val_labels.npy \
    --test_fasta processed_data/oryza_sativa_test.fa \
    --test_labels processed_data/oryza_sativa_test_labels.npy \
    --species_name osa \
    --input_length 10000 \
    --pool_size 5 \
    --label_type peak \
    --batch_size 8 \
    --epochs 50 \
    --lr 1e-4 \
    --output_dir output_peak

# Step 4: Testing
python test.py \
    --model_path output_peak/best_model.pt \
    --test_fasta processed_data/oryza_sativa_test.fa \
    --test_labels processed_data/oryza_sativa_test_labels.npy \
    --batch_size 8
```

---

## 8. Frequently Asked Questions

**Q1: How to modify the downsampling ratio?**

Modify the `pool_size` value in `config.py`, and adjust `input_length` accordingly to ensure it is divisible by `pool_size`. For example: `pool_size=2` + `input_length=10000` → output length 5000.

**Q2: How to add new sequencing data tracks?**

During data processing (in `convert_h5_to_standard.py` or via custom assembly), stack all tracks into the third dimension of the label `.npy` file. The number of channels is automatically read from the label file during training; no model code modification is needed.

**Q3: What to do if GPU memory is insufficient?**

Suggestions:
1. Reduce `batch_size` (e.g., to 2)
2. Enable `--use_checkpointing` (gradient checkpointing)
3. Reduce `dim` or `depth`

**Q4: How to accommodate different input sequence lengths?**

This project supports variable-length sequence input. Entries in the FASTA file may have different sequence lengths; the model automatically applies dynamic padding within each batch. The only requirement is that each sequence length be divisible by `pool_size`. The `--input_length` parameter is for reference only and does not affect actual training.

**Q5: Do padded regions affect training when using variable-length sequences?**

No. The model tracks the true length of each sequence via the `seq_lens` parameter, automatically generates masks, and excludes padded regions from loss and metric computation. Padded portions of shorter sequences do not contribute to gradients.

**Q6: What to do if the Coverage loss is very large (hundreds of thousands)?**

Coverage labels are typically raw count values with enormous dynamic range. The log1p transform is enabled by default. **Metrics are computed in log1p-transformed space by default** (`--metrics_space log1p`); you may switch to original-space computation via `--metrics_space original`. **Ensure that labels are not double log1p-transformed.**

---

## 9. Citation

If you use this code, please cite the original Enformer paper:

> Avsec, Ž., Agarwal, V., Visentin, D. et al. Effective gene expression prediction from sequence by integrating long-range interactions. *Nat Methods* 18, 1196–1203 (2021). https://doi.org/10.1038/s41592-021-01252-x
