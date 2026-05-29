import math
import torch
import numpy as np
from torch import nn, einsum
import torch.nn.functional as F
from einops import rearrange, reduce
from einops.layers.torch import Rearrange
from torch.utils.checkpoint import checkpoint_sequential
from config import EnformerConfig
from transformers import PreTrainedModel


TF_GAMMAS = None


def exists(val):
    return val is not None


def default(val, d):
    return val if exists(val) else d


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x


class GELU(nn.Module):
    def forward(self, x):
        return torch.sigmoid(1.702 * x) * x


class AttentionPool(nn.Module):
    def __init__(self, dim, pool_size=2):
        super().__init__()
        self.pool_size = pool_size
        self.pool_fn = Rearrange('b d (n p) -> b d n p', p=pool_size)
        self.to_attn_logits = nn.Conv2d(dim, dim, 1, bias=False)
        nn.init.dirac_(self.to_attn_logits.weight)
        with torch.no_grad():
            self.to_attn_logits.weight.mul_(2)

    def forward(self, x):
        b, _, n = x.shape
        remainder = n % self.pool_size
        pad_right = (self.pool_size - remainder) if remainder > 0 else 0
        if pad_right > 0:
            x = F.pad(x, (0, pad_right), value=0)
            mask = torch.zeros((b, 1, n), dtype=torch.bool, device=x.device)
            mask = F.pad(mask, (0, pad_right), value=True)

        x = self.pool_fn(x)
        logits = self.to_attn_logits(x)

        if remainder > 0:
            mask_value = -torch.finfo(logits.dtype).max
            logits = logits.masked_fill(self.pool_fn(mask), mask_value)

        attn = logits.softmax(dim=-1)
        return (x * attn).sum(dim=-1)


class ConvBlock(nn.Module):
    def __init__(self, dim, dim_out=None, kernel_size=1):
        super().__init__()
        dim_out = default(dim_out, dim)
        self.net = nn.Sequential(
            nn.BatchNorm1d(dim),
            GELU(),
            nn.Conv1d(dim, dim_out, kernel_size, padding=kernel_size // 2)
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim, *, num_rel_pos_features, heads=8, dim_key=64, dim_value=64,
                 dropout=0., pos_dropout=0., use_tf_gamma=False):
        super().__init__()
        self.scale = dim_key ** -0.5
        self.heads = heads

        self.to_q = nn.Linear(dim, dim_key * heads, bias=False)
        self.to_k = nn.Linear(dim, dim_key * heads, bias=False)
        self.to_v = nn.Linear(dim, dim_value * heads, bias=False)
        self.to_out = nn.Linear(dim_value * heads, dim)

        nn.init.zeros_(self.to_out.weight)
        nn.init.zeros_(self.to_out.bias)

        self.num_rel_pos_features = num_rel_pos_features
        self.to_rel_k = nn.Linear(num_rel_pos_features, dim_key * heads, bias=False)
        self.rel_content_bias = nn.Parameter(torch.randn(1, heads, 1, dim_key))
        self.rel_pos_bias = nn.Parameter(torch.randn(1, heads, 1, dim_key))

        self.pos_dropout = nn.Dropout(pos_dropout)
        self.attn_dropout = nn.Dropout(dropout)
        self.use_tf_gamma = use_tf_gamma

    def forward(self, x):
        n, h, device = x.shape[-2], self.heads, x.device
        q = self.to_q(x)
        k = self.to_k(x)
        v = self.to_v(x)

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), (q, k, v))
        q = q * self.scale

        content_logits = einsum('b h i d, b h j d -> b h i j', q + self.rel_content_bias, k)

        positions = get_positional_embed(
            n,
            self.num_rel_pos_features,
            device,
            use_tf_gamma=self.use_tf_gamma,
            dtype=self.to_rel_k.weight.dtype
        )
        positions = self.pos_dropout(positions)
        rel_k = self.to_rel_k(positions)
        rel_k = rearrange(rel_k, 'n (h d) -> h n d', h=h)
        rel_logits = einsum('b h i d, h j d -> b h i j', q + self.rel_pos_bias, rel_k)
        rel_logits = relative_shift(rel_logits)

        logits = content_logits + rel_logits
        attn = logits.softmax(dim=-1)
        attn = self.attn_dropout(attn)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


def relative_shift(x):
    to_pad = torch.zeros_like(x[..., :1])
    x = torch.cat((to_pad, x), dim=-1)
    _, h, t1, t2 = x.shape
    x = x.reshape(-1, h, t2, t1)
    x = x[:, :, 1:, :]
    x = x.reshape(-1, h, t1, t2 - 1)
    return x[..., :((t2 + 1) // 2)]


def get_positional_embed(seq_len, feature_size, device, use_tf_gamma, dtype=torch.float32):
    distances = torch.arange(-seq_len + 1, seq_len, device=device)

    def get_tf_gamma(*args, **kwargs):
        return TF_GAMMAS.to(device) if use_tf_gamma else get_positional_features_gamma(*args, **kwargs)

    feature_fns = [
        get_positional_features_exponential,
        get_positional_features_central_mask,
        get_tf_gamma
    ]

    num_components = len(feature_fns) * 2
    if feature_size % num_components != 0:
        raise ValueError(f'feature size must be divisible by {num_components}')

    num_basis_per_class = feature_size // num_components
    embeddings = []
    for fn in feature_fns:
        try:
            emb = fn(distances, num_basis_per_class, seq_len, dtype=dtype)
        except TypeError:
            emb = fn(distances, num_basis_per_class, seq_len)
        embeddings.append(emb)

    embeddings = torch.cat(embeddings, dim=-1)
    embeddings = torch.cat((embeddings, torch.sign(distances)[..., None] * embeddings), dim=-1)
    return embeddings.to(dtype)


def get_positional_features_exponential(positions, features, seq_len, min_half_life=3., dtype=torch.float32):
    max_range = math.log(seq_len) / math.log(2.)
    half_life = 2 ** torch.linspace(min_half_life, max_range, features, device=positions.device)
    half_life = half_life[None, ...]
    positions = positions.abs()[..., None]
    return torch.exp(-math.log(2.) / half_life * positions)


def get_positional_features_central_mask(positions, features, seq_len, dtype=torch.float32):
    center_widths = 2 ** torch.arange(1, features + 1, device=positions.device).float()
    center_widths = center_widths - 1
    return (center_widths[None, ...] > positions.abs()[..., None]).float()


def get_positional_features_gamma(positions, features, seq_len, stddev=None, start_mean=None, eps=1e-8,
                                  dtype=torch.float32):
    stddev = default(stddev, seq_len / (2 * features))
    start_mean = default(start_mean, seq_len / features)

    mean = torch.linspace(start_mean, seq_len, features, device=positions.device)
    mean = mean[None, ...]
    concentration = (mean / stddev) ** 2
    rate = mean / stddev ** 2

    probabilities = gamma_pdf(positions.float().abs()[..., None], concentration, rate)
    probabilities = probabilities + eps
    outputs = probabilities / torch.amax(probabilities, dim=-1, keepdim=True)
    return outputs


def gamma_pdf(x, concentration, rate):
    log_unnormalized_prob = torch.xlogy(concentration - 1., x) - rate * x
    log_normalization = (torch.lgamma(concentration) - concentration * torch.log(rate))
    return torch.exp(log_unnormalized_prob - log_normalization)


class WeightedFocalLoss(nn.Module):
    def __init__(self, alpha=0.75, gamma=2):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, pred_logits, target):
        bce_loss = F.binary_cross_entropy_with_logits(pred_logits, target, reduction='none')
        pt = torch.exp(-bce_loss)

        alpha_t = torch.where(target >= 0.5, self.alpha, 1 - self.alpha)
        loss = alpha_t * (1 - pt) ** self.gamma * bce_loss
        return loss.mean()


class MixedLoss(nn.Module):
    def __init__(self, alpha=0.5, gamma=2, bce_weight=1.0, focal_weight=1.0):
        super().__init__()
        self.bce_weight = bce_weight
        self.focal_weight = focal_weight
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.focal_loss = WeightedFocalLoss(gamma=gamma)

    def forward(self, pred_logits, target):
        bce = self.bce_loss(pred_logits, target)
        focal = self.focal_loss(pred_logits, target)
        return self.bce_weight * bce + self.focal_weight * focal


class Enformer(PreTrainedModel):
    config_class = EnformerConfig
    base_model_prefix = "enformer"

    def __init__(self, config):
        super().__init__(config)
        self.dim = config.dim
        half_dim = config.dim // 2
        twice_dim = config.dim * 2
        self.autocast_enabled = True
        self.autocast_device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.label_type = config.label_type
        self.coverage_loss = config.coverage_loss
        self.coverage_activation = config.coverage_activation

        # ========== Stem: 1层Conv + 1层AttentionPool完成下采样 ==========
        # pool_size=5: 10000 -> 2000
        self.stem = nn.Sequential(
            nn.Conv1d(4, half_dim, 15, padding=7),
            Residual(ConvBlock(half_dim)),
            AttentionPool(half_dim, pool_size=config.pool_size)
        )

        # ========== Conv Tower: 通道变换，不做空间下采样(pool_size=1) ==========
        filter_list = exponential_linspace_int(
            half_dim, config.dim,
            num=config.num_conv_layers,
            divisible_by=config.dim_divisible_by
        )
        filter_list = [half_dim, *filter_list]

        conv_layers = []
        for dim_in, dim_out in zip(filter_list[:-1], filter_list[1:]):
            conv_layers.append(nn.Sequential(
                ConvBlock(dim_in, dim_out, kernel_size=5),
                Residual(ConvBlock(dim_out, dim_out, 1)),
                AttentionPool(dim_out, pool_size=1)
            ))

        self.conv_tower = nn.Sequential(*conv_layers)

        # ========== Transformer ==========
        transformer = []
        for _ in range(config.depth):
            transformer.append(nn.Sequential(
                Residual(nn.Sequential(
                    nn.LayerNorm(config.dim),
                    Attention(
                        config.dim,
                        heads=config.heads,
                        dim_key=config.attn_dim_key,
                        dim_value=config.dim // config.heads,
                        dropout=config.attn_dropout,
                        pos_dropout=config.pos_dropout,
                        num_rel_pos_features=config.dim // config.heads,
                        use_tf_gamma=config.use_tf_gamma
                    ),
                    nn.Dropout(config.dropout_rate)
                )),
                Residual(nn.Sequential(
                    nn.LayerNorm(config.dim),
                    nn.Linear(config.dim, config.dim * 2),
                    nn.Dropout(config.dropout_rate),
                    nn.ReLU(),
                    nn.Linear(config.dim * 2, config.dim),
                    nn.Dropout(config.dropout_rate)
                ))
            ))

        self.transformer = nn.Sequential(*transformer)

        # ========== 最终逐点卷积 ==========
        self.final_pointwise = nn.Sequential(
            Rearrange('b n d -> b d n'),
            ConvBlock(filter_list[-1], twice_dim, 1),
            Rearrange('b d n -> b n d'),
            nn.Dropout(config.dropout_rate / 8),
            GELU()
        )

        # ========== 输出头：单物种，通道数可配置 ==========
        self.output_heads = config.output_heads
        self._heads = nn.ModuleDict({
            name: nn.Linear(twice_dim, num_tracks)
            for name, num_tracks in config.output_heads.items()
        })

        # ========== 损失函数 ==========
        if self.label_type == 'peak':
            self.loss_fn = MixedLoss(alpha=0.5, gamma=2, bce_weight=1.0, focal_weight=1.0)
        else:
            self.loss_fn = None

        self.use_checkpointing = config.use_checkpointing

    def _apply_output_activation(self, logits):
        if self.label_type == 'peak':
            return torch.sigmoid(logits)
        else:
            if self.coverage_activation == 'relu':
                return torch.relu(logits)
            elif self.coverage_activation == 'softplus':
                return F.softplus(logits)
            else:
                return logits

    def _generate_target_masks(self, seq_lens, target_length):
        masks = []
        for seq_len in seq_lens:
            valid_target_len = seq_len // self.config.pool_size
            mask = torch.zeros(target_length, dtype=torch.float32, device=self.device)
            mask[:valid_target_len] = 1.0
            masks.append(mask)
        return torch.stack(masks).unsqueeze(-1)

    def _compute_masked_loss(self, logits, target, target_masks):
        expanded_masks = target_masks.expand_as(logits)
        num_valid = expanded_masks.sum()

        if self.label_type == 'peak':
            bce_per_element = F.binary_cross_entropy_with_logits(logits, target, reduction='none')
            masked_bce = (bce_per_element * expanded_masks).sum() / num_valid

            pt = torch.exp(-bce_per_element)
            alpha = self.loss_fn.focal_loss.alpha
            gamma = self.loss_fn.focal_loss.gamma
            alpha_t = torch.where(target >= 0.5, alpha, 1 - alpha)
            focal_per_element = alpha_t * (1 - pt) ** gamma * bce_per_element
            masked_focal = (focal_per_element * expanded_masks).sum() / num_valid

            return self.loss_fn.bce_weight * masked_bce + self.loss_fn.focal_weight * masked_focal
        else:
            if self.coverage_loss == 'poisson':
                nll_per_element = F.poisson_nll_loss(logits, target, log_input=False, reduction='none')
                return (nll_per_element * expanded_masks).sum() / num_valid
            else:
                diff = (logits - target) * expanded_masks
                if self.coverage_loss == 'mse':
                    return (diff ** 2).sum() / num_valid
                elif self.coverage_loss == 'mae':
                    return diff.abs().sum() / num_valid
                else:
                    return (diff ** 2).sum() / num_valid

    def forward(self, x, target=None, return_embeddings=False, seq_lens=None):
        """
        Args:
            x: 输入序列 [batch, seq_len, 4] 或 [batch, seq_len]
            target: 标签 [batch, target_len, num_channels]，可选
            return_embeddings: 是否返回中间特征
            seq_lens: 每条序列的真实长度（变长序列支持），可选
        Returns:
            如果 target 为 None: pred_dict (各 head 的激活后预测)
            如果 target 不为 None: (loss, pred_dict) 元组
        """
        with torch.amp.autocast(self.autocast_device, enabled=self.autocast_enabled):
            if isinstance(x, list):
                x = str_to_one_hot(x)
            elif isinstance(x, torch.Tensor) and x.dtype == torch.long:
                x = seq_indices_to_one_hot(x)

            x = x.to(self.device)

            no_batch = x.ndim == 2
            if no_batch:
                x = rearrange(x, '... -> () ...')

            trunk_fn = self.trunk_checkpointed if self.use_checkpointing else self._trunk
            features = trunk_fn(x)

            if no_batch:
                features = rearrange(features, '() ... -> ...')

            if return_embeddings:
                return features

            logits_dict = {}
            for name, head_module in self._heads.items():
                logits_dict[name] = head_module(features)

            pred_dict = {}
            for name, logits in logits_dict.items():
                pred_dict[name] = self._apply_output_activation(logits)

            if target is not None:
                target_masks = None
                if seq_lens is not None:
                    target_masks = self._generate_target_masks(seq_lens, target.shape[1])

                total_loss = 0.0
                for name, logits in logits_dict.items():
                    if target_masks is not None:
                        total_loss += self._compute_masked_loss(logits, target, target_masks)
                    else:
                        if self.label_type == 'peak':
                            total_loss += self.loss_fn(logits, target)
                        else:
                            if self.coverage_loss == 'mse':
                                total_loss += F.mse_loss(logits, target)
                            elif self.coverage_loss == 'mae':
                                total_loss += F.l1_loss(logits, target)
                            elif self.coverage_loss == 'poisson':
                                total_loss += F.poisson_nll_loss(logits, target, log_input=False)
                            else:
                                total_loss += F.mse_loss(logits, target)
                loss = total_loss / len(logits_dict)
                return loss, pred_dict

            return pred_dict

    @property
    def _trunk(self):
        return nn.Sequential(
            Rearrange('b n d -> b d n'),
            self.stem,
            self.conv_tower,
            Rearrange('b d n -> b n d'),
            self.transformer,
            self.final_pointwise
        )

    def trunk_checkpointed(self, x):
        x = rearrange(x, 'b n d -> b d n')
        x = self.stem(x)
        x = self.conv_tower(x)
        x = rearrange(x, 'b d n -> b n d')
        x = checkpoint_sequential(self.transformer, len(self.transformer), x, use_reentrant=False)
        x = self.final_pointwise(x)
        return x


def exponential_linspace_int(start, end, num, divisible_by=1):
    def _round(x):
        return int(round(x / divisible_by) * divisible_by)

    base = math.exp(math.log(end / start) / (num - 1))
    return [_round(start * base ** i) for i in range(num)]


def str_to_one_hot(seq_strs):
    batched = not isinstance(seq_strs, str)
    seq_strs = [seq_strs] if not batched else seq_strs

    one_hot_embed = torch.zeros(256, 4)
    one_hot_embed[ord('a')] = torch.tensor([1., 0., 0., 0.])
    one_hot_embed[ord('c')] = torch.tensor([0., 1., 0., 0.])
    one_hot_embed[ord('g')] = torch.tensor([0., 0., 1., 0.])
    one_hot_embed[ord('t')] = torch.tensor([0., 0., 0., 1.])
    one_hot_embed[ord('n')] = torch.tensor([0., 0., 0., 0.])
    one_hot_embed[ord('A')] = torch.tensor([1., 0., 0., 0.])
    one_hot_embed[ord('C')] = torch.tensor([0., 1., 0., 0.])
    one_hot_embed[ord('G')] = torch.tensor([0., 0., 1., 0.])
    one_hot_embed[ord('T')] = torch.tensor([0., 0., 0., 1.])
    one_hot_embed[ord('N')] = torch.tensor([0., 0., 0., 0.])
    one_hot_embed[ord('.')] = torch.tensor([0.25, 0.25, 0.25, 0.25])

    seq_chrs = [torch.from_numpy(np.frombuffer(seq.encode(), dtype=np.uint8)) for seq in seq_strs]
    seq_chrs = torch.stack(seq_chrs) if batched else seq_chrs[0]
    return one_hot_embed[seq_chrs.long()]


def seq_indices_to_one_hot(t, padding=-1):
    is_padding = t == padding
    t = t.clamp(min=0)
    one_hot = F.one_hot(t, num_classes=5)
    out = one_hot[..., :4].float()
    out = out.masked_fill(is_padding[..., None], 0.25)
    return out