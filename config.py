from transformers import PretrainedConfig


class EnformerConfig(PretrainedConfig):
    model_type = "enformer"

    def __init__(
        self,
        dim=768,
        depth=8,
        heads=8,
        output_heads=dict(ath=3),
        target_length=256,
        input_length=1024,
        attn_dim_key=64,
        dropout_rate=0.4,
        attn_dropout=0.05,
        pos_dropout=0.01,
        use_checkpointing=False,
        pool_size=4,
        num_conv_layers=7,
        dim_divisible_by=128,
        use_tf_gamma=False,
        tf_gamma_path=None,
        label_type='coverage',
        coverage_loss='mse',
        coverage_activation='relu',
        log1p_transform=True,
        **kwargs,
    ):
        self.dim = dim
        self.depth = depth
        self.heads = heads
        self.output_heads = output_heads
        self.target_length = target_length
        self.input_length = input_length
        self.attn_dim_key = attn_dim_key
        self.dropout_rate = dropout_rate
        self.attn_dropout = attn_dropout
        self.pos_dropout = pos_dropout
        self.use_checkpointing = use_checkpointing
        self.pool_size = pool_size
        self.num_conv_layers = num_conv_layers
        self.dim_divisible_by = dim_divisible_by
        self.use_tf_gamma = use_tf_gamma
        self.tf_gamma_path = tf_gamma_path
        self.label_type = label_type
        self.coverage_loss = coverage_loss
        self.coverage_activation = coverage_activation
        self.log1p_transform = log1p_transform
        super().__init__(**kwargs)