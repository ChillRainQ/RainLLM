from transformers import PretrainedConfig


class RainLLMConfig(PretrainedConfig):
    # dim: int = 512  # dim // n_attention_heads // 2 应该为一个正整数
    # n_layers: int = 16
    # max_seq_len: int = 8192
    # vocab_size: int = 6400
    # n_attention_heads: int = 8
    # n_kv_heads: int = 2
    #
    # # default or flash
    # attention_type: str = "flash"
    # # default or rms
    # norm_type: str = "rms"
    # # default or moe
    # ffn_type: str = "default"
    # norm_eps: float = 1e-5
    # dropout: float = 0.0
    # rope_base: float = 1e6
    # hidden_dim: int = None
    # multiple_of: int = 64
    # use_moe: bool = False

    def __init__(self, dim: int = 256, max_seq_len: int = 8192, n_layers: int = 16, vocab_size: int = 6400,
                 n_heads: int = 8, n_kv_heads: int = 2, attention_type: str = "flash", norm_type: str = "rms",
                 ffn_type: str = "default", norm_eps: float = 1e-5, dropout: float = 0.0, rope_base: float = 1e6,
                 hidden_dim: int = None, multiple_of: int = 64, use_moe: bool = False, ffn_act: str = 'silu',
                 max_position_embs: int = 32768, **kwargs):
        super().__init__(**kwargs)
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.n_layers = n_layers
        self.vocab_size = vocab_size
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.attention_type = attention_type
        self.norm_type = norm_type
        self.ffn_type = ffn_type
        self.norm_eps = norm_eps
        self.dropout = dropout
        self.rope_base = rope_base
        self.hidden_dim = hidden_dim
        self.multiple_of = multiple_of
        self.use_moe = use_moe
        self.ffn_act = ffn_act
        self.max_position_embs = max_position_embs

