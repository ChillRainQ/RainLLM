import math

import torch
from torch import nn
import torch.nn.functional as F
from .model_config import RainLLMConfig


class FFN(nn.Module):
    def __init__(self, config: RainLLMConfig):
        super().__init__()
        self.config = config
        self.eps = config.norm_eps


class FeedForward(FFN):
    def __init__(self, config: RainLLMConfig):
        super().__init__(config)
        if config.hidden_dim is None:
            hidden_dim = 4 * config.dim
            hidden_dim = int(2 * hidden_dim / 3)
            config.hidden_dim = config.multiple_of * ((hidden_dim + config.multiple_of - 1) // config.multiple_of)
        self.w1 = nn.Linear(config.dim, config.hidden_dim, bias=False)
        self.w2 = nn.Linear(config.hidden_dim, config.dim, bias=False)
        self.w3 = nn.Linear(config.dim, config.hidden_dim, bias=False)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        return self.dropout(self.w2(F.silu(self.w1(x)) * self.w3(x)))

class MemoryLayer(FFN):
    """
    meta
    """
    pass

class MOEFeedForward(FFN):
    class MOEGate(nn.Module):
        def __init__(self, config: RainLLMConfig):
            super().__init__()
            self.config = config
            self.per_token_num_experts = config.per_token_num_experts
            self.n_experts = config.n_experts
            self.score_func = config.score_func
            self.aux_loss_alpha = config.aux_loss_alpha
            self.seq_aux = config.seq_aux
            self.norm_topk_prob = config.norm_topk_prob
            self.gate_dim = config.dim
            self.weight = nn.Parameter(torch.empty((self.n_experts, self.gate_dim)))
            self.reset_parameters()

        def reset_parameters(self):
            torch.nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

        def forward(self, hidden_states):
            pass
    def __init__(self, config: RainLLMConfig):
        super().__init__(config)
        self.experts = nn.ModuleList([
            FeedForward(config)
            for _ in range(config.n_experts)
        ])
        self.share_experts = None
        if self.config.n_share_experts > 0:
            self.share_experts = nn.ModuleList([
                FeedForward(config)
                for _ in range(self.config.n_share_experts)
            ])


class FeedForwardFactory:
    @staticmethod
    def ffn(config: RainLLMConfig, feed_forward_type: str) -> FFN:
        if feed_forward_type in ["", "default"]:
            return FeedForward(config)
        elif feed_forward_type == "moe":
            pass
        else:
            raise ValueError(f"Unknown feed forward type: {feed_forward_type}")
