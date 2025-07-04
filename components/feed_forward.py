from torch import nn
import torch.nn.functional as F
from .model_config import RainLLMConfig
from transformers.activations import ACT2FN


class FFN(nn.Module):
    def __init__(self, config: RainLLMConfig):
        super().__init__()
        self.config = config
        self.eps = config.norm_eps
class ConvFeedForward(FFN):
    def __init__(self, config: RainLLMConfig):
        super().__init__(config)
        if config.hidden_dim is None:
            hidden_dim = 4 * config.dim
            hidden_dim = int(2 * hidden_dim / 3)
            config.hidden_dim = config.multiple_of * ((hidden_dim + config.multiple_of - 1) // config.multiple_of)
        self.w1 = nn.Conv1d(config.dim, config.hidden_dim, kernel_size=1, bias=False)
        self.w2 = nn.Conv1d(config.hidden_dim, config.dim, kernel_size=1, bias=False)
        self.w3 = nn.Conv1d(config.dim, config.hidden_dim, kernel_size=1, bias=False)
        self.dropout = nn.Dropout(config.dropout)
        self.act_fn = ACT2FN[self.config.ffn_act]

    def forward(self, x):
        x = x.transpose(1, 2)
        out = self.w2(self.act_fn(self.w1(x)) * self.w3(x))
        return self.dropout(out.transpose(1, 2))

class HighttoLowFeedForward(FFN):
    def __init__(self, config: RainLLMConfig):
        super().__init__(config)
        if config.hidden_dim is None:
            hidden_dim = config.dim / 4
            hidden_dim = int(2 * hidden_dim / 3)
            config.hidden_dim = config.multiple_of * ((hidden_dim + config.multiple_of - 1) // config.multiple_of)
        self.w1 = nn.Conv1d(config.dim, config.hidden_dim, kernel_size=1, bias=False)
        self.w2 = nn.Conv1d(config.hidden_dim, config.dim, kernel_size=1, bias=False)
        self.w3 = nn.Conv1d(config.dim, config.hidden_dim, kernel_size=1, bias=False)
        self.dropout = nn.Dropout(config.dropout)
        self.act_fn = ACT2FN[self.config.ffn_act]

    def forward(self, x):
        return self.dropout(self.w2(self.act_fn(self.w1(x)) * self.w3(x)))
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
        self.act_fn = ACT2FN[self.config.ffn_act]

    def forward(self, x):
        return self.dropout(self.w2(self.act_fn(self.w1(x)) * self.w3(x)))





class MemoryLayer(FFN):
    pass

class MOEFeedForward(FFN):
    pass


class FeedForwardFactory:
    @staticmethod
    def ffn(config: RainLLMConfig, feed_forward_type: str) -> FFN:
        if feed_forward_type in ["", "default"]:
            return FeedForward(config)
        elif feed_forward_type == "conv":
            return ConvFeedForward(config)
        elif feed_forward_type == "moe":
            pass
        else:
            raise ValueError(f"Unknown feed forward type: {feed_forward_type}")
