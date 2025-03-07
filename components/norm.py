import torch
from torch import nn

from .model_config import RainLLMConfig


class RMSNorm(nn.Module):
    name = "rms"

    def __init__(self, dim: int, eps: float):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        x_pow = x.pow(2)
        mean_x = x_pow.mean(dim=-1, keepdim=True)
        sqrt_x = torch.rsqrt(mean_x + self.eps)
        norm_x = x * sqrt_x
        out = self.weight * norm_x
        return out.type_as(x)


class NormFactory:
    name = "factory"

    @staticmethod
    def norm(config: RainLLMConfig, norm_type: str = "") -> nn.Module:
        if norm_type in ["layer", "", "default"]:
            return nn.LayerNorm(config.dim, config.norm_eps)
        elif norm_type == "rms":
            return RMSNorm(config.dim, config.norm_eps)
        else:
            raise ValueError(f"Unknown norm type: {norm_type}")
