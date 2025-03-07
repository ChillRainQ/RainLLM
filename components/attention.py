import math
from typing import Tuple, List

import torch
from torch import nn
import torch.nn.functional as F

from .model_config import RainLLMConfig


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    重复KV
    """
    if n_rep <= 1:
        return x
    else:
        batch_size, seq_len, n_kv_heads, per_head_dim = x.shape
        return (
            x[:, :, :, None, :]
            .expand(batch_size, seq_len, n_kv_heads, n_rep, per_head_dim)
            .reshape(batch_size, seq_len, n_kv_heads * n_rep, per_head_dim)
        )


class Attention(nn.Module):
    def __init__(self, config: RainLLMConfig):
        super().__init__()
        self.config = config
        # 注意力头
        self.n_heads = config.n_heads
        # kv头的数量 一般来说，kv注意力头的设计与注意力头相同
        self.n_kv_heads = config.n_kv_heads if config.n_kv_heads else config.n_heads
        # 每个注意力头的维度
        self.head_dim = config.dim // self.n_heads
        # 需要复制的KV头的数量
        assert self.n_heads % self.n_kv_heads == 0
        self.n_rep = self.n_heads // self.n_kv_heads
        # q矩阵
        self.wq = nn.Linear(config.dim, self.n_heads * self.head_dim, bias=False)
        # k矩阵
        self.wk = nn.Linear(config.dim, self.n_kv_heads * self.head_dim, bias=False)
        # v矩阵
        self.wv = nn.Linear(config.dim, self.n_kv_heads * self.head_dim, bias=False)
        # 线性变换层
        self.wo = nn.Linear(self.head_dim * self.n_heads, config.dim, bias=False)
        # drop
        self.attention_drop = nn.Dropout(config.dropout)
        self.resid_drop = nn.Dropout(config.dropout)
        self.dropout = config.dropout
        # 遮罩
        mask = torch.full((1, 1, config.max_seq_len, config.max_seq_len), float("-inf"))
        mask = torch.triu(mask, diagonal=1)
        # 注册mask方法
        self.register_buffer("mask", mask, persistent=False)

    @staticmethod
    def apply_rotary_emb(q: torch.Tensor, k: torch.Tensor, pos_cis: torch.Tensor):
        """
        对 q k 进行旋转位置编码
        :param q: [batch_size, seq_len, n_heads, per_head_dim]
        :param k: [batch_size, seq_len, n_heads, per_head_dim]
        :param pos_cis: [max_seq_len, per_head_dim]
        """
        # 调整形状用于在batch_size维度和n_heads, per_head_dim维度进行广播
        # 修改q与k为复数域 最后一个维度必须是2
        q_ = torch.view_as_complex(q.float().reshape(*q.shape[: -1], -1, 2))
        k_ = torch.view_as_complex(k.float().reshape(*k.shape[: -1], -1, 2))
        ndim = q.ndim
        # [1, seq_len, 1, 1, 2]
        need_pos_cis_sharp = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(q_.shape)]
        sharp_pos_cis = pos_cis.view(*need_pos_cis_sharp)
        x_out = torch.view_as_real(q_ * sharp_pos_cis).flatten(3).type_as(q)
        k_out = torch.view_as_real(k_ * sharp_pos_cis).flatten(3).type_as(k)
        return x_out, k_out


class MultiHeadAttention(Attention):
    name = "default"

    def __init__(self, config: RainLLMConfig):
        super().__init__(config)

    def forward(self,
                x: torch.Tensor,
                pos_cis: torch.Tensor,
                kv_cache: List[Tuple[torch.Tensor, torch.Tensor]] | None = None,
                use_cache=False):
        batch_size, seq_len, dim = x.shape
        q = self.wq(x).view(batch_size, seq_len, self.n_heads, self.head_dim)
        k = self.wk(x).view(batch_size, seq_len, self.n_kv_heads, self.head_dim)
        q, k = Attention.apply_rotary_emb(q, k, pos_cis)
        v = self.wv(x).view(batch_size, seq_len, self.n_kv_heads, self.head_dim)
        if kv_cache:
            k = torch.cat([kv_cache[0], k], dim=1)
            v = torch.cat([kv_cache[1], v], dim=1)
        past_kv = (k, v) if use_cache else None
        q, k, v = (
            q.transpose(1, 2),
            repeat_kv(k, self.n_rep).transpose(1, 2),
            repeat_kv(v, self.n_rep).transpose(1, 2)
        )
        score = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        score += self.mask[:, :, :seq_len, :seq_len]
        score = F.softmax(score.float(), dim=-1, dtype=q.dtype)
        score = self.attention_drop(score)
        output = score @ v
        output = output.transpose(1, 2).reshape(batch_size, seq_len, -1)
        output = self.resid_drop(self.wo(output))
        return output, past_kv


class FlashAttention(Attention):
    name = "flash"

    def __init__(self, config: RainLLMConfig):
        super().__init__(config)
        self.flash = hasattr(torch.nn.functional, "scaled_dot_product_attention")

    def forward(self,
                x: torch.Tensor,
                pos_cis: torch.Tensor,
                kv_cache: List[Tuple[torch.Tensor, torch.Tensor]] | None = None,
                use_cache=False):
        batch_size, seq_len, dim = x.shape
        q = self.wq(x).view(batch_size, seq_len, self.n_heads, self.head_dim)
        k = self.wk(x).view(batch_size, seq_len, self.n_kv_heads, self.head_dim)
        q, k = Attention.apply_rotary_emb(q, k, pos_cis)
        v = self.wv(x).view(batch_size, seq_len, self.n_kv_heads, self.head_dim)
        if kv_cache:
            k = torch.cat([kv_cache[0], k], dim=1)
            v = torch.cat([kv_cache[1], v], dim=1)
        past_kv = (k, v) if use_cache else None
        q, k, v = (
            q.transpose(1, 2),
            repeat_kv(k, self.n_rep).transpose(1, 2),
            repeat_kv(v, self.n_rep).transpose(1, 2)
        )
        if self.flash and seq_len != 1:
            dropout_p = self.dropout if self.training else 0.0
            output = F.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=dropout_p, is_causal=True)
            output = output.transpose(1, 2).reshape(batch_size, seq_len, -1)
            output = self.resid_drop(self.wo(output))
            return output, past_kv


class AttentionFactory:
    name = "factory"

    @staticmethod
    def attention(config: RainLLMConfig, attention_type: str = "") -> Attention:
        if attention_type in ["", "default", "multi-head"]:
            return MultiHeadAttention(config)
        elif attention_type == "flash":
            return FlashAttention(config)
        else:
            raise ValueError(f"illegal attention_type: {attention_type}!")
