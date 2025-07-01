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
        self.n_attention_heads = config.n_heads
        # kv头的数量 一般来说，kv注意力头的设计与注意力头相同
        self.n_kv_heads = config.n_kv_heads if config.n_kv_heads else config.n_heads
        # 每个注意力头的维度
        self.head_dim = config.dim // self.n_attention_heads
        # 需要复制的KV头的数量
        assert self.n_attention_heads % self.n_kv_heads == 0
        self.n_rep = self.n_attention_heads // self.n_kv_heads
        # q矩阵
        self.wq = nn.Linear(self.config.dim, self.n_attention_heads * self.head_dim, bias=False)
        # k矩阵
        self.wk = nn.Linear(self.config.dim, self.n_kv_heads * self.head_dim, bias=False)
        # v矩阵
        self.wv = nn.Linear(self.config.dim, self.n_kv_heads * self.head_dim, bias=False)
        # 线性变换层
        self.wo = nn.Linear(self.head_dim * self.n_attention_heads, self.config.dim, bias=False)
        # drop
        self.attention_drop = nn.Dropout(self.config.dropout)
        self.resid_drop = nn.Dropout(self.config.dropout)
        self.dropout = self.config.dropout
        # 遮罩
        # self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention') and args.flash_attn

        # mask = torch.full((1, 1, config.max_seq_len, config.max_seq_len), float("-inf"))
        # mask = torch.triu(mask, diagonal=1)
        # # 注册mask方法
        # self.register_buffer("mask", mask, persistent=False)

    @staticmethod
    def apply_rotary_emb(q: torch.Tensor, k: torch.Tensor, cos, sin, unsqueeze_dim=1):
        """
        对 q k 进行旋转位置编码
        :param q: [batch_size, seq_len, n_attention_heads, per_head_dim]
        :param k: [batch_size, seq_len, n_attention_heads, per_head_dim]
        :param pos_cis: [max_seq_len, per_head_dim]
        """
        def rotate_half(x):
            return torch.cat((-x[..., x.shape[-1] // 2:], x[..., : x.shape[-1] // 2]), dim=-1)

        q_embed = (q * cos.unsqueeze(unsqueeze_dim)) + (rotate_half(q) * sin.unsqueeze(unsqueeze_dim))
        k_embed = (k * cos.unsqueeze(unsqueeze_dim)) + (rotate_half(k) * sin.unsqueeze(unsqueeze_dim))
        return q_embed, k_embed


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
        q = self.wq(x).view(batch_size, seq_len, self.n_attention_heads, self.head_dim)
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
                pos_cis: Tuple[torch.Tensor, torch.Tensor],
                kv_cache: Tuple[torch.Tensor, torch.Tensor] | None = None,
                use_cache=False,
                attention_mask: torch.Tensor | None = None):
        batch_size, seq_len, dim = x.shape
        q = self.wq(x).view(batch_size, seq_len, self.n_attention_heads, self.head_dim)
        k = self.wk(x).view(batch_size, seq_len, self.n_kv_heads, self.head_dim)
        v = self.wv(x).view(batch_size, seq_len, self.n_kv_heads, self.head_dim)

        cos, sin = pos_cis
        q, k = Attention.apply_rotary_emb(q, k, cos[:seq_len], sin[:seq_len])

        if kv_cache is not None:
            k = torch.cat([kv_cache[0], k], dim=1)
            v = torch.cat([kv_cache[1], v], dim=1)
        past_kv = (k, v) if use_cache else None

        q, k, v = (
            q.transpose(1, 2),
            repeat_kv(k, self.n_rep).transpose(1, 2),
            repeat_kv(v, self.n_rep).transpose(1, 2)
        )

        if seq_len != 1:
            dropout_p = self.dropout if self.training else 0.0
            attn_mask = None
            if attention_mask is not None:
                attn_mask = attention_mask.view(batch_size, 1, 1, -1).expand(batch_size, self.n_attention_heads, seq_len, -1)
                attn_mask = attn_mask.bool() if attention_mask is not None else None
            output = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, dropout_p=dropout_p, is_causal=True)
        else:
            scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
            scores = scores + torch.triu(
                torch.full((seq_len, seq_len), float("-inf"), device=scores.device),
                diagonal=1
            ).unsqueeze(0).unsqueeze(0)  # scores+mask
            if attention_mask is not None:
                extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
                extended_attention_mask = (1.0 - extended_attention_mask) * -1e9
                scores = scores + extended_attention_mask
            scores = F.softmax(scores.float(), dim=-1).type_as(q)
            scores = self.attn_dropout(scores)
            output = scores @ v
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
