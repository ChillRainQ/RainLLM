import json
from typing import List, Tuple, Generator

import torch
import torch.nn.functional as F
from torch import nn
from transformers import PreTrainedModel
from transformers.modeling_outputs import CausalLMOutputWithPast
from .attention import AttentionFactory
from .feed_forward import FeedForwardFactory, MOEFeedForward
from .model_config import RainLLMConfig
from .norm import NormFactory


def rotate(dim: int, base: float, seq_len: int = 32 * 1024):
    """
    旋转位置编码
    """
    if dim % 2 != 0:
        raise ValueError(f"dim must be even, but got {dim}")
    d = dim // 2
    i = torch.arange(1, d + 1, dtype=torch.float32)
    # 旋转角序列
    theta = base ** (-2 * (i - 1) / dim)
    # 索引位置
    m = torch.arange(seq_len, device=theta.device)
    m_theta = torch.outer(m, theta).float()
    # 复数形式
    pos_cis = torch.polar(torch.ones_like(m_theta), m_theta)
    return pos_cis
# def rotate(dim: int, end: int = int(32 * 1024), theta: float = 1e6):
#     """
#     位置编码预计算
#     """
#     freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
#     t = torch.arange(end, device=freqs.device)  # type: ignore
#     freqs = torch.outer(t, freqs).float()  # type: ignore
#     pos_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
#     return pos_cis


class TransformerBlock(nn.Module):
    """
    解码器块
    """

    def __init__(self, id: int, config: RainLLMConfig):
        super().__init__()
        self.id = id
        self.n_heads = config.n_heads
        self.dim = config.dim
        self.head_dim = config.dim // self.n_heads
        self.attention = AttentionFactory.attention(config, config.attention_type)
        self.attention_norm = NormFactory.norm(config, config.norm_type)
        self.ffn = FeedForwardFactory.ffn(config, config.ffn_type)
        self.ffn_norm = NormFactory.norm(config, config.norm_type)

    def forward(self, vector, pos_cis, kv_cache=None, use_cache=False):
        # vector_attn_norm = self.attention_norm(vector)
        vector_attn, past_kv = self.attention(
            self.attention_norm(vector),
            pos_cis,
            kv_cache=kv_cache,
            use_cache=use_cache
        )
        # 残差链接
        res_add = vector_attn + vector
        out = res_add + self.ffn(self.ffn_norm(res_add))
        return out, past_kv


class RainLLM(PreTrainedModel):
    def __init__(self, llm_config: RainLLMConfig = RainLLMConfig()):
        super().__init__(llm_config)
        self.config = llm_config
        # 词嵌入层
        self.embeddings = nn.Embedding(self.config.vocab_size, self.config.dim)
        # 位置编码预计算
        self.register_buffer("pos_cis",
                             rotate(dim=self.config.dim // self.config.n_heads,
                                    base=self.config.rope_base),
                             persistent=False)
        # Transformer块
        self.transformer_blocks = nn.ModuleList(
            [TransformerBlock(i, self.config) for i in range(self.config.n_layers)]
        )
        # 层归一化
        self.norm = NormFactory.norm(self.config, self.config.norm_type)
        # drop
        self.dropout = nn.Dropout(self.config.dropout)
        # 输出
        self.output = nn.Linear(self.config.dim, self.config.vocab_size, bias=False)
        # 权重共享
        self.embeddings.weight = self.output.weight
        # 输出结构
        self.OUT = CausalLMOutputWithPast()
        print(json.dumps(self.config.__dict__, indent=4))

    def forward(self,
                input_token_ids: torch.Tensor = None,
                kv_cache: List[Tuple[torch.Tensor, torch.Tensor]] | None = None,
                use_cache: bool = False,
                **args):
        # 检查 kv_cache 并初始化
        kv_cache = kv_cache or [None] * len(self.transformer_blocks)
        start_pos = args.get('start_pos', 0)
        # 词嵌入
        vector = self.dropout(self.embeddings(input_token_ids))
        # 旋转位置编码预计算
        pos_cis = self.pos_cis[start_pos: start_pos + input_token_ids.size(1)]
        # 传播每一个Transformer块
        past_kvs = []
        for i, layer in enumerate(self.transformer_blocks):
            vector, past_kv = layer(
                vector,
                pos_cis,
                kv_cache=kv_cache[i],
                use_cache=use_cache)
            past_kvs.append(past_kv)
        # 概率
        logits = self.output(self.norm(vector))
        # aux_loss = sum(l.ffn.aux_loss for l in self.transformer_blocks if isinstance(l.feed_forward, MOEFeedForward))
        self.OUT.__setitem__("logits", logits)
        self.OUT.__setitem__("aux_loss", 0)
        self.OUT.__setitem__("past_key_values", past_kvs)
        return self.OUT

    @torch.inference_mode()
    def generate(self,
                 input_ids: torch.Tensor,
                 eos_token_id: int = 2,
                 pad_token_id: int = 0,
                 max_new_tokens: int = 1024,
                 temperature: float = 0.75,
                 top_p: float = 0.90,
                 stream: bool = False,
                 rp=1.,
                 use_kv_cache: bool = True,
                 **args) -> Generator | torch.Tensor:
        if stream:
            return self._stream(input_ids, eos_token_id, max_new_tokens, temperature, top_p, rp, use_kv_cache, **args)
        generated = []
        # 在训练时 input_ids.size(0)不为1，这里是批量处理。
        for i in range(input_ids.size(0)):
            # 去除 pad_token_id
            one_no_pad_token_ids = input_ids[i][input_ids[i] != pad_token_id].unsqueeze(0)
            out = self._stream(one_no_pad_token_ids, eos_token_id, max_new_tokens,
                               temperature, top_p, rp, use_kv_cache, **args)
            # 获取最新的生成
            tokens_list = [tokens[:, -1:] for tokens in out]
            # 拼接所有生成
            generated_tokens = torch.cat(tokens_list, dim=-1) if tokens_list else one_no_pad_token_ids
            # 前后拼接
            full_seq = torch.cat([one_no_pad_token_ids, generated_tokens], dim=-1)
            generated.append(full_seq)
        # 获取最长，并补pad
        max_len = max(seq.size(1) for seq in generated)
        generated = [
            torch.cat(
                [seq, torch.full((1, max_len - seq.size(1)), pad_token_id, dtype=seq.dtype, device=seq.device)], dim=-1)
            for seq in generated
        ]
        return torch.cat(generated, dim=0)
    def _stream(self,
                input_ids: torch.Tensor,
                eos_token_id: int,
                max_new_tokens: int,
                temperature: float,
                top_p: float,
                rp: float,
                use_cache: bool,
                **args) -> Generator:
        # 初始化
        start, first, past_kv = input_ids.shape[1], True, None
        # 通过网络后对概率分布进行处理和采样，最后在通过softmax得到token
        while input_ids.shape[1] < max_new_tokens - 1:
            if first or not use_cache:
                out = self(input_token_ids=input_ids, kv_cache=past_kv, use_cache=use_cache, **args)
            else:
                out = self(input_token_ids=input_ids[:, -1:], kv_cache=past_kv, use_cache=use_cache,
                           start_pos=input_ids.shape[1] - 1, **args)
            logits, past_kvs = out.logits[:, -1, :], out.past_key_values
            logits[:, list(set(input_ids.tolist()[0]))] /= rp
            logits /= (temperature + 1e-9)
            # top_p采样
            # if top_p is not None and top_p < 1.0:
            #     sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
            #     sorted_probs = F.softmax(sorted_logits, dim=-1)
            #     cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
            #     # 拿到满足top_p的索引
            #     sorted_indices_to_remove = cumulative_probs > top_p
            #     sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
            #     sorted_indices_to_remove[:, 0] = False
            #     indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
            #     logits[indices_to_remove] = -float('Inf')
            if top_p is not None and top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
                sorted_probs = F.softmax(sorted_logits, dim=-1)
                cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
                sorted_indices_to_remove[:, 0] = False
                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                logits[indices_to_remove] = -float('Inf')
            input_ids_next = torch.multinomial(F.softmax(logits, dim=-1), num_samples=1)
            input_ids = torch.cat((input_ids, input_ids_next), dim=1)
            yield input_ids[:, start:]
            if input_ids_next.item() == eos_token_id:
                break
