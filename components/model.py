import json
from typing import List, Tuple, Generator

import torch
import torch.nn.functional as F
from torch import nn
from transformers import PreTrainedModel, GenerationMixin
from transformers.modeling_outputs import CausalLMOutputWithPast
from .attention import AttentionFactory
from .feed_forward import FeedForwardFactory, MOEFeedForward
from .model_config import RainLLMConfig
from .norm import NormFactory


def rotate(dim: int, base: float, len: int = 32 * 1024):
    """
    旋转位置编码
    """
    freqs = 1.0 / (base ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(len, device=freqs.device)
    freqs = torch.outer(t, freqs).float()
    freqs_cos = torch.cat([torch.cos(freqs), torch.cos(freqs)], dim=-1)
    freqs_sin = torch.cat([torch.sin(freqs), torch.sin(freqs)], dim=-1)
    return freqs_cos, freqs_sin


class TransformerBlock(nn.Module):
    """
    解码器块
    """

    def __init__(self, id: int, config: RainLLMConfig):
        super().__init__()
        self.n_heads = config.n_heads
        self.dim = config.dim
        self.head_dim = config.dim // self.n_heads
        self.attention = AttentionFactory.attention(config, config.attention_type)

        self.layer_id = id
        self.input_norm = NormFactory.norm(config, config.norm_type)
        self.attention_norm = NormFactory.norm(config, config.norm_type)
        self.ffn = FeedForwardFactory.ffn(config, config.ffn_type)

    def forward(self, hidden_states, pos_cis, kv_cache=None, use_cache=False, attention_mask=None):
        # vector_attn_norm = self.attention_norm(vector)
        input = hidden_states
        hidden_states, past_kv = self.attention(
            self.input_norm(hidden_states),
            pos_cis,
            kv_cache=kv_cache,
            use_cache=use_cache,
            attention_mask=attention_mask
        )
        # 残差链接
        res_add = hidden_states + input
        out = res_add + self.ffn(self.attention_norm(res_add))
        return out, past_kv


class RainModule(nn.Module):
    def __init__(self, llm_config: RainLLMConfig = RainLLMConfig()):
        super().__init__()
        self.config = llm_config
        self.embeddings = nn.Embedding(self.config.vocab_size, self.config.dim)
        self.dropout = nn.Dropout(self.config.dropout)
        self.transformer_blocks = nn.ModuleList(
            [TransformerBlock(i, self.config) for i in range(self.config.n_layers)]
        )
        self.norm = NormFactory.norm(self.config, self.config.norm_type)
        cos, sin = rotate(dim=self.config.dim // self.config.n_heads,
                          base=self.config.rope_base,
                          len=self.config.max_position_embs)
        self.register_buffer("freqs_cos", cos, persistent=False)
        self.register_buffer("freqs_sin", sin, persistent=False)


    def forward(self,
                input_token_ids: torch.Tensor = None,
                attention_mask: torch.Tensor | None = None,
                kv_cache: List[Tuple[torch.Tensor, torch.Tensor]] | None = None,
                use_cache: bool = False,
                **args):
        # 检查 kv_cache 并初始化
        _, seq_len = input_token_ids.shape
        kv_cache = kv_cache or [None] * len(self.transformer_blocks)
        start_pos = kv_cache[0][0].shape[1] if kv_cache[0] is not None else 0
        # 词嵌入
        hidden_states = self.dropout(self.embeddings(input_token_ids))
        # 旋转位置编码计算
        position_embeddings = (
            self.freqs_cos[start_pos:start_pos + seq_len],
            self.freqs_sin[start_pos:start_pos + seq_len]
        )
        # 传播每一个Transformer块
        presents = []
        for layer_ids, (layer, past_key_value) in enumerate(zip(self.transformer_blocks, kv_cache)):
            hidden_states, present = layer(
                hidden_states,
                position_embeddings,
                kv_cache=past_key_value,
                use_cache=use_cache,
                attention_mask=attention_mask)
            presents.append(present)
        hidden_states = self.norm(hidden_states)
        return hidden_states, presents


class RainForCausalLM(PreTrainedModel, GenerationMixin):
    def __init__(self, config: RainLLMConfig):
        self.config = config or RainLLMConfig()
        super().__init__(self.config)
        self.model = RainModule(self.config)
        self.output = nn.Linear(self.config.dim, self.config.vocab_size, bias=False)
        self.model.embeddings.weight = self.output.weight
        self.OUT = CausalLMOutputWithPast()

    def forward(self,
                input_ids: torch.Tensor | None = None,
                attention_mask: torch.Tensor | None = None,
                past_kvs: List[Tuple[torch.Tensor, torch.Tensor]] | None = None,
                use_cache: bool = False,
                logits_keep: int | torch.Tensor = 0,
                **args):
        hidden_states, past_kvs = self.model(
            input_token_ids=input_ids,
            attention_mask=attention_mask,
            kv_cache=past_kvs,
            use_cache=use_cache,
            **args
        )
        slice_indices = slice(-logits_keep, None) if isinstance(logits_keep, int) else logits_keep
        logits = self.output(hidden_states[:, slice_indices, :])
        self.OUT.__setitem__('last_hidden_state', hidden_states)
        self.OUT.__setitem__('logits', logits)
        self.OUT.__setitem__('aux_loss', 0)
        self.OUT.__setitem__('past_key_values', past_kvs)
        return self.OUT

    # @torch.inference_mode()
    # def generate(self,
    #              input_ids: torch.Tensor,
    #              eos_token_id: int = 2,
    #              pad_token_id: int = 0,
    #              max_new_tokens: int = 1024,
    #              temperature: float = 0.75,
    #              top_p: float = 0.90,
    #              stream: bool = False,
    #              rp=1.,
    #              use_kv_cache: bool = True,
    #              **args) -> Generator | torch.Tensor:
    #     if stream:
    #         return self._stream(input_ids, eos_token_id, max_new_tokens, temperature, top_p, rp, use_kv_cache, **args)
    #     generated = []
    #     # 在训练时 input_ids.size(0)不为1，这里是批量处理。
    #     for i in range(input_ids.size(0)):
    #         # 去除 pad_token_id
    #         one_no_pad_token_ids = input_ids[i][input_ids[i] != pad_token_id].unsqueeze(0)
    #         out = self._stream(one_no_pad_token_ids, eos_token_id, max_new_tokens,
    #                            temperature, top_p, rp, use_kv_cache, **args)
    #         # 获取最新的生成
    #         tokens_list = [tokens[:, -1:] for tokens in out]
    #         # 拼接所有生成
    #         generated_tokens = torch.cat(tokens_list, dim=-1) if tokens_list else one_no_pad_token_ids
    #         # 前后拼接
    #         full_seq = torch.cat([one_no_pad_token_ids, generated_tokens], dim=-1)
    #         generated.append(full_seq)
    #     # 获取最长，并补pad
    #     max_len = max(seq.size(1) for seq in generated)
    #     generated = [
    #         torch.cat(
    #             [seq, torch.full((1, max_len - seq.size(1)), pad_token_id, dtype=seq.dtype, device=seq.device)], dim=-1)
    #         for seq in generated
    #     ]
    #     return torch.cat(generated, dim=0)
    #
    # def _stream(self,
    #             input_ids: torch.Tensor,
    #             eos_token_id: int,
    #             max_new_tokens: int,
    #             temperature: float,
    #             top_p: float,
    #             rp: float,
    #             use_cache: bool,
    #             **args) -> Generator:
    #     # 初始化
    #     start, first, past_kv = input_ids.shape[1], True, None
    #     # 通过网络后对概率分布进行处理和采样，最后在通过softmax得到token
    #     while input_ids.shape[1] < max_new_tokens - 1:
    #         if first or not use_cache:
    #             out = self(input_token_ids=input_ids, kv_cache=past_kv, use_cache=use_cache, **args)
    #         else:
    #             out = self(input_token_ids=input_ids[:, -1:], kv_cache=past_kv, use_cache=use_cache,
    #                        start_pos=input_ids.shape[1] - 1, **args)
    #         logits, past_kvs = out.logits[:, -1, :], out.past_key_values
    #         logits[:, list(set(input_ids.tolist()[0]))] /= rp
    #         logits /= (temperature + 1e-9)
    #         if top_p is not None and top_p < 1.0:
    #             sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
    #             sorted_probs = F.softmax(sorted_logits, dim=-1)
    #             cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
    #             sorted_indices_to_remove = cumulative_probs > top_p
    #             sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
    #             sorted_indices_to_remove[:, 0] = False
    #             indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
    #             logits[indices_to_remove] = -float('Inf')
    #         input_ids_next = torch.multinomial(F.softmax(logits, dim=-1), num_samples=1)
    #         input_ids = torch.cat((input_ids, input_ids_next), dim=1)
    #         yield input_ids[:, start:]
    #         if input_ids_next.item() == eos_token_id:
    #             break
