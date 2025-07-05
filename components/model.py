import json
import math
import time
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


class FlowNet(nn.Module):
    def __init__(self, config : RainLLMConfig, num_freq_bands=64, max_period=365*24*60*60):
        super().__init__()
        self.num_freq_bands = num_freq_bands
        self.max_period = max_period
        self.frequencies = 1.0 / (2.0 ** torch.linspace(
            math.log2(max_period),
            0.0,
            num_freq_bands
        ))
        self.dim = config.dim
        self.flow_layer = nn.Linear(2 * num_freq_bands, self.dim)

    def forward(self, input):
        timestamps = input.float()
        angles = timestamps.unsqueeze(-1) * 2 * math.pi * self.frequencies
        sin_enc = torch.sin(angles)
        cos_enc = torch.cos(angles)
        time_enc = torch.cat([sin_enc, cos_enc], dim=-1)
        return self.flow_layer(time_enc)


class TransformerBlock(nn.Module):
    """
    解码器块
    """

    def __init__(self, id: int, config: RainLLMConfig, runtime_auto_eval_cut: bool = False):
        super().__init__()
        self.n_heads = config.n_heads
        self.prune_threshold = config.prune_threshold
        self.dim = config.dim
        self.save_memory = config.save_memory
        self.head_dim = config.dim // self.n_heads
        self.attention = AttentionFactory.attention(config, config.attention_type)

        self.layer_id = id
        self.pre_attn_norm = NormFactory.norm(config, config.norm_type)
        self.pre_ffn_norm = NormFactory.norm(config, config.norm_type)
        self.ffn = FeedForwardFactory.ffn(config, config.ffn_type)

        self.alpha = nn.Parameter(torch.ones(1))
        self.beta = nn.Parameter(torch.ones(1))

        self._attention_active = torch.abs(self.alpha).item() < self.prune_threshold
        self._ffn_active = torch.abs(self.beta).item() < self.prune_threshold
        self.useless = False
        self.first_run = True
        self.runtime_auto_eval_cut_first_run = runtime_auto_eval_cut if not self.training else False

    def _update_pruning_state(self) -> None:
        # return
        """更新剪枝状态，避免在forward中重复计算"""
        with torch.no_grad():
            self._attention_active = torch.abs(self.alpha).item() < self.prune_threshold
            self._ffn_active = torch.abs(self.beta).item() < self.prune_threshold
            if not self._ffn_active and not self._attention_active:
                self.useless = True


    def forward(self, hidden_states, pos_cis, past_key_value=None, use_cache=False, attention_mask=None):

        res = hidden_states
        hidden_states, present_key_value = self.attention(
            self.pre_attn_norm(hidden_states),
            pos_cis,
            past_key_value=past_key_value,
            use_cache=use_cache,
            attention_mask=attention_mask
        )
        res_add = hidden_states + self.alpha * res
        # 残差链接
        out = self.beta * res_add + self.ffn(self.pre_ffn_norm(res_add))
        # if self.runtime_auto_eval_cut_first_run:
        #     self._update_pruning_state()
        #     self.runtime_auto_eval_cut_first_run = False
        #
        # res = hidden_states
        # if self.useless:
        #     if self.save_memory:
        #         del self.ffn, self.attention
        #     return res, past_key_value
        # if torch.abs(self.alpha).item() < self.prune_threshold:
        #     hidden_states, present_key_value = self.attention(
        #         self.pre_attn_norm(hidden_states),
        #         pos_cis,
        #         past_key_value=past_key_value,
        #         use_cache=use_cache,
        #         attention_mask=attention_mask
        #     )
        #     res_add = 1 * hidden_states + self.alpha * res
        # else:
        #     present_key_value = past_key_value
        #     res_add = res
        # if torch.abs(self.beta).item() < self.prune_threshold:
        #     out = self.beta * res_add  + 1 * self.ffn(self.pre_ffn_norm(res_add))
        # else:
        #     out = res_add
        return out, present_key_value


class RainModule(nn.Module):
    def __init__(self, llm_config: RainLLMConfig = RainLLMConfig()):
        super().__init__()
        self.config = llm_config
        self.embeddings = nn.Embedding(self.config.vocab_size, self.config.dim)
        self.dropout = nn.Dropout(self.config.dropout)
        self.transformer_blocks = nn.ModuleList(
            [TransformerBlock(i, self.config, False) for i in range(self.config.n_layers)]
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
                past_key_values: List[Tuple[torch.Tensor, torch.Tensor]] | None = None,
                use_cache: bool = False,
                flow: torch.Tensor | None = None,
                **args):
        # 检查 kv_cache 并初始化
        _, seq_len = input_token_ids.shape
        past_key_values = past_key_values or [None] * len(self.transformer_blocks)
        start_pos = past_key_values[0][0].shape[1] if past_key_values[0] is not None else 0
        # 词嵌入
        hidden_states = self.dropout(self.embeddings(input_token_ids))
        # 旋转位置编码计算
        position_embeddings = (
            self.freqs_cos[start_pos:start_pos + seq_len],
            self.freqs_sin[start_pos:start_pos + seq_len]
        )
        # 传播每一个Transformer块
        presents = []
        if flow is not None:
            hidden_states += flow
        for layer_ids, (layer, past_key_value) in enumerate(zip(self.transformer_blocks, past_key_values)):
            hidden_states, present = layer(
                hidden_states,
                position_embeddings,
                past_key_value=past_key_value,
                use_cache=use_cache,
                attention_mask=attention_mask)
            presents.append(present)
        hidden_states = self.norm(hidden_states)
        return hidden_states, presents


class RainForCausalLM(PreTrainedModel, GenerationMixin):
    def __init__(self, config: RainLLMConfig):
        self.config = config or RainLLMConfig()
        super().__init__(self.config)
        if config.flow:
            self.flow = FlowNet(self.config)
        else:
            self.flow = None
        self.model = RainModule(self.config)
        self.output = nn.Linear(self.config.dim, self.config.vocab_size, bias=False)
        self.model.embeddings.weight = self.output.weight
        self.OUT = CausalLMOutputWithPast()
        self.flow_state = None
        self.user_input = None

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
            flow=self.flow_state,
            **args
        )
        slice_indices = slice(-logits_keep, None) if isinstance(logits_keep, int) else logits_keep
        logits = self.output(hidden_states[:, slice_indices, :])
        self.OUT.__setitem__('last_hidden_state', hidden_states)
        self.OUT.__setitem__('logits', logits)
        self.OUT.__setitem__('aux_loss', 0)
        self.OUT.__setitem__('past_key_values', past_kvs)
        return self.OUT

    def input(self, x):
        self.user_input = x

    def tick(self):
        if self.user_input is not None:
            # todo 输入结合，把flow也放进去
            res = self.forward()
            self.user_input = None
            return res
        if self.flow is not None:
            self.flow_state = self.flow(time.time())


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
