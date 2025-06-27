import os.path

import torch
from torch import nn, Generator
from transformers import AutoTokenizer, AutoModelForCausalLM

from components.model import RainLLM
from components.model_config import RainLLMConfig


class RainClient:
    history: list
    model: RainLLM
    tokenizer: AutoTokenizer
    config: RainLLMConfig
    deivce: str

    def __init__(self, model_path, tokenizer_path, model_config_path, **args):
        """
        RainLLM模型初始化方法
        
        参数:
            model_path (str): 模型文件路径，可以是本地文件或预训练模型名称
            tokenizer_path (str): tokenizer文件路径
            model_config_path (str): 模型配置文件路径
            **args: 其他配置参数，包括:
                dim (int): 模型维度，默认256
                n_layers (int): 模型层数，默认16
                max_seq_len (int): 最大序列长度，默认8192
                stream (bool): 是否使用流式处理，默认True
        
        功能:
            1. 初始化tokenizer和设备
            2. 加载模型配置
            3. 根据路径类型(本地文件/预训练模型)加载模型
            4. 将模型移动到指定设备
            5. 打印模型参数量
        """
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model_config_path = model_config_path
        self.config = RainLLMConfig(
                dim=args.get('dim', 256),
                n_layers=args.get('n_layers', 16),
                max_seq_len=args.get('max_seq_len', 8192),
                stream=True
        )
        if os.path.isfile(model_path):
            self.model = RainLLM(self.config)
            self.model.load_config(self.model_config_path)
            self.model.init_model()
            state_dict = torch.load(model_path, map_location='cuda' if torch.cuda.is_available() else 'cpu')
            self.model.load_state_dict({k: v for k, v in state_dict.items() if 'mask' not in k}, strict=True)
        else:
            self.model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True)
        self.model.to(self.device)
        self.history = []
        print(f"RainLLM参数量{sum(p.numel() for p in self.model.parameters() if p.requires_grad) / 1e6:.2f}Million")

    def _get_input(self, message):
        messages = [
            {"role": "system", "content": self.config.system_prompt},
            {"role": "user", "content": message}
        ]
        trimmed_history = self.history[-self.config.history_cnt:] if self.config.history_cnt else []
        messages.extend(trimmed_history)
        template = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        model_inputs = self.tokenizer([template], return_tensors="pt").to(self.device)
        return model_inputs
    def generate(self, message) -> Generator | torch.Tensor:
        inputs = self._get_input(message)
        # inputs = torch.tensor(self.tokenizer(messages)['input_ids'], device=self.device).unsqueeze(0)
        with torch.no_grad():
            outputs = self.model.generate(input_ids=inputs['input_ids'],
                                          eos_token_id=self.tokenizer.eos_token_id,
                                          max_new_tokens=self.config.max_seq_len,
                                          temperature=0.85,
                                          top_p=0.85,
                                          pad_token_id=self.tokenizer.pad_token_id,
                                          stream=self.config.stream)

        if not self.config.stream:
            answer = self.tokenizer.decode(
                outputs[0][inputs["input_ids"].shape[1]:],
                skip_special_tokens=True
            )
            print(answer)
            # print(self.tokenizer.decode(outputs.squeeze()[outputs.shape[1]:].tolist(), skip_special_tokens=True), end='')
        else:
            history_idx = 0
            for y in outputs:
                answer = self.tokenizer.decode(y[0].tolist(), skip_special_tokens=True)
                if (answer and answer[-1] == '�') or not answer:
                    continue
                print(answer[history_idx:], end='', flush=True)
                history_idx = len(answer)
        self.history.append({"role": "user", "content": message})
        self.history.append({"role": "assistant", "content": answer})


if __name__ == '__main__':
    client = RainClient(tokenizer_path="./models/tokenizer",
                        model_path="D:/PythonCode/RainLLM/out/full_sft_256_out_learn.pth")
    print("start--->")
    while True:
        client.generate(input())

