import json
import math
import os.path

import torch

import out.resources
from components.model import RainLLM
from components.model_config import RainLLMConfig


def get_lr(current_step, total_steps, lr):
    return lr / 10 + 0.5 * lr * (1 + math.cos(math.pi * current_step / total_steps))
CURRENT_DIR = os.path.dirname(__file__)
output_path = out.resources.CURRENT_DIR
class Trainer:
    model: RainLLM
    def __init__(self, config: RainLLMConfig):
        self.llm_config = config
        self.model = RainLLM(self.llm_config)

    def model_init(self, model_path):
        pass

    def model_save(self, output_name, train_type):
        output_dir = str(os.path.join(output_path, output_name))
        model_ckp = output_dir + f'/model_{train_type}.pth'
        runtime_data_ckp = output_dir + f'/model_runtime_data_{train_type}.json'
        if isinstance(self.model, torch.nn.parallel.DistributedDataParallel):
            state_dict = self.model.module.state_dict()
        else:
            state_dict = self.model.state_dict()
        torch.save(state_dict, model_ckp)
        with open(runtime_data_ckp, 'w', encoding="utf-8") as f:
            json.dump(self.model.runtime_data, f, indent=4)


    def train(self, mode: str, output_dir):
        """
         pre -> pre_train
         sft -> sft_train
         rl -> rl_train
        :param mode:
        :return:
        """
        if mode == "pre":
            self._pre_train()
        elif mode == "sft":
            self._sft_train()
        elif mode == "rl":
            self._rl_train()

    def _pre_train(self, dataset):
        pass
    def _sft_train(self, dataset):
        pass
    def _rl_train(self, dataset):
        pass