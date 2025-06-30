import abc
import argparse
import math
import os.path
import time
from abc import ABC
from contextlib import nullcontext

import torch.cuda
import torch.distributed as dist
from torch import optim, nn
from torch.distributed.tensor.parallel import ddp
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, PreTrainedTokenizerFast

from components.dataset import PretrainDataset
from components.model import RainForCausalLM
from components.model_config import RainLLMConfig

CURRENT_DIR = os.path.dirname(__file__)


class Trainer(ABC):
    model: RainForCausalLM | None
    tokenizer: PreTrainedTokenizerFast | None
    train_config: dict

    def __init__(self, arg: argparse.Namespace):
        print(f'Trainer init....')
        self.model = None
        self.tokenizer = None
        self.llm_config = RainLLMConfig(
            dim=arg.dim,
            n_layers=arg.n_layers,
        )
        self.arg = arg
        self.device = arg.device
        self.tokens_per_iter = arg.batch_size * arg.max_seq_len
        wandb_run_name = f"Rain Train-{arg.epochs}-BatchSize-{arg.batch_size}-LearningRate-{arg.learning_rate}"

        base_seed = 1337
        torch.manual_seed(base_seed)
        torch.cuda.manual_seed(base_seed)
        if arg.use_wandb:
            import wandb
            # wandb.login('')
            self.wandb = wandb.init(project=arg.wandb_project, name=wandb_run_name)
        else:
            self.wandb = None

    def train(self, tokenizer_path):
        self.tokenizer, self.model = self.init_model(lm_config=self.llm_config, tokenizer_path=tokenizer_path)
        train_dataset = PretrainDataset(self.arg.data_path, self.tokenizer, max_length=self.arg.max_seq_len)
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.arg.batch_size,
            pin_memory=True,
            drop_last=False,
            shuffle=False,
            num_workers=self.arg.num_workers,
            sampler=None
        )
        self.scaler = torch.cuda.amp.GradScaler(enabled=(self.arg.dtype in ['float16', 'bfloat16']))
        self.optimizer = optim.AdamW(self.model.parameters(), lr=self.arg.learning_rate)
        iter_per_epoch = len(train_loader)
        ctx = nullcontext() if self.device == "cpu" else torch.cuda.amp.autocast()
        for epoch in range(self.arg.epochs):
            self.train_epoch(train_loader, epoch, iter_per_epoch, ctx)

    def Logger(self, content):
        print(content)

    def get_lr(self, current_step, total_steps, lr):
        """
        余弦退火
        """
        return lr / 10 + 0.5 * lr * (1 + math.cos(math.pi * current_step / total_steps))

    def init_model(self, lm_config, tokenizer_path):

        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        model = RainForCausalLM(config=lm_config).to(self.device)
        self.Logger(f'LLM可训练总参数量：{sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6:.3f} 百万')
        return tokenizer, model

    @abc.abstractmethod
    def train_epoch(self, train_loader, epoch, iter_per_epoch, ctx):
        raise ValueError


class PretrainTrainer(Trainer):

    def train_epoch(self, train_loader, epoch, iter_per_epoch, ctx):
        loss_fct = nn.CrossEntropyLoss(reduction='none')
        start_time = time.time()
        for step, (X, Y, loss_mask) in enumerate(train_loader):
            X = X.to(self.arg.device)
            Y = Y.to(self.arg.device)
            loss_mask = loss_mask.to(self.arg.device)

            lr = self.get_lr(epoch * iter_per_epoch + step, self.arg.epochs * iter_per_epoch, self.arg.learning_rate)
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr

            with ctx:
                res = self.model(X)
                loss = loss_fct(
                    res.logits.view(-1, res.logits.size(-1)),
                    Y.view(-1)
                ).view(Y.size())
                loss = (loss * loss_mask).sum() / loss_mask.sum()
                loss += res.aux_loss
                loss = loss / self.arg.accumulation_steps

            self.scaler.scale(loss).backward()

            if (step + 1) % self.arg.accumulation_steps == 0:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.arg.grad_clip)

                self.scaler.step(self.optimizer)
                self.scaler.update()

                self.optimizer.zero_grad(set_to_none=True)

            if step % self.arg.log_interval == 0:
                spend_time = time.time() - start_time
                self.Logger(
                    'Epoch:[{}/{}]({}/{}) loss:{:.3f} lr:{:.12f} epoch_Time:{}min:'.format(
                        epoch + 1,
                        self.arg.epochs,
                        step,
                        iter_per_epoch,
                        loss.item() * self.arg.accumulation_steps,
                        self.optimizer.param_groups[-1]['lr'],
                        spend_time / (step + 1) * iter_per_epoch // 60 - spend_time // 60))

                if (self.wandb is not None) and (not ddp):
                    self.wandb.log({"loss": loss.item() * self.arg.accumulation_steps,
                                    "lr": self.optimizer.param_groups[-1]['lr'],
                                    "epoch_Time": spend_time / (step + 1) * iter_per_epoch // 60 - spend_time // 60})

            if (step + 1) % self.arg.save_interval == 0 and (not ddp):
                self.model.eval()
                moe_path = '_moe' if self.llm_config.use_moe else ''
                ckp = f'{self.arg.save_dir}/pretrain_{self.llm_config.hidden_size}{moe_path}.pth'

                if isinstance(self.model, torch.nn.parallel.DistributedDataParallel):
                    state_dict = self.model.module.state_dict()
                else:
                    state_dict = self.model.state_dict()

                state_dict = {k: v.half() for k, v in state_dict.items()}  # 半精度保存
                self.wandb.log({f'now save at {ckp}'})
                torch.save(state_dict, ckp)
                self.model.train()


class SFTTrainer(Trainer):
    pass


class RLTrainer(Trainer):
    pass


class DPOTrainer(Trainer):
    pass


class TrainerFactory:
    @staticmethod
    def trainer(trainer: str, arg) -> Trainer:
        if trainer == "0":
            return PretrainTrainer(arg)
        elif trainer == "1":
            return SFTTrainer(arg)
        elif trainer == "2":
            return DPOTrainer(arg)
        elif trainer == "3":
            return RLTrainer(arg)
