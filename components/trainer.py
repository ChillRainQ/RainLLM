import abc
import argparse
import math
import os.path
import time
from abc import ABC
from contextlib import nullcontext

import torch.cuda
from torch import optim, nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, PreTrainedTokenizerFast

import constant
from components.dataset import PretrainDataset, SFTDataset, StreamingSFTDataset
from components.model import RainForCausalLM
from components.model_config import RainLLMConfig

CURRENT_DIR = os.path.dirname(__file__)


class Trainer(ABC):
    model: RainForCausalLM | None
    tokenizer: PreTrainedTokenizerFast | None
    train_config: dict
    name: str

    def __init__(self, arg: argparse.Namespace):
        print(f'Trainer init....')
        self.model = None
        self.tokenizer = None
        self.llm_config = RainLLMConfig(
            dim=arg.dim,
            n_layers=arg.n_layers,
            ffn_type=arg.ffn

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

    def train_start(self, tokenizer_path):
        self.tokenizer, self.model = self.init_model(lm_config=self.llm_config, tokenizer_path=tokenizer_path)
        dataset = self.get_dataset()
        train_loader = DataLoader(
            dataset,
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
        self.model.eval()
        self.save()

    # def train_start(self, tokenizer_path):
    #     self.tokenizer, self.model = self.init_model(lm_config=self.llm_config, tokenizer_path=tokenizer_path)
    #     dataset = self.get_dataset()
    #
    #     train_loader = DataLoader(
    #         dataset,
    #         batch_size=self.arg.batch_size,
    #         pin_memory=True,
    #         drop_last=False,
    #         shuffle=False,  # IterableDataset 不支持 shuffle
    #         num_workers=self.arg.num_workers,
    #     )
    #
    #     self.scaler = torch.cuda.amp.GradScaler(enabled=(self.arg.dtype in ['float16', 'bfloat16']))
    #     self.optimizer = optim.AdamW(self.model.parameters(), lr=self.arg.learning_rate)
    #     ctx = nullcontext() if self.device == "cpu" else torch.cuda.amp.autocast()
    #
    #     for epoch in range(self.arg.epochs):
    #         self.train_epoch(train_loader, epoch, None, ctx)  # ❗️去掉 iter_per_epoch
    #     self.model.eval()
    #     self.save()

    def save(self):
        moe_path = '_moe' if self.llm_config.use_moe else ''
        if not os.path.isdir(self.arg.out_dir):
            os.mkdir(self.arg.out_dir)
        ckp = f'{self.arg.out_dir}/{self.arg.out_model_name}'
        # ckp = f'{self.arg.out_dir}/{self.name}_{self.llm_config.dim}_{self.llm_config.n_layers}{moe_path}.pth'
        if isinstance(self.model, torch.nn.parallel.DistributedDataParallel):
            state_dict = self.model.module.state_dict()
        else:
            state_dict = self.model.state_dict()
        state_dict = {k: v.half() for k, v in state_dict.items()}
        torch.save(state_dict, ckp)

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

    @abc.abstractmethod
    def get_dataset(self):
        raise ValueError


class PretrainTrainer(Trainer):
    name = "pretrain"

    def get_dataset(self):
        return PretrainDataset(os.path.join(self.arg.data_path, self.arg.data_name), self.tokenizer,
                               max_length=self.arg.max_seq_len)

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

                if self.wandb is not None:
                    self.wandb.log({"loss": loss.item() * self.arg.accumulation_steps,
                                    "lr": self.optimizer.param_groups[-1]['lr'],
                                    "epoch_Time": spend_time / (step + 1) * iter_per_epoch // 60 - spend_time // 60})

            if (step + 1) % self.arg.save_interval == 0:
                self.model.eval()
                self.save()
                self.model.train()


class SFTTrainer(Trainer):
    name = "full_sft"

    def get_dataset(self):
        return SFTDataset(os.path.join(constant.DATASET_DIR, self.arg.data_name), self.tokenizer,
                          max_length=self.arg.max_seq_len)

    def init_model(self, lm_config, tokenizer_path):

        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        model = RainForCausalLM(config=lm_config).to(self.device)
        ckp = os.path.join(constant.INPUT_DIR, self.arg.input_model_name)
        print(f"now use {ckp}")
        model.load_state_dict(torch.load(ckp, map_location=self.arg.device), strict=False)
        self.Logger(f'LLM可训练总参数量：{sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6:.3f} 百万')
        model = model.to(self.arg.device)
        return tokenizer, model

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

                if self.wandb is not None:
                    self.wandb.log({"loss": loss * self.arg.accumulation_steps,
                                    "lr": self.optimizer.param_groups[-1]['lr'],
                                    "epoch_Time": spend_time / (step + 1) * iter_per_epoch // 60 - spend_time // 60})

            if (step + 1) % self.arg.save_interval == 0:
                self.model.eval()
                self.save()
                self.model.train()
    # def train_epoch(self, train_loader, i, epoch, ctx):
    #     loss_fct = nn.CrossEntropyLoss(reduction='none')
    #     start_time = time.time()
    #     total_steps = 1000  # 🟡可配置最大步数（防止无限训练）
    #
    #     for step, (X, Y, loss_mask) in enumerate(train_loader):
    #         if step >= total_steps:
    #             break
    #
    #         X = X.to(self.arg.device)
    #         Y = Y.to(self.arg.device)
    #         loss_mask = loss_mask.to(self.arg.device)
    #
    #         global_step = 1 * total_steps + step  # 用 total_steps 替代 iter_per_epoch
    #         total_train_steps = self.arg.epochs * total_steps
    #         lr = self.get_lr(global_step, total_train_steps, self.arg.learning_rate)
    #         for param_group in self.optimizer.param_groups:
    #             param_group['lr'] = lr
    #
    #         with ctx:
    #             res = self.model(X)
    #             loss = loss_fct(
    #                 res.logits.view(-1, res.logits.size(-1)),
    #                 Y.view(-1)
    #             ).view(Y.size())
    #
    #             loss = (loss * loss_mask).sum() / loss_mask.sum()
    #             loss += res.aux_loss
    #             loss = loss / self.arg.accumulation_steps
    #
    #         self.scaler.scale(loss).backward()
    #
    #         if (step + 1) % self.arg.accumulation_steps == 0:
    #             self.scaler.unscale_(self.optimizer)
    #             torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.arg.grad_clip)
    #
    #             self.scaler.step(self.optimizer)
    #             self.scaler.update()
    #             self.optimizer.zero_grad(set_to_none=True)
    #
    #         if step % self.arg.log_interval == 0:
    #             spend_time = time.time() - start_time
    #             est_total = spend_time / (step + 1) * total_steps
    #             eta = est_total - spend_time
    #             self.Logger(
    #                 'Epoch:[{}/{}]({}/{}) loss:{:.3f} lr:{:.12f} ETA:{}min'.format(
    #                      1,
    #                     self.arg.epochs,
    #                     step,
    #                     total_steps,
    #                     loss.item() * self.arg.accumulation_steps,
    #                     self.optimizer.param_groups[-1]['lr'],
    #                     int(eta) // 60
    #                 )
    #             )
    #
    #             if self.wandb is not None:
    #                 self.wandb.log({
    #                     "loss": loss * self.arg.accumulation_steps,
    #                     "lr": self.optimizer.param_groups[-1]['lr'],
    #                     "epoch_time(min)": int(eta) // 60
    #                 })
    #
    #         if (step + 1) % self.arg.save_interval == 0:
    #             self.model.eval()
    #             moe_path = '_moe' if self.llm_config.use_moe else ''
    #             ckp = f'{self.arg.out_dir}/full_sft_{self.llm_config.hidden_size}_{self.llm_config.n_layers}{moe_path}.pth'
    #             state_dict = self.model.module.state_dict() if isinstance(self.model,
    #                                                                       torch.nn.parallel.DistributedDataParallel) else self.model.state_dict()
    #             state_dict = {k: v.half() for k, v in state_dict.items()}
    #             torch.save(state_dict, ckp)
    #             self.model.train()


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
