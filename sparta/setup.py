import torch
import os
from typing import Optional, Iterator, Tuple
from torch.distributed import init_process_group as init_process_group, destroy_process_group
import torch.distributed as dist
from copy import deepcopy
from torch.utils.data import DataLoader, DistributedSampler
from torch.optim.lr_scheduler import LambdaLR
from .config import DilocoSimulatorConfig
import wandb
from tqdm import tqdm
import math
import datetime
import random
class DilocoSetup:
    rank: int
    device: torch.device
    model: torch.nn.Module
    optimizer: torch.optim.Optimizer
    scheduler: Optional[torch.optim.lr_scheduler.CosineAnnealingLR] = None
    master_model: torch.nn.Module
    master_optimizer: torch.optim.Optimizer
    train_dataloader: DataLoader
    eval_dataloader: Optional[DataLoader] = None
    train_data_iter: Iterator[Tuple[torch.Tensor, torch.Tensor]]
    eval_data_iter: Optional[Iterator[Tuple[torch.Tensor, torch.Tensor]]] = None
    max_local_step: int
    local_step: int = 0
    epoch: int = 0
    pbar: Optional[tqdm] = None
    pp_id: int
    pp_stage: int
    ranks_per_stage: int
    pp_group: Optional[dist.ProcessGroup] = None
    dp_group: Optional[dist.ProcessGroup] = None
    backend: str = "nccl"

    def __init__(self, config: DilocoSimulatorConfig) -> None:
        self.config = config
        self.max_local_step = self.config.max_local_step
        if self.config.num_nodes_per_instance is None: # single instance
            self.config.num_nodes_per_instance = self.config.num_nodes
        assert self.config.num_nodes % self.config.num_nodes_per_instance == 0, 'num_nodes must be a multiple of num_nodes_per_instance'
        self.num_instances = self.config.num_nodes // self.config.num_nodes_per_instance
        print(f"Number of instances: {self.num_instances}, instance_id: {self.config.instance_id}")
        
        self.diloco_interval = self.config.diloco_interval
        self.async_sparta_delay = self.config.async_sparta_delay
        self.eval_interval = self.config.eval_interval
        self.sparta_interval = self.config.sparta_interval
        self.num_inner_steps = self.config.num_inner_steps
        
        self.ranks_per_stage = self.config.num_nodes // self.config.num_pp_stages
            
        if self.config.wandb_project:
            wandb.login()

    def _initialize_logging(self) -> None:
        print(f"DilocoSimulator initialized with config: {self.config}")
        self.pbar = tqdm(total=self.max_local_step)

        if self.config.wandb_project:
            wandb.init(project=self.config.wandb_project, config=self.config.__dict__, name=self.config.wandb_name)

    def _initialize_distributed(self, rank: int):
        os.environ["MASTER_ADDR"] = str(self.config.master_addr)
        os.environ["MASTER_PORT"] = str(self.config.port)
        self.rank = rank
        if self.config.backend is None:
            self.backend = "nccl" if torch.cuda.is_available() and len(self.config.devices) == self.config.num_nodes else "gloo"
        else:
            self.backend = self.config.backend

        init_process_group(
            backend=self.backend,
            # init_method="env://",
            rank=rank,
            world_size=self.config.num_nodes,
            timeout=datetime.timedelta(minutes=10),
        )
        self.device = torch.device(
            f"cuda:{self.config.devices[rank % len(self.config.devices)]}"
            if torch.cuda.is_available() and self.config.devices
            else "cpu"
        )
        torch.cuda.set_device(self.device) if self.device.type == "cuda" else None
        print(f"Initialized process group with rank {rank} on device {self.device}")

        self._setup_distributed_groups()

    def _setup_distributed_groups(self):
        assert self.config.num_nodes % self.config.num_pp_stages == 0, 'Only supports symmetric DP groups'
        # Calculate the number of ranks per PP stage
        self.pp_id = self.rank // self.config.num_pp_stages
        self.pp_stage = self.rank % self.config.num_pp_stages   # dp_id

        if self.config.num_pp_stages <= 1 or self.ranks_per_stage <= 1:
            self.pp_group = None
            self.dp_group = None
            return

        # Create PP groups
        pp_groups = []
        for i in range(self.ranks_per_stage):
            pp_ranks = list(range(i * self.config.num_pp_stages, (i + 1) * self.config.num_pp_stages))
            pp_group = dist.new_group(ranks=pp_ranks)
            pp_groups.append(pp_group)
            if self.rank == 0:
                print(f"PP group {i} created with ranks {pp_ranks}", flush=True)

        # Create DP groups for each PP stage
        dp_groups = []
        for i in range(self.config.num_pp_stages):
            dp_ranks = [i + j * self.config.num_pp_stages for j in range(self.ranks_per_stage)]
            dp_group = dist.new_group(ranks=dp_ranks)
            dp_groups.append(dp_group)
            if self.rank == 0:
                print(f"DP group {i} created with ranks {dp_ranks}", flush=True)

        self.pp_group = pp_groups[self.pp_id]
        self.dp_group = dp_groups[self.pp_stage]

    def _is_first_stage(self):
        return self.pp_stage == 0

    def _is_last_stage(self):
        return self.pp_stage == self.config.num_pp_stages - 1

    def _is_stage_master(self):
        return self.rank < self.config.num_pp_stages

    def _get_stage_master(self):
        return self.pp_stage

    def _cleanup(self):
        if self.rank == 0:
            wandb.finish()
        if self.pbar:
            self.pbar.close()
        if dist.is_initialized():
            destroy_process_group()

    def _setup_master_model(self):
        print("Setting up master model")
        self.master_model = deepcopy(self.model).to(self.device)
        for param in self.master_model.parameters():
            param.requires_grad = True

    def _setup_master_optimizer(self):
        print("Setting up master optimizer")
        self.master_optimizer = self.config.outer_optimizer_cls(
            self.master_model.parameters(), **self.config.outer_optimizer_kwargs
        )

    def _setup_model(self):
        if self.rank == 0:
            print("Setting up model")
        self.model = self.config.model_cls(**self.config.model_kwargs).to(self.device)
        for name, param in self.model.named_parameters():
            dist.broadcast(param.data, src=0, group=self.dp_group)

        self.model.train()

    def _setup_optimizer(self):
        if self.rank == 0:
            print("Setting up optimizer")
        self.optimizer = self.config.optimizer_cls(self.model.parameters(), **self.config.optimizer_kwargs)

    def _setup_scheduler(self):
        if self.rank == 0:
            print("Setting up scheduler")

        def lr_lambda(current_step):
            if current_step < self.config.warmup_steps:
                return float(current_step) / float(max(self.config.warmup_steps, 1))
            elif self.config.cosine_anneal:
                min_lr_factor = 0.1
                progress = (current_step - self.config.warmup_steps) / float(
                    max(1, self.max_local_step - self.config.warmup_steps)
                )
                cosine_term = 0.5 * (1.0 + math.cos(math.pi * progress))
                return (1 - min_lr_factor) * cosine_term + min_lr_factor
            else:
                return 1.0

        self.scheduler = LambdaLR(self.optimizer, lr_lambda)

    def _setup_train_dataloader(self):
        if self.rank == 0:
            print("Setting up train dataloader")
        sampler = None
        if not isinstance(self.config.train_dataset, torch.utils.data.IterableDataset):
            sampler = DistributedSampler(
                self.config.train_dataset, num_replicas=self.ranks_per_stage, rank=self.pp_id, shuffle=True, drop_last=True
            )  
        if sampler is None:
            self.config.train_dataset._shard_dataset(self.pp_id)
        self.train_dataloader = DataLoader(
            self.config.train_dataset, batch_size=self.config.batch_size, sampler=sampler, pin_memory=True, drop_last=True
        )
        self.train_data_iter = iter(self.train_dataloader)

    def _setup_eval_dataloader(self):
        if self.rank == 0:
            print("Setting up eval dataloader")
        shuffle = not isinstance(self.config.eval_dataset, torch.utils.data.IterableDataset)
        self.eval_dataloader = DataLoader(
            self.config.eval_dataset, batch_size=self.config.batch_size, pin_memory=True, shuffle=shuffle, drop_last=True
        )
        self.eval_data_iter = iter(self.eval_dataloader)

    def _save_checkpoint(self):
        torch.save(self.model.state_dict(), os.path.join(self.config.save_dir, f"model_{self.pp_stage}_{self.epoch}.pt"))

    def _get_batch(self, eval=False):
        if not eval or self.eval_data_iter is None:
            try:
                x, y = next(self.train_data_iter)
            except StopIteration:
                self.epoch += 1
                self.train_data_iter = iter(self.train_dataloader)
                x, y = next(self.train_data_iter)
        else:
            try:
                x, y = next(self.eval_data_iter)
            except StopIteration:
                self.eval_data_iter = iter(self.eval_dataloader)
                x, y = next(self.eval_data_iter)

        x, y = x.to(self.device), y.to(self.device)

        return x, y

    def _setup(self, rank: int):
        gl_rank = rank + self.config.instance_id * self.config.num_nodes_per_instance   # global rank
        self._initialize_distributed(gl_rank)
        self._setup_model()
        self._setup_optimizer()
        self._setup_scheduler()
        self._setup_train_dataloader()
        if self._is_stage_master():
            self._setup_master_model()
            self._setup_master_optimizer()
            if self.config.eval_dataset:
                self._setup_eval_dataloader()
        if self._is_last_stage() and self._is_stage_master():
            self._initialize_logging()
        dist.barrier()

    def load_model(self, path):
        self.master_model.load_state_dict(torch.load(path))
        for model in self.models:
            model.load_state_dict(self.master_model.state_dict())
