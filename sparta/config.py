from dataclasses import dataclass, field
from typing import Optional, Callable, Type
import torch


@dataclass
class DilocoSimulatorConfig:
    model_cls: Type[torch.nn.Module]
    model_kwargs: dict
    loss_fn: Callable[..., torch.Tensor]
    train_dataset: torch.utils.data.Dataset
    optimizer_kwargs: dict
    optimizer_cls: Type[torch.optim.Optimizer] = torch.optim.AdamW
    batch_size: int = 16
    eval_dataset: Optional[torch.utils.data.Dataset] = None
    ckpt_interval: Optional[int] = None  # num of outersteps to save model
    eval_iters: int = 400
    eval_interval: int = 1000
    save_dir: Optional[str] = None
    p_sparta: float = 0.0
    sparta_interval: int = 1
    cosine_anneal: bool = False
    warmup_steps: int = 0
    model_path: Optional[str] = None
    num_nodes: int = 4
    num_nodes_per_instance: Optional[int] = None
    instance_id: int = 0
    diloco_interval: int = 500
    outer_optimizer_cls: Type[torch.optim.Optimizer] = torch.optim.SGD
    outer_optimizer_kwargs: dict = field(default_factory=lambda: {"lr": 0.7, "nesterov": True, "momentum": 0.9})
    max_local_step: Optional[int] = None
    wandb_project: Optional[str] = None
    max_minibatch_size: Optional[int] = None
    master_addr: str = "127.0.0.1"
    port: int = 12355
    devices: Optional[list[int]] = None
    max_norm: Optional[float] = None
    async_sparta_delay: int = 0
    wandb_name: Optional[str] = None
    num_pp_stages: int = 2
    num_microbatches: int = 2
    adaptive_momentum: bool = False
    num_inner_steps: int = 1
    backend: Optional[str] = None
    method: Optional[str] = 'diloco'
    sparta_method: Optional[str] = 'avg'
    sparta_lambda: Optional[float] = 0.9
    sparta_optimizer_kwargs: dict = field(default_factory=lambda: {"lr": 1.0, "momentum": 0.0, "nesterov": False})
    buffer_to_cpu: bool = False