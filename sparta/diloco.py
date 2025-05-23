import torch
import torch.distributed as dist
from tqdm import tqdm
from .config import DilocoSimulatorConfig
from .setup import DilocoSetup
from .eval import Evaluator
from .sparta import SpartaInterpolator
from dataclasses import dataclass
import wandb
import torch.nn.utils as nn_utils


@dataclass
class TrainStats:
    loss: float
    perplexity: float


class DilocoSimulator(Evaluator, SpartaInterpolator):

    def __init__(self, config: DilocoSimulatorConfig) -> None:
        super().__init__(config)

    def _average_models(self) -> None:
        for param in self.model.parameters():
            dist.all_reduce(param.data, op=dist.ReduceOp.SUM, group=self.dp_group)
            param.data /= self.ranks_per_stage

    def _broadcast_model_params(self) -> None:
        for param in self.model.parameters():
            dist.broadcast(param.data, src=self._get_stage_master(), group=self.dp_group)

    def _set_master_grad(self) -> None:
        for name, param in self.master_model.named_parameters():
            param.grad = param.data - self.model.state_dict()[name].data

    def _synchronize_master_model(self) -> None:
        for name, param in self.model.named_parameters():
            param.data = self.master_model.state_dict()[name].data

    def _outer_step(self) -> None:
        if self.config.method == 'diloco' or self.config.method == 'avg':
            self._average_models()

        if self.config.method == 'diloco':
            if self._is_stage_master():
                self.master_optimizer.zero_grad()
                self._set_master_grad()
                self.master_optimizer.step()
                self._synchronize_master_model()

            self._broadcast_model_params()
            
    def _train_step(self):
        x, y = self._get_batch()
        self.optimizer.zero_grad()
        mini_batch_size = self.config.max_minibatch_size or self.config.batch_size
        for i in range(0, len(x), mini_batch_size):
            x_mini = x[i : i + mini_batch_size]
            y_mini = y[i : i + mini_batch_size]
            output = self.model(x_mini)
            loss = self.config.loss_fn(output, y_mini)
            loss.backward()
        if self.config.max_norm:
            nn_utils.clip_grad_norm_(self.model.parameters(), max_norm=self.config.max_norm)
        self.optimizer.step()
        if self.scheduler:
            self.scheduler.step()

        if self.rank == 0:
            self._log_train(TrainStats(loss=loss.item(), perplexity=torch.exp(loss).item()))

        return loss.item()

    def _log_train(self, train_stats: TrainStats):
        lr = self.optimizer.param_groups[0]["lr"]
        self.pbar.update(1)
        self.pbar.set_postfix(
            {
                "loss": f"{train_stats.loss:.4f}",
                "perplexity": f"{train_stats.perplexity:.4f}",
                "lr": f"{lr:.4f}",
            }
        )

        if self.config.wandb_project is None:
            return
        wandb.log(
            {
                "step": self.local_step,
                "train_loss": train_stats.loss,
                "train_perplexity": train_stats.perplexity,
                "learning_rate": lr,
            }
        )

    def _train_loop(self):

        while self.local_step < self.max_local_step:

            if self.ranks_per_stage > 1:
                if self.config.p_sparta > 0.0 and self.local_step % self.sparta_interval == 0 and self.local_step > 0:
                    self._interpolate_models()

                if self.local_step % self.diloco_interval == 0 and self.local_step > 0:
                    self._outer_step()

            if self.local_step % self.eval_interval == 0:
                self._evaluate()

            self._train_step()

            self.local_step += self.config.num_inner_steps
            dist.barrier()
        self._evaluate()

    def _train(self, rank: int):
        try:
            self._setup(rank)
            self._train_loop()

            if self.rank == 0 and self.config.save_dir:
                self._save_checkpoint()
        finally:
            self._cleanup()

    def train(self):
        torch.multiprocessing.spawn(self._train, args=(), nprocs=self.config.num_nodes_per_instance, join=True)
