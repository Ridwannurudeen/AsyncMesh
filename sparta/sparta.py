import torch
import torch.distributed as dist
from torch.optim import SGD
from .config import DilocoSimulatorConfig
from .setup import DilocoSetup
import math

class SparseSGD(SGD):
    def __init__(self, params, lr=0.01, momentum=0.0, weight_decay=0.0, 
                 adaptive_momentum=False, total_steps=30000, warmup_steps=1000, **kwargs):
        super(SparseSGD, self).__init__(params, lr=lr, momentum=momentum, weight_decay=weight_decay, **kwargs)
        self.adaptive_momentum = adaptive_momentum
        self.total_steps = total_steps
        self.warmup_steps = warmup_steps
        self.current_step = 0
        self.initial_momentum = momentum

    def step(self, indices_dict):
        """
        Perform a single optimization step.

        Args:
            indices_dict (dict): A dictionary where keys are parameters and values are the indices to update.
        """
        loss = None

        # Adjust momentum using cosine annealing if enabled and past warmup
        if self.adaptive_momentum and self.current_step >= self.warmup_steps:
            t = (self.current_step - self.warmup_steps) / (self.total_steps - self.warmup_steps - 1)
            new_momentum = self.initial_momentum + (0.99 - self.initial_momentum) * (1 - 0.5 * (1 + math.cos(math.pi * t)))
            for group in self.param_groups:
                group['momentum'] = new_momentum

        for group in self.param_groups:
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            nesterov = group['nesterov']
            lr = group['lr']

            for p in group['params']:
                if p.grad is None:
                    continue

                # Get the indices for this parameter
                indices = indices_dict.get(p, None)
                if indices is None:
                    continue

                d_p = p.grad

                if weight_decay != 0:
                    d_p[indices] = d_p[indices] + weight_decay * p.data[indices]

                if momentum != 0:
                    param_state = self.state[p]
                    if 'momentum_buffer' not in param_state:
                        buf = param_state['momentum_buffer'] = torch.clone(d_p).detach()
                    else:
                        buf = param_state['momentum_buffer']
                        buf[indices] = buf[indices] * momentum + d_p[indices] * (1 - momentum)  # ema of grads
                    if nesterov:
                        d_p[indices] = d_p[indices] + momentum * buf[indices]
                    else:
                        d_p[indices] = buf[indices]

                # Update only the selected indices
                p.data[indices] = p.data[indices] - lr * d_p[indices]
        
        self.current_step += 1

        return loss


class SpartaInterpolator(DilocoSetup):

    def __init__(self, config: DilocoSimulatorConfig) -> None:
        super().__init__(config)
        self.index_selector = PartitionedIndexSelector(self.config.p_sparta)
        self.buffer: dict[torch.nn.Parameter, list[tuple[torch.Tensor, torch.Tensor]]] = {}  # param-specific buffers
        self.grad_buffer: dict[torch.nn.Parameter, list[tuple[torch.Tensor, torch.Tensor]]] = {}
    
    def _init_sparta_optimizer(self):
        self.sparta_optimizer = SparseSGD(self.model.parameters(), **self.config.sparta_optimizer_kwargs)

    def _interpolate_models(self):
        indices_dict = {}
        with torch.no_grad():
            if 'ema' == self.config.sparta_method:
                self.sparta_optimizer.zero_grad()
            for param in self.model.parameters():
                if not param.requires_grad:
                    continue
                if param not in self.buffer:
                    self.buffer[param] = []  # Initialize buffer for each param
                    self.grad_buffer[param] = []
                indices = self.index_selector.get_indices(param)
                dist.broadcast(indices, src=self._get_stage_master(), group=self.dp_group)
                sparse_data = param.data[indices]
                dist.all_reduce(sparse_data, op=dist.ReduceOp.SUM, group=self.dp_group)
                sparse_data /= self.ranks_per_stage
                sparse_grad = sparse_data - param.data[indices]
                if self.config.buffer_to_cpu:
                    self.buffer[param].append((indices.cpu(), sparse_data.cpu()))   # average params
                    self.grad_buffer[param].append((indices.cpu(), sparse_grad.cpu()))   # avg - param_i
                else:
                    self.buffer[param].append((indices, sparse_data))   # average params
                    self.grad_buffer[param].append((indices, sparse_grad))   # avg - param_i
                
                # simulating async sparta with buffering and specified delay
                if len(self.buffer[param]) > self.async_sparta_delay:
                    indices_popped, sparse_data_popped = self.buffer[param].pop(0)
                    _, sparse_grad_popped = self.grad_buffer[param].pop(0)
                    if self.config.buffer_to_cpu:
                        indices_popped = indices_popped.to(param.device)
                        sparse_data_popped = sparse_data_popped.to(param.device)
                        sparse_grad_popped = sparse_grad_popped.to(param.device)
                    param_data = param.data[indices_popped].clone()

                    if self.config.sparta_method == 'avg':
                        param.masked_scatter_(indices_popped, sparse_data_popped)
                    elif self.config.sparta_method == 'weight_update':         
                        # w{t+1} = w-avg{t-\tau} + lambda * (w{t} - w{t-\tau})               
                        sparse_data_popped = sparse_data_popped + (param_data - (sparse_data_popped - sparse_grad_popped)) * self.config.sparta_lambda
                        param.masked_scatter_(indices_popped, sparse_data_popped)
                    elif self.config.sparta_method == 'ema':
                        # w{t+1} = w-avg{t-\tau} + lambda * ema(w{t} - w{t-\tau})
                        # Set sparse gradients
                        param.grad = torch.zeros_like(param.data)
                        param.grad[indices_popped] = -(param_data - (sparse_data_popped - sparse_grad_popped))
                        # Collect indices for the custom optimizer
                        indices_dict[param] = indices_popped
                        param.masked_scatter_(indices_popped, sparse_data_popped)
                        # update in optimizer step
                    else:
                        raise ValueError(f"Invalid sparta method: {self.config.sparta_method}")
        # return indices_dict
        if 'ema' == self.config.sparta_method:
            # Perform optimizer step with the collected indices
            self.sparta_optimizer.step(indices_dict)


class IndexSelector:
    def __init__(self, p):
        self.state = {}
        self.p = p

    def get_indices(self, param):
        return torch.ones(param.shape).bool()


class RandomIndexSelector(IndexSelector):
    def get_indices(self, param):
        return torch.bernoulli(torch.full(param.shape, self.p, device=param.device)).bool()


class PartitionedIndexSelector(IndexSelector):
    def __init__(self, p):
        super().__init__(p)

    def _set_partition(self, param):
        param_state = self.state[param]
        param_state["curr_partition"] = 0
        param_state["num_partitions"] = min(math.ceil(1 / self.p), param.numel())
        param_state["partitions"] = (
            torch.rand(param.numel(), device=param.device).argsort().view(param.shape) % param_state["num_partitions"]
        )

    def get_indices(self, param):
        if param not in self.state:
            self.state[param] = {}
            self._set_partition(param)
        elif self.state[param]["curr_partition"] >= self.state[param]["num_partitions"]:
            self._set_partition(param)

        indices = (self.state[param]["partitions"] == self.state[param]["curr_partition"]).bool()

        self.state[param]["curr_partition"] += 1

        return indices

class TopKIndexSelector(IndexSelector):
    def __init__(self, p):
        super().__init__(p)

    def get_indices(self, param):
        # Flatten the parameter tensor and get the absolute values
        flat_param = param.view(-1)
        abs_param = flat_param.abs()

        # Determine the number of top elements to select
        k = max(1, int(self.p * flat_param.numel()))

        # Get the indices of the top-k elements
        topk_indices = abs_param.topk(k, largest=True).indices

        # Create a boolean mask with the same shape as the parameter tensor
        mask = torch.zeros_like(flat_param, dtype=torch.bool)
        mask[topk_indices] = True

        # Reshape the mask to the original parameter shape
        return mask.view(param.shape)
    