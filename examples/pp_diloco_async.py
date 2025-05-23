import torch
import torch.nn as nn
import torch.distributed as dist
import sys
import math
import argparse
import os
import numpy as np
import importlib
from data_utils import ShakespeareDataset, WikiTextDataset, OpenWebTextDataset, BookCorpusDataset, FineWebDataset
from transformers import AutoTokenizer
from copy import deepcopy

sys.path.append("..")
from sparta import DilocoSimulator, DilocoSimulatorConfig, TrainStats, EvalStats
from asyncpp import StageRuntime, NAdamWithWeightStashing, AdamWWithWeightStashing

# Define the PPTrainer class
class PPAsyncTrainer(DilocoSimulator):
    def __init__(self, config: DilocoSimulatorConfig):
        super().__init__(config)

    def _setup_pipeline(self):
        if self.config.num_pp_stages <= 1:
            self.model = nn.Sequential(*[s[0] for s in self.model[:-1]]).to(self.device)
            self.config.loss_fn = reshaped_cross_entropy
            return

        assert self.config.batch_size % self.config.num_microbatches == 0, 'Batch size must be divisible by number of microbatches'

        # determine shapes of all tensors in passed-in model
        microbatch_size = self.config.batch_size // self.config.num_microbatches
        input_size = [microbatch_size, self.config.model_kwargs['block_size']]
        training_tensor_shapes = {"input0": input_size, "target": input_size}
        dtypes = {"input0": torch.int64, "target": torch.int64}
        inputs_module_destinations = {"input": 0}
        target_tensor_names = {"target"}
        for (stage, inputs, outputs) in self.model[:-1]:        # Skip last layer (loss):  
            input_tensors = []
            for input in inputs:
                input_tensor = torch.zeros(tuple(training_tensor_shapes[input]),
                                           dtype=dtypes[input])
                input_tensors.append(input_tensor)
            with torch.no_grad():
                output_tensors = stage(*tuple(input_tensors))
            if not type(output_tensors) is tuple:
                output_tensors = [output_tensors]
            for output, output_tensor in zip(outputs,
                                             list(output_tensors)):
                training_tensor_shapes[output] = list(output_tensor.size())
                dtypes[output] = output_tensor.dtype

        eval_tensor_shapes = {}
        for key in training_tensor_shapes:
            eval_tensor_shapes[key] = tuple([microbatch_size] + training_tensor_shapes[key][1:])
            training_tensor_shapes[key] = tuple(training_tensor_shapes[key])
        
        # create sequential stage map
        if self.pp_group is not None:
            pp_ranks = dist.get_process_group_ranks(self.pp_group)
        else:
            pp_ranks = list(range(self.config.num_pp_stages))
        configuration_maps = {
            'module_to_stage_map': [i for i in range(self.config.num_pp_stages)] + [self.config.num_pp_stages-1], 
            'stage_to_rank_map': {i: [pp_ranks[i]] for i in range(self.config.num_pp_stages)}, 
            'stage_to_depth_map': None
        }
        # Create pipeline, ie, pipedream runtime
        self.pipeline = StageRuntime(
            model=self.model, distributed_backend=self.backend,
            training_tensor_shapes=training_tensor_shapes,
            eval_tensor_shapes=eval_tensor_shapes,
            training_tensor_dtypes=dtypes,
            inputs_module_destinations=inputs_module_destinations,
            target_tensor_names=target_tensor_names,
            configuration_maps=configuration_maps,
            rank=self.rank, local_rank=self.device.index
        )       
        self.model = self.pipeline.model()
    
    def _setup_model(self):
        if self.rank == 0:
            print("Setting up model")
        self.model = self.config.model_cls(**self.config.model_kwargs)
        self._setup_pipeline()

        if self.ranks_per_stage > 1:
            for param in self.model.parameters():
                dist.broadcast(param.data, src=self._get_stage_master(), group=self.dp_group)

        if 'ema' == self.config.sparta_method:
            self._init_sparta_optimizer()

    def _setup_optimizer(self):
        if self.config.num_pp_stages <= 1:
            self.config.optimizer_cls = torch.optim.AdamW
            super()._setup_optimizer()
            return
        
        if self.rank == 0:
            print("Setting up optimizer")

        num_versions = self.pipeline.num_warmup_minibatches + 1
        print(f"## Stage: {self.pp_stage}, Num versions: {num_versions}", flush=True)
        beta1, beta2 = self.config.optimizer_kwargs.get('betas', (0.9, 0.999))
        if self.config.adaptive_momentum:
            beta1 = beta1 + (num_versions - 1) * (0.99 - beta1) / self.config.num_pp_stages
            assert beta1 <= 1.0 and beta1 >= 0.0
        self.config.optimizer_kwargs['betas'] = (beta1, beta2)
        self.optimizer = self.config.optimizer_cls(self.pipeline.modules(), self.pipeline.master_parameters,
                                          self.pipeline.model_parameters, num_versions=num_versions,
                                          clip_grad=self.config.max_norm, **self.config.optimizer_kwargs)

    def _train_step(self):
        if self.config.num_pp_stages <= 1:
            for i in range(self.num_inner_steps):
                super()._train_step()
                # do sparta if enabled
                if self.ranks_per_stage > 1 and self.config.p_sparta > 0.0 and i % self.sparta_interval == 0:
                    self._interpolate_models()
            return 

        self.pipeline.train(self.num_inner_steps)
        self.pipeline.set_batch_sampler(self._get_batch if self._is_first_stage() else None, is_eval=False)
        # reset weight stashes
        self.optimizer.initialize_queue()

        for i in range(self.pipeline.num_warmup_minibatches):
            self.pipeline.run_forward()

        for i in range(self.num_inner_steps - self.pipeline.num_warmup_minibatches):
            self.pipeline.run_forward()

            if self._is_last_stage() and self._is_stage_master():
                loss = self.pipeline.loss.item()
                self._log_train(TrainStats(loss=loss, perplexity=math.exp(loss)))

            self.optimizer.zero_grad()
            self.optimizer.load_old_params()    # load old weights for backprop
            self.pipeline.run_backward()
            self.optimizer.load_new_params()   # load new weights for optimizer step
            self.optimizer.step()
            if self.scheduler: self.scheduler.step()

            # do sparta if enabled
            if self.ranks_per_stage > 1 and self.config.p_sparta > 0.0 and i % self.sparta_interval == 0 and i > 0:
                self._interpolate_models()
                # since sparta updates the model, we need to replace the latest stash with the new params
                self.optimizer.insert_to_queue(self.optimizer.get_params(clone=False), -1)

        for i in range(self.num_inner_steps - self.pipeline.num_warmup_minibatches, self.num_inner_steps):
            self.optimizer.zero_grad()
            self.optimizer.load_old_params()    # load old weights for backprop
            self.pipeline.run_backward()
            self.optimizer.load_new_params()   # load new weights for optimizer step
            self.optimizer.step()
            if self.scheduler: self.scheduler.step()

            # do sparta if enabled
            if self.ranks_per_stage > 1 and self.config.p_sparta > 0.0 and i % self.sparta_interval == 0 and i > 0:
                self._interpolate_models()
                # since sparta updates the model, we need to replace the latest stash with the new params
                self.optimizer.insert_to_queue(self.optimizer.get_params(clone=False), -1)

        # wait for all helper threads to complete
        self.pipeline.wait()

    def _evaluate(self):
        if self.config.num_pp_stages <= 1:
            super()._evaluate()
            return     

        if self.ranks_per_stage > 1:
            original_state_dict = {k: v.clone() for k, v in self.model.state_dict().items()}
            for param in self.model.parameters():
                dist.all_reduce(param.data, op=dist.ReduceOp.SUM, group=self.dp_group)
                param.data /= self.ranks_per_stage

        losses = []
        # num_batches = math.ceil(self.config.eval_iters / self.config.batch_size)
        num_batches = self.config.eval_iters
        assert num_batches > self.pipeline.num_warmup_minibatches, f"Number of batches must be greater than number of warmup minibatches: {num_batches} > {self.pipeline.num_warmup_minibatches}"
        
        self.pipeline.eval(num_batches)
        self.pipeline.set_batch_sampler(self._get_batch if self._is_first_stage() else None, is_eval=True)
        
        with torch.no_grad():
            for i in range(self.pipeline.num_warmup_minibatches):
                self.pipeline.run_forward()

            for i in range(num_batches - self.pipeline.num_warmup_minibatches):
                self.pipeline.run_forward()
                self.pipeline.run_ack()
                if self._is_last_stage():
                    losses.append(self.pipeline.loss.item())

            for i in range(self.pipeline.num_warmup_minibatches):
                self.pipeline.run_ack()
        
        if self._is_last_stage():
            avg_loss = sum(losses) / len(losses)

        if self._is_stage_master() and self._is_last_stage():
            print(f"Eval Loss: {avg_loss:.4f}, Eval Perplexity: {math.exp(avg_loss):.4f}")
            self._log_eval(EvalStats(loss=avg_loss, perplexity=math.exp(avg_loss)))

        if self.ranks_per_stage > 1:
            self.model.load_state_dict(original_state_dict)
        # wait for all helper threads to complete
        self.pipeline.wait()

    def _train_loop(self):

        while self.local_step < self.max_local_step:

            if self.ranks_per_stage > 1:

                if self.local_step % self.config.diloco_interval == 0 and self.local_step > 0:
                    self._outer_step()

            if self.local_step % self.eval_interval == 0:
                self._evaluate()

            self._train_step()

            self.local_step += self.num_inner_steps
            dist.barrier()

            if self.local_step > 5000:  # test
                break
        self._evaluate()


def seed_torch(deterministic=False, seed=1337):
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    if deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    else:
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

def get_dataset(args):
    print(f"Loading dataset: {args.dataset}")
    #  and create datasets
    tokenizer = AutoTokenizer.from_pretrained("google/byt5-small" if args.dataset == "shakespeare" else "gpt2")
    tokenizer.pad_token = tokenizer.eos_token  # Use EOS token as padding token
    vocab_size = tokenizer.vocab_size
    if args.dataset == "shakespeare":        
        train_dataset = ShakespeareDataset(tokenizer=tokenizer, train=True, block_size=args.block_size, dp_chunks=args.dp_chunks)
        val_dataset = ShakespeareDataset(tokenizer=tokenizer, train=False, block_size=args.block_size)
    elif args.dataset == "wikitext-103-v1":
        train_dataset = WikiTextDataset(tokenizer=tokenizer, train=True, block_size=args.block_size, dp_chunks=args.dp_chunks)
        val_dataset = WikiTextDataset(tokenizer=tokenizer, train=False, block_size=args.block_size)
    elif args.dataset == "openwebtext":
        train_dataset = OpenWebTextDataset(tokenizer=tokenizer, train=True, block_size=args.block_size, dp_chunks=args.dp_chunks)
        val_dataset = OpenWebTextDataset(tokenizer=tokenizer, train=False, block_size=args.block_size)
    elif args.dataset == "bookcorpus":
        train_dataset = BookCorpusDataset(tokenizer=tokenizer, train=True, block_size=args.block_size, dp_chunks=args.dp_chunks)
        val_dataset = BookCorpusDataset(tokenizer=tokenizer, train=False, block_size=args.block_size)
    elif args.dataset == "fineweb":
        train_dataset = FineWebDataset(tokenizer=tokenizer, train=True, block_size=args.block_size, dp_chunks=args.dp_chunks)
        val_dataset = FineWebDataset(tokenizer=tokenizer, train=False, block_size=args.block_size)
    else:
        raise Exception("Invalid dataset name")

    return train_dataset, val_dataset, vocab_size

def reshaped_cross_entropy(outputs, targets):
    loss_fn = nn.CrossEntropyLoss()
    sz = targets.numel()
    outputs = outputs.reshape(sz, -1)
    targets = targets.reshape(-1)
    return loss_fn(outputs, targets)

def main(args):
    seed_torch(args.deterministic, args.seed)

    args.num_pp_stages = len(args.stages)
    args.dp_chunks = args.num_nodes // args.num_pp_stages

    # Load dataset from HuggingFace
    train_dataset, val_dataset, vocab_size = get_dataset(args)
    if args.dataset != "fineweb":   # streaming dataset
        print(f"Train dataset: {len(train_dataset)}")
        print(f"Val dataset: {len(val_dataset)}")
        print(f"Vocab size: {vocab_size}")

    # define loss function (criterion)
    criterion = nn.CrossEntropyLoss()
    
    # create stages of the model
    module = importlib.import_module(args.module)
    args.arch = module.arch()
    if args.arch == "gptn":
        model = module.model(criterion, vocab_size=vocab_size, block_size=args.block_size, 
                         n_embd=args.n_embd, n_head=args.n_head, n_layer=args.n_layer, stages=args.stages)
    else:
        raise Exception("Invalid architecture name")

    args.nparams = float(sum(sum(p.numel() for p in s.parameters()) for s, _, _ in model[:-1])) / 1e6
    print(f"#Params: {args.nparams:.2f} M")

    if args.optimizer == "nadamw":
        optimizer_cls = NAdamWithWeightStashing
    elif args.optimizer == "adamw":
        optimizer_cls = AdamWWithWeightStashing
    else:
        raise Exception("Invalid optimizer name")

    # Create diloco config
    config = DilocoSimulatorConfig(
        model_cls=module.model,
        model_kwargs={"vocab_size": vocab_size, "block_size": args.block_size, "criterion": criterion,
                      "n_embd": args.n_embd, "n_head": args.n_head, "n_layer": args.n_layer, 
                      "stages": args.stages},
        optimizer_kwargs={
            "weight_decay": args.weight_decay,
            "lr": args.learning_rate,
            "betas": (args.beta1, args.beta2),
        },
        sparta_optimizer_kwargs={
            "lr": args.sparta_lambda,
            "momentum": args.sparta_momentum,
            "nesterov": args.sparta_nesterov,
            "adaptive_momentum": args.sparta_adaptive_momentum,
            "total_steps": args.max_local_step,
            "warmup_steps": args.sparta_warmup_steps,
        },
        loss_fn=criterion,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        batch_size=args.batch_size,
        save_dir=args.checkpoint_dir,
        eval_iters=args.eval_iters,
        ckpt_interval=args.ckpt_interval,
        num_nodes=args.num_nodes,
        diloco_interval=args.diloco_interval,
        devices=args.devices,
        p_sparta=args.p_sparta,
        cosine_anneal=args.cosine_anneal,
        warmup_steps=args.warmup_steps,
        max_local_step=args.max_local_step,
        wandb_project=args.wandb_project,
        port=args.port,
        async_sparta_delay=args.async_sparta_delay,
        wandb_name=args.wandb_name,
        eval_interval=args.eval_interval,
        num_pp_stages=len(args.stages),
        num_microbatches=args.num_microbatches,
        max_norm=args.max_norm,
        num_inner_steps=args.num_inner_steps,
        adaptive_momentum=args.adaptive_momentum,
        optimizer_cls=optimizer_cls,
        backend=args.backend,
        sparta_interval=args.sparta_interval,
        method=args.method,
        sparta_method=args.sparta_method,
        sparta_lambda=args.sparta_lambda,
        instance_id=args.instance_id,
        num_nodes_per_instance=args.num_nodes_per_instance,
        master_addr=args.master_addr,
        buffer_to_cpu=args.buffer_to_cpu,
    )

    # Create checkpoint directory if it doesn't exist
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    # Instantiate the trainer
    trainer = PPAsyncTrainer(config)

    # Run the training loop
    trainer.train()


# Main function to run the training
if __name__ == "__main__":
    # Command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="shakespeare", help="which dataset to use")
    parser.add_argument("--num_nodes", type=int, default=2)
    parser.add_argument("--devices", type=lambda s: [int(item) for item in s.split(',')], default=[0])
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--module", type=str, default="models.gptn", help="which module to use")
    parser.add_argument("--block_size", type=int, default=1024)
    parser.add_argument("--n_embd", type=int, default=768, help="embedding dimensionality")
    parser.add_argument("--n_layer", type=int, default=12, help="number of layers")
    parser.add_argument("--n_head", type=int, default=12, help="number of attention heads")
    parser.add_argument("--stages", type=lambda s: [int(item) for item in s.split(',')], default=[6,6], help="Stage split for PP")
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-2)
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.999)
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints")
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--p_sparta", type=float, default=0.0)
    parser.add_argument('--eval_iters', type=int, default=25, help='Number of evaluation iterations')
    parser.add_argument('--ckpt_interval', type=int, default=1000, help='Checkpoint interval')
    parser.add_argument('--diloco_interval', type=int, default=10000000, help='Diloco interval')    # disable diloco by default
    parser.add_argument('--cosine_anneal', type=bool, default=True, help='Use cosine annealing')
    parser.add_argument('--warmup_steps', type=int, default=3000, help='Number of warmup steps')
    parser.add_argument('--max_local_step', type=int, default=30000, help='Maximum local step')
    parser.add_argument('--wandb_project', type=str, default=None, help='WandB project name')
    parser.add_argument('--port', type=int, default=12345, help='Port number')
    parser.add_argument('--async_sparta_delay', type=int, default=0, help='Async Sparta delay')
    parser.add_argument('--wandb_name', type=str, default=None, help='WandB name')
    parser.add_argument('--eval_interval', type=int, default=1000, help='Evaluation interval')
    parser.add_argument('--num_microbatches', type=int, default=1, help='Number of microbatches')
    parser.add_argument('--deterministic', type=bool, default=False, help='Deterministic training')
    parser.add_argument('--max_norm', type=float, default=1.0, help='Maximum norm')
    parser.add_argument('--num_inner_steps', type=int, default=1000, help='Number of inner steps')
    parser.add_argument('--adaptive_momentum', type=bool, default=False, help='Use adaptive momentum')
    parser.add_argument('--optimizer', type=str, default="nadamw", help='Optimizer class')
    parser.add_argument('--backend', type=str, default="gloo", help='Backend')
    parser.add_argument('--sparta_interval', type=int, default=1, help='Sparta interval')
    parser.add_argument('--method', type=str, default='diloco', help='Method')
    parser.add_argument('--sparta_method', type=str, default='avg', help='Sparta method')
    parser.add_argument('--sparta_lambda', type=float, default=1.0, help='Sparta lambda')
    parser.add_argument('--sparta_momentum', type=float, default=0.5, help='Sparta momentum')
    parser.add_argument('--sparta_nesterov', type=bool, default=False, help='Sparta nesterov')
    parser.add_argument('--sparta_adaptive_momentum', type=bool, default=True, help='Sparta adaptive momentum')
    parser.add_argument('--sparta_warmup_steps', type=int, default=1000, help='Number of warmup steps')
    parser.add_argument("--instance_id", type=int, default=0, help="Instance ID")
    parser.add_argument("--num_nodes_per_instance", type=int, default=None, help="Number of nodes per instance")
    parser.add_argument('--master_addr', type=str, default="127.0.0.1", help='Master address for distributed training')
    parser.add_argument('--buffer_to_cpu', type=bool, default=False, help='Buffer to CPU')
    args = parser.parse_args()

    main(args)
