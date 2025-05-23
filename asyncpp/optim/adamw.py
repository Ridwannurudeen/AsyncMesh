# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from torch.optim.optimizer import required

from .optimizer import OptimizerWithWeightStashing

class AdamWWithWeightStashing(OptimizerWithWeightStashing):
    """
    Adam optimizer with weight stashing.
    """
    def __init__(self, modules, master_parameters, model_parameters,
                 num_versions, lr=required, betas=(0.9,0.999), loss_scale=1.,
                 weight_decay=0, 
                 verbose_freq=0, macrobatch=False, 
                 clip_grad=None, save_dir=None, stash_to_cpu=False):
        super(AdamWWithWeightStashing, self).__init__(
            optim_name='AdamW',
            modules=modules, master_parameters=master_parameters,
            model_parameters=model_parameters, loss_scale=loss_scale,
            num_versions=num_versions, lr=lr, betas=betas,
            weight_decay=weight_decay, 
            verbose_freq=verbose_freq, macrobatch=macrobatch, 
            clip_grad=clip_grad, save_dir=save_dir, stash_to_cpu=stash_to_cpu
        )
