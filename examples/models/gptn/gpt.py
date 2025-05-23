import torch.nn as nn

from .stage_first import StageFirst
from .stage_mid import StageMid
from .stage_last import StageLast

class GPTLanguageModel(nn.Module):

    def __init__(self, n_embd=384, n_head=6, block_size=256, vocab_size=1000, dropout=0.0, n_layer=2):
        super().__init__()
        assert(n_layer >= 2)
        n_layer_mid = n_layer - 2
        stages = [StageFirst(n_embd, n_head, block_size, vocab_size, dropout, 1)]
        for _ in range(n_layer_mid):
            stages.append(StageMid(n_embd, n_head, block_size, dropout, 1))
        stages.append(StageLast(n_embd, n_head, block_size, vocab_size, dropout, 1))
        self.stages = nn.Sequential(*stages)

    def forward(self, input0):
        return self.stages(input0)

    