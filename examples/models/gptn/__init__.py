from .gpt import GPTLanguageModel
from .stage_first import StageFirst
from .stage_mid import StageMid
from .stage_last import StageLast

import json

def arch():
    return "gptn"

def model(criterion, vocab_size, block_size, dropout=0.0, n_layer=2, n_head=6, n_embd=384, stages=[1,1]):
    assert(n_layer >= 2)
    assert(sum(stages) == n_layer)
    n_stages = len(stages)

    pp_stages = []
    if n_stages == 1:
        pp_stages = [(gpt.GPTLanguageModel(vocab_size=vocab_size, block_size=block_size, dropout=dropout, n_layer=n_layer, n_head=n_head, n_embd=n_embd), ["input0"], ["output"])]
    elif n_stages == 2:
        pp_stages = [(StageFirst(vocab_size=vocab_size, block_size=block_size, dropout=dropout, n_head=n_head, n_embd=n_embd, n_layer=stages[0]), ["input0"], ["out0"])]
        pp_stages.append((StageLast(vocab_size=vocab_size, block_size=block_size, dropout=dropout, n_head=n_head, n_embd=n_embd, n_layer=stages[1]), ["out0"], ["output"]))
    else:
        pp_stages = [(StageFirst(vocab_size=vocab_size, block_size=block_size, dropout=dropout, n_head=n_head, n_embd=n_embd, n_layer=stages[0]), ["input0"], ["out0"])]
        for i in range(1, n_stages - 1):
            pp_stages.append((StageMid(block_size=block_size, dropout=dropout, n_head=n_head, n_embd=n_embd, n_layer=stages[i]), [f"out{i-1}"], [f"out{i}"]))
        pp_stages.append((StageLast(vocab_size=vocab_size, block_size=block_size, dropout=dropout, n_head=n_head, n_embd=n_embd, n_layer=stages[n_stages - 1]), [f"out{n_stages - 2}"], ["output"]))
    pp_stages.append((criterion, ["output"], ["loss"]))
    return pp_stages

def full_model(vocab_size, block_size, dropout=0.0, n_layer=2, n_head=6, n_embd=384):
    return GPTLanguageModel(vocab_size=vocab_size, block_size=block_size, dropout=dropout, n_layer=n_layer, n_head=n_head, n_embd=n_embd)
