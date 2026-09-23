import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint


def run_blocks(blocks, x, use_checkpoint=False):
    # DONE: apply each block in sequence to x
    # DONE: when use_checkpoint is True, run each block through
    #       checkpoint(...) with use_reentrant explicitly set to False
    for block in blocks:
        if use_checkpoint:
            x = checkpoint(block, x, use_reentrant=False)
        else:
            x = block(x)
    return x