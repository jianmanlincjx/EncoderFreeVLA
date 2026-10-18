"""Activation-checkpoint wrapper used *inside* an FSDP2 unit: shard the wrapper, the inner module has no hooks."""
import torch.nn as nn
from torch.utils.checkpoint import checkpoint


class CkptWrap(nn.Module):
    def __init__(self, inner: nn.Module):
        super().__init__()
        self.inner = inner
        self.grad_ckpt = False

    def forward(self, *args, **kwargs):
        if self.grad_ckpt and self.training:
            return checkpoint(self.inner, *args, use_reentrant=False, **kwargs)
        return self.inner(*args, **kwargs)
