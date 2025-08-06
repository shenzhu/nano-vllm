import torch
from torch import nn
import torch.nn.functional as F


class SiluAndMul(nn.Module):

    def __init__(self):
        super().__init__()

    @torch.compile
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Chunks the output to 2 parts and apply SwiGLU(silu and element-wise multiply)
        x, y = x.chunk(2, -1)
        return F.silu(x) * y
