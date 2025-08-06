import torch
from torch import nn
import torch.nn.functional as F
import torch.distributed as dist


def divide(numerator, denominator):
    assert numerator % denominator == 0
    return numerator // denominator


class LinearBase(nn.Module):

    def __init__(
        self,
        input_size: int,
        output_size: int,
        tp_dim: int | None = None,
    ):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.tp_dim = tp_dim
        self.tp_rank = dist.get_rank()
        self.tp_size = dist.get_world_size()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class ReplicatedLinear(LinearBase):

    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = False,
    ):
        super().__init__(input_size, output_size)
        self.weight = nn.Parameter(torch.empty(self.output_size, self.input_size))
        self.weight.weight_loader = self.weight_loader
        if bias:
            self.bias = nn.Parameter(torch.empty(self.output_size))
            self.bias.weight_loader = self.weight_loader
        else:
            self.register_parameter("bias", None)

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        param.data.copy_(loaded_weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.weight, self.bias)


class ColumnParallelLinear(LinearBase):

    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = False,
    ):
        # 0 stands for column parallelism
        super().__init__(input_size, output_size, 0)

        self.input_size_per_partition = input_size
        self.output_size_per_partition = divide(output_size, self.tp_size)

        # In PyTorch the dimensions of linear layer is transposed, so the
        # input size is the second dimension and the output size is the first
        # dimension
        self.weight = nn.Parameter(torch.empty(self.output_size_per_partition, self.input_size))
        self.weight.weight_loader = self.weight_loader
        if bias:
            self.bias = nn.Parameter(torch.empty(self.output_size_per_partition))
            self.bias.weight_loader = self.weight_loader
        else:
            self.register_parameter("bias", None)

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        param_data = param.data
        # Get the the size of the dimension on `self.tp_dim`
        shard_size = param_data.size(self.tp_dim)
        start_idx = self.tp_rank * shard_size
        # narrow() is a PyTorch function used to extract a sub-tensor (a "narrowed" version)
        # from an input tensor. It returns a new tensor that is a view of the original tensor,
        # meaning they share the same underlying storage.
        #
        # From loaded_weight, slice a sub-tensor with the size of `shard_size`
        # on the dimension `self.tp_dim` and start from `start_idx`.
        loaded_weight = loaded_weight.narrow(self.tp_dim, start_idx, shard_size)
        param_data.copy_(loaded_weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # After forward function, the outputs in different GPUs are not combined
        # together
        return F.linear(x, self.weight, self.bias)


class MergedColumnParallelLinear(ColumnParallelLinear):
    """
    In LLM architectures, there're some linear layers that have multiple output
    sizes. For example, in the Qwen3 model, the gate layer and up layer of the
    SwiGLU feed forward layer have the same input size but different output
    sizes.

    Instead of calcualting them separately, we could merge their weight matrices
    together and calculate them in one forward pass.

    MergedColumnParallelLinear is designed for this purpose, it accepts a list
    of output sizes(`output_sizes`) standing for the output sizes of each
    sub-layer.
    """

    def __init__(
        self,
        input_size: int,
        output_sizes: list[int],
        bias: bool = False,
    ):
        self.output_sizes = output_sizes
        super().__init__(input_size, sum(output_sizes), bias=bias)

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor, loaded_shard_id: int):
        """When loading weights from checkpoint, the weights are usually separated together, the responsibility
        of this function is to read a un-merged weight and correctly loads it into the giant, merged weight matrix.


        Args:
            param (nn.Parameter): The merged weight matrix.
            loaded_weight (torch.Tensor): Un-merged weight matrix read from files.
            loaded_shard_id (int): Stands for the loading sub-tensor.
        """
        # Suppose we have 2 GPU(tp_size=2), and is currently loading the weights of QKV, the output_sizes could
        # be something like [4096, 4096, 4096], not we are loading the weights for k_proj, and loaded_shard_id
        # is 1
        #
        # Get the tensor of the merged, giant matrix
        param_data = param.data
    
        # self.output_sizes[:loaded_shard_id] -> [4096]: Since loaded_shard_id is 1, gets the 0th element
        # sum(...) -> 4096: In the merged, giant matrix, the first 4096 dimension stands for the Q weights
        # // self.tp_size -> 4096 // 2 = 2048: Current GPU only has half of the overall weights, K weights
        # should be placed starting at the 2048th column in param_data
        shard_offset = sum(self.output_sizes[:loaded_shard_id]) // self.tp_size
    
        # self.output_sizes[loaded_shard_id] -> 4096: The full output dimension of K weights
        # // self.tp_size -> 4096 // 2 = 2048: Current GPU needs to load size of 2048
        shard_size = self.output_sizes[loaded_shard_id] // self.tp_size
    
        # narrow returns a view that points to the place in param_data, starting at `shard_offset(2048)`
        # with length `shard_size(2048)`, this region is the place to load weights
        param_data = param_data.narrow(self.tp_dim, shard_offset, shard_size)
    
        # chunks the full k_proj weights into 2 dimensions along the self.tp_dim dimension, and gets
        # its own copy based on self.tp_rank
        loaded_weight = loaded_weight.chunk(self.tp_size, self.tp_dim)[self.tp_rank]
        param_data.copy_(loaded_weight)


class QKVParallelLinear(ColumnParallelLinear):

    def __init__(
        self,
        hidden_size: int,
        head_size: int,
        total_num_heads: int,
        total_num_kv_heads: int | None = None,
        bias: bool = False,
    ):
        self.head_size = head_size
        self.total_num_heads = total_num_heads
        self.total_num_kv_heads = total_num_kv_heads or total_num_heads
        tp_size = dist.get_world_size()
        self.num_heads = divide(self.total_num_heads, tp_size)
        self.num_kv_heads = divide(self.total_num_kv_heads, tp_size)
        input_size = hidden_size
        output_size = (self.total_num_heads + 2 * self.total_num_kv_heads) * self.head_size
        super().__init__(input_size, output_size, bias)

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor, loaded_shard_id: str):
        param_data = param.data
        assert loaded_shard_id in ["q", "k", "v"]
        if loaded_shard_id == "q":
            shard_size = self.num_heads * self.head_size
            shard_offset = 0
        elif loaded_shard_id == "k":
            shard_size = self.num_kv_heads * self.head_size
            shard_offset = self.num_heads * self.head_size
        else:
            shard_size = self.num_kv_heads * self.head_size
            shard_offset = self.num_heads * self.head_size + self.num_kv_heads * self.head_size
        param_data = param_data.narrow(self.tp_dim, shard_offset, shard_size)
        loaded_weight = loaded_weight.chunk(self.tp_size, self.tp_dim)[self.tp_rank]
        param_data.copy_(loaded_weight)


class RowParallelLinear(LinearBase):

    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = False,
    ):
        super().__init__(input_size, output_size, 1)
        # Slice the input dimension, for comparison ColumnParallelLinear slices
        # along the output dimension
        self.input_size_per_partition = divide(input_size, self.tp_size)
        self.output_size_per_partition = output_size

        self.weight = nn.Parameter(torch.empty(self.output_size, self.input_size_per_partition))
        self.weight.weight_loader = self.weight_loader
        if bias:
            self.bias = nn.Parameter(torch.empty(self.output_size))
            self.bias.weight_loader = self.weight_loader
        else:
            self.register_parameter("bias", None)

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        param_data = param.data
        shard_size = param_data.size(self.tp_dim)
        start_idx = self.tp_rank * shard_size
        loaded_weight = loaded_weight.narrow(self.tp_dim, start_idx, shard_size)
        param_data.copy_(loaded_weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = F.linear(x, self.weight, self.bias if self.tp_rank == 0 else None)
        if self.tp_size > 1:
            # By default, the operation of all_reduce is sum, which means after row
            # parallelism, the outputs in different GPUs are summed together.
            dist.all_reduce(y)
        return y
