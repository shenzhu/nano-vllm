import torch
from torch import nn
import torch.nn.functional as F
import torch.distributed as dist

from nanovllm.utils.context import get_context


class VocabParallelEmbedding(nn.Module):
    """Embedding layer that partitions vocabulary across multiple GPUs.
    """

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
    ):
        """
        Args:
            num_embeddings (int): The size of the vocabulary.
            embedding_dim (int): The size of the embedding vector.
        """
        super().__init__()

        # Information about GPUs
        self.tp_rank = dist.get_rank()
        self.tp_size = dist.get_world_size()
        assert num_embeddings % self.tp_size == 0

        # Calculate the vocab size on current GPU, split the embedding matrix into
        # `self.tp_size` parts and only keep the part corresponding to the current rank
        self.num_embeddings = num_embeddings
        self.num_embeddings_per_partition = self.num_embeddings // self.tp_size
        self.vocab_start_idx = self.num_embeddings_per_partition * self.tp_rank
        self.vocab_end_idx = self.vocab_start_idx + self.num_embeddings_per_partition

        self.weight = nn.Parameter(torch.empty(self.num_embeddings_per_partition, embedding_dim))
        self.weight.weight_loader = self.weight_loader

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        param_data = param.data
        shard_size = param_data.size(0)
        start_idx = self.tp_rank * shard_size
        loaded_weight = loaded_weight.narrow(0, start_idx, shard_size)
        assert param_data.size() == loaded_weight.size()
        param_data.copy_(loaded_weight)

    def forward(self, x: torch.Tensor):
        # x shape: (seq_len)
        if self.tp_size > 1:
            # Mask out the tokens that are not in the current GPU's partition
            mask = (x >= self.vocab_start_idx) & (x < self.vocab_end_idx)
            # Adjust the token indices to the local vocabulary
            x = mask * (x - self.vocab_start_idx)

        # shape: (seq_len, embedding_dim)
        y = F.embedding(x, self.weight)

        if self.tp_size > 1:
            y = mask.unsqueeze(1) * y
            # Synchronize the embeddings from all GPUs
            dist.all_reduce(y)
        return y


class ParallelLMHead(VocabParallelEmbedding):

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        bias: bool = False,
    ):
        super().__init__(num_embeddings, embedding_dim)
        if bias:
            self.bias = nn.Parameter(torch.empty(self.num_embeddings_per_partition))
            self.bias.weight_loader = self.weight_loader
        else:
            self.register_parameter("bias", None)

    def forward(self, x: torch.Tensor):
        context = get_context()
        if context.is_prefill:
            last_indices = context.cu_seqlens_q[1:] - 1
            x = x[last_indices].contiguous()
        
        # Convert embedding to vocab size
        logits = F.linear(x, self.weight, self.bias)
        if self.tp_size > 1:
            # Only allocate memory if it's GPU number 0
            all_logits = [torch.empty_like(logits) for _ in range(self.tp_size)] if self.tp_rank == 0 else None
            dist.gather(logits, all_logits, 0)
            logits = torch.cat(all_logits, -1) if self.tp_rank == 0 else None
        return logits
