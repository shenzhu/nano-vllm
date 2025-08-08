import torch
from torch import nn
import triton
import triton.language as tl

from flash_attn import flash_attn_varlen_func, flash_attn_with_kvcache
from nanovllm.utils.context import get_context


@triton.jit
def store_kvcache_kernel(
    key_ptr,
    key_stride,
    value_ptr,
    value_stride,
    k_cache_ptr,
    v_cache_ptr,
    slot_mapping_ptr,
    D: tl.constexpr,
):
    idx = tl.program_id(0)
    key_offsets = idx * key_stride + tl.arange(0, D)
    value_offsets = idx * value_stride + tl.arange(0, D)
    key = tl.load(key_ptr + key_offsets)
    value = tl.load(value_ptr + value_offsets)
    slot = tl.load(slot_mapping_ptr + idx)
    cache_offsets = slot * D + tl.arange(0, D)
    tl.store(k_cache_ptr + cache_offsets, key)
    tl.store(v_cache_ptr + cache_offsets, value)

def store_kvcache_simpified(
    key: torch.Tensor,
    value: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
):
    N, num_heads, head_dim = key.shape
    
    flat_key = key.view(N, -1)
    flat_value = value.view(N, -1)

    for i in range(N):
        slot = slot_mapping[i].item()
        k_cache[slot] = flat_key[i]
        v_cache[slot] = flat_value[i]


def store_kvcache(key: torch.Tensor, value: torch.Tensor, k_cache: torch.Tensor, v_cache: torch.Tensor, slot_mapping: torch.Tensor):
    # N: The total number of tokens to process in this batch
    #
    # In prefill stage, suppose the batch contains 2 prompts from users, the first one has 100 tokens,
    # while the second one has 50. Then in this batch, N = 150.
    #
    # in Decode stage, model only generate 1 token for each sequence in the batch,
    # so N = batch_size
    N, num_heads, head_dim = key.shape

    # D: The total number of dimensions after concatenating all the heads
    D = num_heads * head_dim
    assert key.stride(-1) == 1 and value.stride(-1) == 1
    assert key.stride(1) == head_dim and value.stride(1) == head_dim
    assert k_cache.stride(1) == D and v_cache.stride(1) == D

    # Make sure a slot is prepared for each token in the batch
    assert slot_mapping.numel() == N

    # Launch grid, starting N instances of store_kvcache_kernel and have them execute in parallel on GPU
    #
    # When N instances are launched, each instance can get its own id with `tl.program_id(0)`, ranging from
    # 0 to N - 1. In this way:
    # - 0th kernal instance with idx 0, will process 0th token
    # - 1st kernal instance with idx 1, will process 1st token
    # - ...
    # - (N - 1)th kernal instance with idx N - 1, will process (N - 1)th token
    #
    # stride defines how many elements to skip to access the next element in memory
    # Suppose key has dimension (8, 32, 128), this tensor is stored in memory as a contiguous 1-dimensional
    # array with 8 * 32 * 128 = 32768 elements
    # The stride of this tensor would be (4096, 128, 1)
    # - key.stride(2) = 1: In the last dimension, to move from key[i, j, k] to key[i, j, k + 1], we need to skip 1 element
    # - key.stride(1) = 128: In the second dimension, to move from key[i, j, k] to key[i, j + 1, k], we need to skip 128 elements
    # - key.stride(0) = 4096: In the first dimension, to move from key[i, j, k] to key[i + 1, j, k], we need to skip 4096 elements
    #
    # In this way, we can access the next element in memory by adding the stride to the current element
    #
    # In this kernel, we need to access the next token in the key and value tensors, so we use key.stride(0) and value.stride(0)
    store_kvcache_kernel[(N,)](key, key.stride(0), value, value.stride(0), k_cache, v_cache, slot_mapping, D)


class Attention(nn.Module):

    def __init__(
        self,
        num_heads,
        head_dim,
        scale,
        num_kv_heads,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = scale
        self.num_kv_heads = num_kv_heads
        self.k_cache = self.v_cache = torch.tensor([])

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        o: torch.Tensor
        # shape: (num_tokens, num_heads, head_dim)
        q = q.view(-1, self.num_heads, self.head_dim)
        k = k.view(-1, self.num_kv_heads, self.head_dim)
        v = v.view(-1, self.num_kv_heads, self.head_dim)

        context = get_context()
        k_cache, v_cache = self.k_cache, self.v_cache
        if k_cache.numel() and v_cache.numel():
            store_kvcache(k, v, k_cache, v_cache, context.slot_mapping)
        if context.is_prefill:
            if context.block_tables is not None:    # prefix cache
                k, v = k_cache, v_cache
            o = flash_attn_varlen_func(q, k, v,
                                       max_seqlen_q=context.max_seqlen_q, cu_seqlens_q=context.cu_seqlens_q,
                                       max_seqlen_k=context.max_seqlen_k, cu_seqlens_k=context.cu_seqlens_k,
                                       softmax_scale=self.scale, causal=True, block_table=context.block_tables)
        else:    # decode
            o = flash_attn_with_kvcache(q.unsqueeze(1), k_cache, v_cache,
                                        cache_seqlens=context.context_lens, block_table=context.block_tables, 
                                        softmax_scale=self.scale, causal=True)
        o = o.view(-1, self.num_heads * self.head_dim)
        return o
