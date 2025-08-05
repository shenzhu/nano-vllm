from dataclasses import dataclass
import torch


@dataclass
class Context:
    """Set up some global variables.
    """
    is_prefill: bool = False # prefill or decode
    cu_seqlens_q: torch.Tensor | None = None # prefill
    cu_seqlens_k: torch.Tensor | None = None # prefill
    max_seqlen_q: int = 0 # prefill
    max_seqlen_k: int = 0 # prefill
    slot_mapping: torch.Tensor | None = None # prefill or decode
    context_lens: torch.Tensor | None = None # decode
    block_tables: torch.Tensor | None = None # prefill or decode

_CONTEXT = Context()

def get_context():
    return _CONTEXT

def set_context(is_prefill, cu_seqlens_q=None, cu_seqlens_k=None, max_seqlen_q=0, max_seqlen_k=0, slot_mapping=None, context_lens=None, block_tables=None):
    global _CONTEXT
    _CONTEXT = Context(is_prefill, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, slot_mapping, context_lens, block_tables)

def reset_context():
    global _CONTEXT
    _CONTEXT = Context()
