import os
from dataclasses import dataclass
from transformers import AutoConfig


@dataclass
class Config:
    model: str
    max_num_batched_tokens: int = 16384 # Maximum number of total tokens that can be processed in a single batch
    max_num_seqs: int = 512 # Maximum number of sequences that can be processed concurrently
    max_model_len: int = 4096 # Maximum length the model can handle
    gpu_memory_utilization: float = 0.9 # Fraction of GPU memory to allocate
    tensor_parallel_size: int = 1 # Number of GPUs to use for tensor parallelism
    enforce_eager: bool = False # Whether to disable CUDA graphs and use eager mode, true for debugging
    hf_config: AutoConfig | None = None # HuggingFace model config
    eos: int = -1 # End-of-sequence token ID (-1 means not set)
    kvcache_block_size: int = 256 # Size of blocks for KV cache allocation (must be multiple of 256)
    num_kvcache_blocks: int = -1 # Total number of KV cache blocks (-1 means auto-calculate)

    def __post_init__(self):
        assert os.path.isdir(self.model)
        assert self.kvcache_block_size % 256 == 0
        assert 1 <= self.tensor_parallel_size <= 8
        self.hf_config = AutoConfig.from_pretrained(self.model)
        self.max_model_len = min(self.max_model_len, self.hf_config.max_position_embeddings)

        # Batch capacity must be at least as large as the model length
        assert self.max_num_batched_tokens >= self.max_model_len


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="read config")
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="model path"
    )
    args = parser.parse_args()
    config = Config(model=args.model)
    print(config)

"""
~/nano-vllm $ python3 nanovllm/config.py --model ~/huggingface/Qwen3-0.6B/
Config(model='/home/bento/huggingface/Qwen3-0.6B/', max_num_batched_tokens=16384, max_num_seqs=512, max_model_len=4096, gpu_memory_utilization=0.9, tensor_parallel_size=1, enforce_eager=False, hf_config=Qwen3Config {
  "architectures": [
    "Qwen3ForCausalLM"
  ],
  "attention_bias": false,
  "attention_dropout": 0.0,
  "bos_token_id": 151643,
  "eos_token_id": 151645,
  "head_dim": 128,
  "hidden_act": "silu",
  "hidden_size": 1024,
  "initializer_range": 0.02,
  "intermediate_size": 3072,
  "layer_types": [
    "full_attention",
    "full_attention",
    "full_attention",
    "full_attention",
    "full_attention",
    "full_attention",
    "full_attention",
    "full_attention",
    "full_attention",
    "full_attention",
    "full_attention",
    "full_attention",
    "full_attention",
    "full_attention",
    "full_attention",
    "full_attention",
    "full_attention",
    "full_attention",
    "full_attention",
    "full_attention",
    "full_attention",
    "full_attention",
    "full_attention",
    "full_attention",
    "full_attention",
    "full_attention",
    "full_attention",
    "full_attention"
  ],
  "max_position_embeddings": 40960,
  "max_window_layers": 28,
  "model_type": "qwen3",
  "num_attention_heads": 16,
  "num_hidden_layers": 28,
  "num_key_value_heads": 8,
  "rms_norm_eps": 1e-06,
  "rope_scaling": null,
  "rope_theta": 1000000,
  "sliding_window": null,
  "tie_word_embeddings": true,
  "torch_dtype": "bfloat16",
  "transformers_version": "4.54.1",
  "use_cache": true,
  "use_sliding_window": false,
  "vocab_size": 151936
}
, eos=-1, kvcache_block_size=256, num_kvcache_blocks=-1)
"""