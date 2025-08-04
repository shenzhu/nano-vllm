from dataclasses import dataclass


@dataclass
class SamplingParams:
    temperature: float = 1.0 # Temperature for sampling, use larger values for more randomness
    max_tokens: int = 64 # Maximum number of tokens to generate
    ignore_eos: bool = False # Whether to ignore the end-of-sequence token
