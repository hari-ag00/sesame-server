from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class ServerConfig:
    host: str = "0.0.0.0"
    port: int = 8000
    debug: bool = False
    workers: int = 1


@dataclass
class ModelConfig:
    model_path: str = "sesame/csm-1b"
    device: str = "cuda"
    dtype: str = "bfloat16"  # "float32", "float16", "bfloat16"


@dataclass
class CacheConfig:
    # Page size in number of tokens/frames
    block_size: int = 16
    # Maximum number of blocks to allocate
    max_blocks: int = 512
    # GPU reserved memory ratio (0-1)
    gpu_memory_utilization: float = 0.9


@dataclass
class BatchConfig:
    # Maximum batch size
    max_batch_size: int = 32
    # Maximum waiting time for batch formation (ms)
    max_waiting_tokens: int = 8
    # Maximum sequence length
    max_model_len: int = 2048
    # Maximum audio length in milliseconds
    max_audio_length_ms: float = 90_000


@dataclass
class GenerationConfig:
    temperature: float = 0.9
    topk: int = 50


@dataclass
class Config:
    server: ServerConfig = field(default_factory=ServerConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    cache: CacheConfig = field(default_factory=CacheConfig)
    batch: BatchConfig = field(default_factory=BatchConfig)
    generation: GenerationConfig = field(default_factory=GenerationConfig)


# Create a default config
config = Config()