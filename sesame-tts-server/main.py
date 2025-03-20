import argparse
import os
import torch
from src.server.app import start_server
from src.utils.config import config


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Sesame TTS Server")

    # Server settings
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Server host")
    parser.add_argument("--port", type=int, default=8000, help="Server port")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    parser.add_argument("--workers", type=int, default=1, help="Number of worker processes")

    # Model settings
    parser.add_argument("--model", type=str, default="sesame/csm-1b", help="Model path or HF hub ID")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device to run on")
    parser.add_argument("--dtype", type=str, default="bfloat16", choices=["float32", "float16", "bfloat16"],
                        help="Data type")

    # Cache settings
    parser.add_argument("--block-size", type=int, default=16, help="KV cache block size")
    parser.add_argument("--max-blocks", type=int, default=512, help="Maximum number of KV cache blocks")
    parser.add_argument("--gpu-mem-util", type=float, default=0.9, help="GPU memory utilization ratio")

    # Batch settings
    parser.add_argument("--max-batch-size", type=int, default=32, help="Maximum batch size")
    parser.add_argument("--max-waiting-tokens", type=int, default=8, help="Maximum tokens to wait before processing")
    parser.add_argument("--max-model-len", type=int, default=2048, help="Maximum sequence length")
    parser.add_argument("--max-audio-length", type=float, default=90000, help="Maximum audio length in milliseconds")

    # Generation settings
    parser.add_argument("--temperature", type=float, default=0.9, help="Sampling temperature")
    parser.add_argument("--topk", type=int, default=50, help="Top-k sampling parameter")

    return parser.parse_args()


def update_config_from_args(args):
    """Update config from command line arguments."""
    # Server settings
    config.server.host = args.host
    config.server.port = args.port
    config.server.debug = args.debug
    config.server.workers = args.workers

    # Model settings
    config.model.model_path = args.model
    config.model.device = args.device
    config.model.dtype = args.dtype

    # Cache settings
    config.cache.block_size = args.block_size
    config.cache.max_blocks = args.max_blocks
    config.cache.gpu_memory_utilization = args.gpu_mem_util

    # Batch settings
    config.batch.max_batch_size = args.max_batch_size
    config.batch.max_waiting_tokens = args.max_waiting_tokens
    config.batch.max_model_len = args.max_model_len
    config.batch.max_audio_length_ms = args.max_audio_length

    # Generation settings
    config.generation.temperature = args.temperature
    config.generation.topk = args.topk


def main():
    """Main entry point."""
    # Parse command line arguments
    args = parse_args()

    # Update config from arguments
    update_config_from_args(args)

    # Start server
    start_server()


if __name__ == "__main__":
    main()