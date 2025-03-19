import torch
from typing import Dict, List, Optional, Tuple
from .paged_attention import PagedAttention
from ..utils.config import config


class CacheManager:
    """Manages KV caches for both backbone and decoder models."""

    def __init__(
            self,
            backbone_layers: int,
            backbone_heads: int,
            backbone_head_dim: int,
            decoder_layers: int,
            decoder_heads: int,
            decoder_head_dim: int,
            dtype: torch.dtype = torch.bfloat16,
    ):
        self.dtype = dtype

        # Create paged attention for backbone and decoder
        self.backbone_cache = PagedAttention(
            num_layers=backbone_layers,
            num_heads=backbone_heads,
            head_dim=backbone_head_dim,
            block_size=config.cache.block_size,
            max_blocks=config.cache.max_blocks // 2,  # Split blocks between backbone and decoder
            dtype=dtype,
        )

        self.decoder_cache = PagedAttention(
            num_layers=decoder_layers,
            num_heads=decoder_heads,
            head_dim=decoder_head_dim,
            block_size=config.cache.block_size,
            max_blocks=config.cache.max_blocks // 2,
            dtype=dtype,
        )

        # Map external request_id to internal sequence_ids
        self.request_map: Dict[str, Tuple[int, int]] = {}  # request_id -> (backbone_seq_id, decoder_seq_id)

    def allocate_request(self, request_id: str) -> bool:
        """
        Allocate caches for a new request.
        Returns True if successful, False if out of memory.
        """
        if request_id in self.request_map:
            return True

        backbone_seq_id = self.backbone_cache.allocate_sequence()
        decoder_seq_id = self.decoder_cache.allocate_sequence()

        self.request_map[request_id] = (backbone_seq_id, decoder_seq_id)
        return True

    def append_backbone_kv(
            self,
            request_id: str,
            key: torch.Tensor,  # [num_layers, num_heads, seq_len, head_dim]
            value: torch.Tensor,  # [num_layers, num_heads, seq_len, head_dim]
    ) -> bool:
        """
        Append key-value pairs to the backbone cache for the specified request.
        Returns True if successful, False if out of memory.
        """
        if request_id not in self.request_map:
            if not self.allocate_request(request_id):
                return False

        backbone_seq_id, _ = self.request_map[request_id]
        return self.backbone_cache.append_kv(backbone_seq_id, key, value)

    def append_decoder_kv(
            self,
            request_id: str,
            key: torch.Tensor,  # [num_layers, num_heads, seq_len, head_dim]
            value: torch.Tensor,  # [num_layers, num_heads, seq_len, head_dim]
    ) -> bool:
        """
        Append key-value pairs to the decoder cache for the specified request.
        Returns True if successful, False if out of memory.
        """
        if request_id not in self.request_map:
            if not self.allocate_request(request_id):
                return False

        _, decoder_seq_id = self.request_map[request_id]
        return self.decoder_cache.append_kv(decoder_seq_id, key, value)

    def get_backbone_kv_cache(self, request_id: str, positions: List[int]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get backbone key-value cache for the specified positions in the request.

        Args:
            request_id: ID of the request
            positions: List of positions to retrieve

        Returns:
            key: Tensor of shape [num_layers, num_heads, len(positions), head_dim]
            value: Tensor of shape [num_layers, num_heads, len(positions), head_dim]
        """
        backbone_seq_id, _ = self.request_map[request_id]
        return self.backbone_cache.get_kv_cache(backbone_seq_id, positions)

    def get_decoder_kv_cache(self, request_id: str, positions: List[int]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get decoder key-value cache for the specified positions in the request.

        Args:
            request_id: ID of the request
            positions: List of positions to retrieve

        Returns:
            key: Tensor of shape [num_layers, num_heads, len(positions), head_dim]
            value: Tensor of shape [num_layers, num_heads, len(positions), head_dim]
        """
        _, decoder_seq_id = self.request_map[request_id]
        return self.decoder_cache.get_kv_cache(decoder_seq_id, positions)

    def free_request(self, request_id: str):
        """Free all caches associated with a request."""
        if request_id not in self.request_map:
            return

        backbone_seq_id, decoder_seq_id = self.request_map[request_id]
        self.backbone_cache.free_sequence(backbone_seq_id)
        self.decoder_cache.free_sequence(decoder_seq_id)

        del self.request_map[request_id]

    def reset(self):
        """Reset all caches."""
        for request_id in list(self.request_map.keys()):
            self.free_request(request_id)