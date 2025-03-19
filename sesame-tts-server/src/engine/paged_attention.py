import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple
import numpy as np


class Block:
    """A block of key-value pairs in memory."""

    def __init__(self, block_size: int, num_layers: int, num_heads: int, head_dim: int, dtype: torch.dtype):
        self.block_size = block_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.dtype = dtype

        # Initialize key and value tensors
        # Shape: [num_layers, num_heads, block_size, head_dim]
        self.key = torch.zeros(
            (num_layers, num_heads, block_size, head_dim),
            dtype=dtype, device="cuda"
        )
        self.value = torch.zeros(
            (num_layers, num_heads, block_size, head_dim),
            dtype=dtype, device="cuda"
        )

        # Track the number of slots used in this block
        self.slots_used = 0

    def reset(self):
        """Reset the block to be reused."""
        self.slots_used = 0


class PagedAttention:
    """Implementation of paged attention for KV caching."""

    def __init__(
            self,
            num_layers: int,
            num_heads: int,
            head_dim: int,
            block_size: int,
            max_blocks: int,
            dtype: torch.dtype = torch.bfloat16,
    ):
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.block_size = block_size
        self.max_blocks = max_blocks
        self.dtype = dtype

        # Initialize free blocks list
        self.free_blocks = [
            Block(block_size, num_layers, num_heads, head_dim, dtype)
            for _ in range(max_blocks)
        ]

        # Map sequence_id -> list of blocks
        self.sequence_blocks: Dict[int, List[Block]] = {}

        # Map sequence_id -> position mapping
        self.sequence_positions: Dict[int, List[Tuple[int, int]]] = {}

        # Current sequence counter
        self.next_sequence_id = 0

    def allocate_sequence(self) -> int:
        """Allocate a new sequence."""
        sequence_id = self.next_sequence_id
        self.next_sequence_id += 1
        self.sequence_blocks[sequence_id] = []
        self.sequence_positions[sequence_id] = []
        return sequence_id

    def append_kv(
            self,
            sequence_id: int,
            key: torch.Tensor,  # [num_layers, num_heads, seq_len, head_dim]
            value: torch.Tensor,  # [num_layers, num_heads, seq_len, head_dim]
    ) -> bool:
        """
        Append key-value pairs to the specified sequence.
        Returns True if successful, False if out of memory.
        """
        seq_len = key.size(2)
        blocks_needed = (seq_len + self.block_size - 1) // self.block_size

        # Check if we have enough free blocks
        if len(self.free_blocks) < blocks_needed:
            return False

        # Allocate blocks for this sequence
        for i in range(blocks_needed):
            block = self.free_blocks.pop()
            self.sequence_blocks[sequence_id].append(block)

            # How many slots to copy for this block
            slots_to_copy = min(self.block_size, seq_len - i * self.block_size)

            # Determine start position in the key and value tensors
            start_pos = i * self.block_size
            end_pos = start_pos + slots_to_copy

            # Copy data to the block
            block.key[:, :, :slots_to_copy, :] = key[:, :, start_pos:end_pos, :]
            block.value[:, :, :slots_to_copy, :] = value[:, :, start_pos:end_pos, :]
            block.slots_used = slots_to_copy

            # Update position mapping
            for j in range(slots_to_copy):
                pos = start_pos + j
                self.sequence_positions[sequence_id].append((block, j))

        return True

    def get_kv_cache(self, sequence_id: int, positions: List[int]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get key-value cache for the specified positions in the sequence.

        Args:
            sequence_id: ID of the sequence
            positions: List of positions to retrieve

        Returns:
            key: Tensor of shape [num_layers, num_heads, len(positions), head_dim]
            value: Tensor of shape [num_layers, num_heads, len(positions), head_dim]
        """
        seq_positions = self.sequence_positions[sequence_id]

        # Ensure positions are in range
        max_pos = len(seq_positions) - 1
        positions = [min(p, max_pos) for p in positions]

        # Initialize output tensors
        key = torch.zeros(
            (self.num_layers, self.num_heads, len(positions), self.head_dim),
            dtype=self.dtype, device="cuda"
        )
        value = torch.zeros(
            (self.num_layers, self.num_heads, len(positions), self.head_dim),
            dtype=self.dtype, device="cuda"
        )

        # Fill the key and value tensors
        for i, pos in enumerate(positions):
            block, block_pos = seq_positions[pos]
            key[:, :, i, :] = block.key[:, :, block_pos, :]
            value[:, :, i, :] = block.value[:, :, block_pos, :]

        return key, value

    def free_sequence(self, sequence_id: int):
        """Free all blocks associated with a sequence."""
        if sequence_id not in self.sequence_blocks:
            return

        for block in self.sequence_blocks[sequence_id]:
            block.reset()
            self.free_blocks.append(block)

        # Remove sequence from tracking
        del self.sequence_blocks[sequence_id]
        del self.sequence_positions[sequence_id]