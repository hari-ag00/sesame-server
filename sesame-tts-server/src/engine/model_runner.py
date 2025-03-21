

import asyncio
import time
import uuid
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any, Callable, Awaitable
import torch
import torchaudio
from huggingface_hub import hf_hub_download
from ..sesame.models import Model, ModelArgs
from moshi.models import loaders
from transformers import AutoTokenizer
from tokenizers.processors import TemplateProcessing
from ..sesame.watermarking import CSM_1B_GH_WATERMARK, load_watermarker, watermark
from .batch_manager import BatchManager, Request
from .cache_manager import CacheManager
from ..utils.config import config


def load_llama3_tokenizer():
    """Load the Llama3 tokenizer with appropriate post-processing."""
    tokenizer_name = "meta-llama/Llama-3.2-1B"
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    bos = tokenizer.bos_token
    eos = tokenizer.eos_token
    tokenizer._tokenizer.post_processor = TemplateProcessing(
        single=f"{bos}:0 $A:0 {eos}:0",
        pair=f"{bos}:0 $A:0 {eos}:0 {bos}:1 $B:1 {eos}:1",
        special_tokens=[(f"{bos}", tokenizer.bos_token_id), (f"{eos}", tokenizer.eos_token_id)],
    )
    return tokenizer


@dataclass
class Segment:
    speaker: int
    text: str
    # (num_samples,), sample_rate = 24_000
    audio: torch.Tensor


class CustomKVCache:
    """Custom KV cache implementation to intercept and manage cache operations."""

    def __init__(
            self,
            cache_manager: CacheManager,
            request_id: str,
            backbone: bool = True,
    ):
        self.cache_manager = cache_manager
        self.request_id = request_id
        self.backbone = backbone

    def append_kv(self, key: torch.Tensor, value: torch.Tensor) -> bool:
        if self.backbone:
            return self.cache_manager.append_backbone_kv(self.request_id, key, value)
        else:
            return self.cache_manager.append_decoder_kv(self.request_id, key, value)

    def get_kv_cache(self, positions: List[int]) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.backbone:
            return self.cache_manager.get_backbone_kv_cache(self.request_id, positions)
        else:
            return self.cache_manager.get_decoder_kv_cache(self.request_id, positions)


class ModelRunner:
    """Runs the Sesame TTS model with batching and caching."""

    def __init__(self):
        self.device = torch.device(config.model.device)
        self.dtype = getattr(torch, config.model.dtype)

        # Set up batch manager
        self.batch_manager = BatchManager()

        # Load the model
        self.model = None
        self.text_tokenizer = None
        self.audio_tokenizer = None
        self.watermarker = None
        self.sample_rate = None

        # Set up cache manager (will initialize after model loading)
        self.cache_manager = None

        # Background task for processing requests
        self.processing_task = None

    # async def initialize(self):
    #     """Initialize the model and other components."""
    #     # Load model
    #     print("Loading model...")
    #     model = Model.from_pretrained(config.model.model_path)
    #     model = model.to(device=self.device, dtype=self.dtype)
    #     model.eval()

    #     # Store model configuration
    #     self.model = model
    #     self.config = model.config
        
        # # Extract model configuration parameters
        # # For csm-1b, we know the backbone is 16 layers with 32 heads and decoder is 4 layers with 8 heads
        # # These values are from FLAVORS in the original models.py
        # backbone_layers = 16   # llama3_2_1B config
        # backbone_heads = 32
        # backbone_head_dim = 64  # 2048 / 32
        
        # decoder_layers = 4     # llama3_2_100M config
        # decoder_heads = 8
        # decoder_head_dim = 128  # 1024 / 8
        
    #     print(f"Using model configuration: backbone={backbone_layers}L/{backbone_heads}H, decoder={decoder_layers}L/{decoder_heads}H")
        
    #     # Initialize cache manager
        # self.cache_manager = CacheManager(
        #     backbone_layers=backbone_layers,
        #     backbone_heads=backbone_heads,
        #     backbone_head_dim=backbone_head_dim,
        #     decoder_layers=decoder_layers,
        #     decoder_heads=decoder_heads,
        #     decoder_head_dim=decoder_head_dim,
        #     dtype=self.dtype,
        # )
    #     # Load tokenizers
    #     print("Loading tokenizers...")
    #     self.text_tokenizer = load_llama3_tokenizer()

    #     device_str = str(self.device)
    #     mimi_weight = hf_hub_download(loaders.DEFAULT_REPO, loaders.MIMI_NAME)
    #     mimi = loaders.get_mimi(mimi_weight, device=device_str)
    #     mimi.set_num_codebooks(32)
    #     self.audio_tokenizer = mimi

    #     # Load watermarker
    #     print("Loading watermarker...")
    #     self.watermarker = load_watermarker(device=device_str)

    #     # Store sample rate
    #     self.sample_rate = mimi.sample_rate

    #     # Start processing task
    #     self.processing_task = asyncio.create_task(self._process_requests())

    #     print("Model initialization complete.")

    async def initialize(self):
        """Initialize the model and other components."""
        # Load the real model
        print(f"Loading model from {config.model.model_path}...")
        model = Model.from_pretrained(config.model.model_path)
        model = model.to(device=self.device, dtype=self.dtype)
        model.eval()
        
        # Store model configuration
        self.model = model
        self.config = model.config
        
        backbone_layers = 16   # llama3_2_1B config
        backbone_heads = 32
        backbone_head_dim = 64  # 2048 / 32
        
        decoder_layers = 4     # llama3_2_100M config
        decoder_heads = 8
        decoder_head_dim = 128  # 1024 / 8        
        
        # Initialize cache manager
        self.cache_manager = CacheManager(
            backbone_layers=backbone_layers,
            backbone_heads=backbone_heads,
            backbone_head_dim=backbone_head_dim,
            decoder_layers=decoder_layers,
            decoder_heads=decoder_heads,
            decoder_head_dim=decoder_head_dim,
            dtype=self.dtype,
        )
        
        # Set up model caches
        print("Setting up model caches...")
        self.model.setup_caches(1)
        
        # Load tokenizers
        print("Loading tokenizers...")
        self.text_tokenizer = load_llama3_tokenizer()
        
        device_str = str(self.device)
        mimi_weight = hf_hub_download(loaders.DEFAULT_REPO, loaders.MIMI_NAME)
        mimi = loaders.get_mimi(mimi_weight, device=device_str)
        mimi.set_num_codebooks(32)
        self.audio_tokenizer = mimi
        
        # Load watermarker
        print("Loading watermarker...")
        self.watermarker = load_watermarker(device=device_str)
        
        # Store sample rate
        self.sample_rate = mimi.sample_rate
        
        # CRITICAL: Cancel any existing processing task
        if self.processing_task is not None:
            self.processing_task.cancel()
            try:
                await self.processing_task
            except asyncio.CancelledError:
                pass
        
        # Start processing task
        print("Starting request processing task...")
        self.processing_task = asyncio.create_task(self._process_requests())
        
        print("Model initialization complete.")

    def _tokenize_text_segment(self, text: str, speaker: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Tokenize a text segment."""
        frame_tokens = []
        frame_masks = []

        text_tokens = self.text_tokenizer.encode(f"[{speaker}]{text}")
        text_frame = torch.zeros(len(text_tokens), 33).long()
        text_frame_mask = torch.zeros(len(text_tokens), 33).bool()
        text_frame[:, -1] = torch.tensor(text_tokens)
        text_frame_mask[:, -1] = True

        frame_tokens.append(text_frame.to(self.device))
        frame_masks.append(text_frame_mask.to(self.device))

        return torch.cat(frame_tokens, dim=0), torch.cat(frame_masks, dim=0)

    def _tokenize_audio(self, audio: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Tokenize an audio segment."""
        frame_tokens = []
        frame_masks = []

        # (K, T)
        audio = audio.to(self.device)
        audio_tokens = self.audio_tokenizer.encode(audio.unsqueeze(0).unsqueeze(0))[0]
        # add EOS frame
        eos_frame = torch.zeros(audio_tokens.size(0), 1).to(self.device)
        audio_tokens = torch.cat([audio_tokens, eos_frame], dim=1)

        audio_frame = torch.zeros(audio_tokens.size(1), 33).long().to(self.device)
        audio_frame_mask = torch.zeros(audio_tokens.size(1), 33).bool().to(self.device)
        audio_frame[:, :-1] = audio_tokens.transpose(0, 1)
        audio_frame_mask[:, :-1] = True

        frame_tokens.append(audio_frame)
        frame_masks.append(audio_frame_mask)

        return torch.cat(frame_tokens, dim=0), torch.cat(frame_masks, dim=0)

    def _tokenize_segment(self, segment: Segment) -> Tuple[torch.Tensor, torch.Tensor]:
        """Tokenize a segment (text + audio)."""
        text_tokens, text_masks = self._tokenize_text_segment(segment.text, segment.speaker)
        audio_tokens, audio_masks = self._tokenize_audio(segment.audio)

        return torch.cat([text_tokens, audio_tokens], dim=0), torch.cat([text_masks, audio_masks], dim=0)

    # @torch.inference_mode()
    # async def _generate_for_request(self, request: Request) -> torch.Tensor:
    #     """Generate audio for a single request."""
    #     # Reset model caches
    #     self.model.reset_caches()

    #     # Prepare request data
    #     text = request.text
    #     speaker = request.speaker
    #     context = request.context
    #     max_audio_length_ms = request.max_audio_length_ms
    #     temperature = request.temperature
    #     topk = request.topk

    # @torch.inference_mode()
    # async def _generate_for_request(self, request: Request) -> torch.Tensor:
    #     """Generate audio for a single request."""
    #     # Ensure the model caches are properly set up
    #     # First reset any existing caches
    #     self.model.reset_caches()
        
    #     # Then explicitly set up the caches with batch size 1
    #     self.model.setup_caches(1)
        
    #     # Prepare request data
    #     text = request.text
    #     speaker = request.speaker
    #     context = request.context
    #     max_audio_length_ms = request.max_audio_length_ms
    #     temperature = request.temperature
    #     topk = request.topk

    #     # Calculate maximum audio frames
    #     max_audio_frames = int(max_audio_length_ms / 80)

    #     # Prepare tokens and masks
    #     tokens, tokens_mask = [], []

    #     # Add context segments if any
    #     for segment in context:
    #         segment_tokens, segment_tokens_mask = self._tokenize_segment(segment)
    #         tokens.append(segment_tokens)
    #         tokens_mask.append(segment_tokens_mask)

    #     # Add the text to generate
    #     gen_segment_tokens, gen_segment_tokens_mask = self._tokenize_text_segment(text, speaker)
    #     tokens.append(gen_segment_tokens)
    #     tokens_mask.append(gen_segment_tokens_mask)

    #     # Combine all tokens
    #     prompt_tokens = torch.cat(tokens, dim=0).long().to(self.device)
    #     prompt_tokens_mask = torch.cat(tokens_mask, dim=0).bool().to(self.device)

    #     # Check sequence length
    #     max_seq_len = 2048 - max_audio_frames
    #     if prompt_tokens.size(0) >= max_seq_len:
    #         raise ValueError(f"Inputs too long, must be below max_seq_len - max_audio_frames: {max_seq_len}")

    #     # Set up initial state
    #     samples = []
    #     curr_tokens = prompt_tokens.unsqueeze(0)
    #     curr_tokens_mask = prompt_tokens_mask.unsqueeze(0)
    #     curr_pos = torch.arange(0, prompt_tokens.size(0)).unsqueeze(0).long().to(self.device)
    
    @torch.inference_mode()
    async def _generate_for_request(self, request: Request) -> torch.Tensor:
        """Generate audio for a single request."""
        # Note: Caches are already reset and set up in _process_batch
        
        # Prepare request data
        text = request.text
        speaker = request.speaker
        context = request.context
        max_audio_length_ms = request.max_audio_length_ms
        temperature = request.temperature
        topk = request.topk
        
        # Calculate maximum audio frames
        max_audio_frames = int(max_audio_length_ms / 80)
        
        # Prepare tokens and masks
        tokens, tokens_mask = [], []
        
        # Add context segments if any
        for segment in context:
            segment_tokens, segment_tokens_mask = self._tokenize_segment(segment)
            tokens.append(segment_tokens)
            tokens_mask.append(segment_tokens_mask)
        
        # Add the text to generate
        gen_segment_tokens, gen_segment_tokens_mask = self._tokenize_text_segment(text, speaker)
        tokens.append(gen_segment_tokens)
        tokens_mask.append(gen_segment_tokens_mask)
        
        # Combine all tokens
        prompt_tokens = torch.cat(tokens, dim=0).long().to(self.device)
        prompt_tokens_mask = torch.cat(tokens_mask, dim=0).bool().to(self.device)
        
        # Check sequence length
        max_seq_len = 2048 - max_audio_frames
        if prompt_tokens.size(0) >= max_seq_len:
            raise ValueError(f"Inputs too long, must be below max_seq_len - max_audio_frames: {max_seq_len}")
        
        # Verify caches are enabled
        if not hasattr(self.model, 'backbone') or not hasattr(self.model.backbone, 'caches_are_enabled'):
            print("Warning: Model doesn't have expected caches_are_enabled method, proceeding anyway...")
        elif not self.model.backbone.caches_are_enabled():
            print("Warning: Model backbone caches are not enabled, attempting to enable...")
            self.model.setup_caches(1)
        
        # Set up initial state
        samples = []
        curr_tokens = prompt_tokens.unsqueeze(0)
        curr_tokens_mask = prompt_tokens_mask.unsqueeze(0)
        curr_pos = torch.arange(0, prompt_tokens.size(0)).unsqueeze(0).long().to(self.device)

        # Generate audio frames
        for _ in range(max_audio_frames):
            # Allow asyncio to yield control
            await asyncio.sleep(0)

            # Generate next frame
            sample = self.model.generate_frame(curr_tokens, curr_tokens_mask, curr_pos, temperature, topk)

            # Check for EOS
            if torch.all(sample == 0):
                break

            # Add to samples
            samples.append(sample)

            # Update current tokens for next iteration
            curr_tokens = torch.cat([sample, torch.zeros(1, 1).long().to(self.device)], dim=1).unsqueeze(1)
            curr_tokens_mask = torch.cat(
                [torch.ones_like(sample).bool(), torch.zeros(1, 1).bool().to(self.device)], dim=1
            ).unsqueeze(1)
            curr_pos = curr_pos[:, -1:] + 1

            # Send intermediate chunk if we have enough frames (every ~1 second)
            if len(samples) % 12 == 0:  # ~1 second of audio
                # Decode intermediate audio
                intermediate_samples = torch.stack(samples).permute(1, 2, 0)
                intermediate_audio = self.audio_tokenizer.decode(intermediate_samples).squeeze(0).squeeze(0)

                # Notify audio chunk
                await self.batch_manager.notify_audio_chunk(request.request_id, intermediate_audio)

        # Decode final audio
        if not samples:
            # No samples generated, return empty audio
            audio = torch.zeros(1, dtype=torch.float32, device=self.device)
        else:
            # Decode audio from samples
            audio = self.audio_tokenizer.decode(torch.stack(samples).permute(1, 2, 0)).squeeze(0).squeeze(0)

        # Apply watermarking
        audio, wm_sample_rate = watermark(self.watermarker, audio, self.sample_rate, CSM_1B_GH_WATERMARK)
        audio = torchaudio.functional.resample(audio, orig_freq=wm_sample_rate, new_freq=self.sample_rate)

        return audio

    # async def _process_batch(self, batch: List[Request]):
    #     """Process a batch of requests."""
    #     # Process each request one by one for now
    #     # In a more advanced implementation, we would process them in parallel
    #     for request in batch:
    #         try:
    #             # Generate audio for this request
    #             audio = await self._generate_for_request(request)

    #             # Mark request as finished
    #             await self.batch_manager.mark_request_finished(request.request_id, audio)
    #         except Exception as e:
    #             # Mark request as failed
    #             await self.batch_manager.mark_request_error(request.request_id, str(e))

    async def _process_batch(self, batch: List[Request]):
        """Process a batch of requests."""
        # Process each request one by one for now
        # In a more advanced implementation, we would process them in parallel
        for request in batch:
            try:
                # Reset model before each request
                self.model.reset_caches()
                
                # Ensure caches are set up with appropriate batch size
                try:
                    print(f"Setting up model caches for request {request.request_id}...")
                    self.model.setup_caches(1)
                except Exception as e:
                    raise RuntimeError(f"Failed to set up model caches: {e}")
                
                # Generate audio for this request
                audio = await self._generate_for_request(request)
                
                # Mark request as finished
                await self.batch_manager.mark_request_finished(request.request_id, audio)
            except Exception as e:
                print(f"Error processing request {request.request_id}: {e}")
                # Mark request as failed
                await self.batch_manager.mark_request_error(request.request_id, str(e))

    # async def _process_requests(self):
    #     """Background task that processes requests."""
    #     while not self.batch_manager.shutdown_flag:
    #         try:
    #             # Get the next batch of requests
    #             batch = await self.batch_manager.get_next_batch(config.batch.max_batch_size)

    #             if batch:
    #                 # Process the batch
    #                 await self._process_batch(batch)
    #             else:
    #                 # Wait for new requests
    #                 await self.batch_manager.wait_for_requests(timeout=1.0)

    #             # Clean up old requests periodically
    #             self.batch_manager.cleanup_old_requests()
    #         except Exception as e:
    #             print(f"Error processing batch: {e}")
    #             await asyncio.sleep(1.0)
    
    async def _process_requests(self):
        """Background task that processes requests."""
        print("Starting request processing loop")
        while not self.batch_manager.shutdown_flag:
            try:
                # Get the next batch of requests
                batch = await self.batch_manager.get_next_batch(config.batch.max_batch_size)
                
                if batch:
                    print(f"Processing batch with {len(batch)} requests")
                    # Process the batch
                    await self._process_batch(batch)
                else:
                    # No requests to process, wait briefly and check again
                    await asyncio.sleep(0.5)
                
            except Exception as e:
                print(f"Error in processing loop: {e}")
                await asyncio.sleep(1.0)

    async def add_request(
            self,
            text: str,
            speaker: int,
            context: List[Segment] = None,
            max_audio_length_ms: float = None,
            temperature: float = None,
            topk: int = None,
            priority: int = 0,
            on_audio_chunk: Optional[Callable[[str, torch.Tensor], Awaitable[None]]] = None,
            on_finished: Optional[Callable[[str, Optional[torch.Tensor], Optional[str]], Awaitable[None]]] = None,
    ) -> str:
        """
        Add a request to generate audio.

        Args:
            text: Text to synthesize
            speaker: Speaker ID
            context: List of previous segments for context
            max_audio_length_ms: Maximum audio length in milliseconds
            temperature: Sampling temperature
            topk: Top-k sampling parameter
            priority: Request priority (lower = higher priority)
            on_audio_chunk: Callback for audio chunks
            on_finished: Callback for request completion

        Returns:
            request_id: Unique identifier for the request
        """
        if context is None:
            context = []

        if max_audio_length_ms is None:
            max_audio_length_ms = config.batch.max_audio_length_ms

        if temperature is None:
            temperature = config.generation.temperature

        if topk is None:
            topk = config.generation.topk

        return await self.batch_manager.add_request(
            text=text,
            speaker=speaker,
            context=context,
            max_audio_length_ms=max_audio_length_ms,
            temperature=temperature,
            topk=topk,
            priority=priority,
            on_audio_chunk=on_audio_chunk,
            on_finished=on_finished,
        )

    async def get_request_status(self, request_id: str) -> Optional[Dict[str, Any]]:
        """Get the status of a request."""
        return await self.batch_manager.get_request_status(request_id)

    async def cancel_request(self, request_id: str) -> bool:
        """Cancel a request if it's still waiting."""
        return await self.batch_manager.cancel_request(request_id)

    async def shutdown(self):
        """Shutdown the model runner."""
        # Shutdown batch manager
        await self.batch_manager.shutdown()

        # Cancel processing task
        if self.processing_task:
            self.processing_task.cancel()
            try:
                await self.processing_task
            except asyncio.CancelledError:
                pass

        # Clear caches
        if self.cache_manager:
            self.cache_manager.reset()

        # Free model memory
        if self.model:
            self.model = None

        # Force garbage collection
        import gc
        gc.collect()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
