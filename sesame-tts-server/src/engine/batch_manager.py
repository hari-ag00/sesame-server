import asyncio
import time
import uuid
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Tuple, Any, Callable, Awaitable
import torch
import numpy as np
from ..utils.config import config


class RequestStatus(Enum):
    WAITING = 0  # Request is waiting in the queue
    RUNNING = 1  # Request is currently running
    FINISHED = 2  # Request has finished successfully
    ERROR = 3  # Request has encountered an error
    CANCELLED = 4  # Request was cancelled by the user


@dataclass
class Request:
    # Request identifier
    request_id: str

    # Request parameters
    text: str
    speaker: int
    context: List[Any]  # List of Segment objects
    max_audio_length_ms: float
    temperature: float
    topk: int

    # Priority (lower value = higher priority)
    priority: int = 0

    # Request status
    status: RequestStatus = RequestStatus.WAITING

    # Tracking information
    create_time: float = 0.0
    start_time: float = 0.0
    end_time: float = 0.0

    # Result storage
    result: Optional[torch.Tensor] = None
    error: Optional[str] = None

    # Callbacks
    on_audio_chunk: Optional[Callable[[str, torch.Tensor], Awaitable[None]]] = None
    on_finished: Optional[Callable[[str, Optional[torch.Tensor], Optional[str]], Awaitable[None]]] = None

    def __post_init__(self):
        self.create_time = time.time()

    def mark_running(self):
        self.status = RequestStatus.RUNNING
        self.start_time = time.time()

    def mark_finished(self, result: torch.Tensor):
        self.status = RequestStatus.FINISHED
        self.end_time = time.time()
        self.result = result

    def mark_error(self, error: str):
        self.status = RequestStatus.ERROR
        self.end_time = time.time()
        self.error = error

    def mark_cancelled(self):
        self.status = RequestStatus.CANCELLED
        self.end_time = time.time()

    async def notify_audio_chunk(self, chunk: torch.Tensor):
        if self.on_audio_chunk:
            await self.on_audio_chunk(self.request_id, chunk)

    async def notify_finished(self):
        if self.on_finished:
            await self.on_finished(self.request_id, self.result, self.error)

    @property
    def waiting_time(self) -> float:
        """Get the time this request has been waiting (in seconds)."""
        if self.status == RequestStatus.WAITING:
            return time.time() - self.create_time
        elif self.status == RequestStatus.RUNNING:
            return self.start_time - self.create_time
        else:
            return self.end_time - self.create_time

    @property
    def running_time(self) -> float:
        """Get the time this request has been running (in seconds)."""
        if self.status == RequestStatus.WAITING:
            return 0.0
        elif self.status == RequestStatus.RUNNING:
            return time.time() - self.start_time
        else:
            return self.end_time - self.start_time

    @property
    def total_time(self) -> float:
        """Get the total time from creation to completion (in seconds)."""
        if self.status in [RequestStatus.FINISHED, RequestStatus.ERROR, RequestStatus.CANCELLED]:
            return self.end_time - self.create_time
        else:
            return time.time() - self.create_time


class BatchManager:
    """Manages request batching and scheduling."""

    def __init__(self):
        # Request queues (by priority)
        self.waiting_requests: Dict[int, List[Request]] = {}

        # Active requests
        self.active_requests: Dict[str, Request] = {}

        # Completed requests (keep for a while for status queries)
        self.completed_requests: Dict[str, Request] = {}

        # Lock for thread safety
        self.lock = asyncio.Lock()

        # Event for notifying when new requests arrive
        self.new_request_event = asyncio.Event()

        # Flag to signal shutdown
        self.shutdown_flag = False

    async def add_request(
            self,
            text: str,
            speaker: int,
            context: List[Any],
            max_audio_length_ms: float = config.batch.max_audio_length_ms,
            temperature: float = config.generation.temperature,
            topk: int = config.generation.topk,
            priority: int = 0,
            on_audio_chunk: Optional[Callable[[str, torch.Tensor], Awaitable[None]]] = None,
            on_finished: Optional[Callable[[str, Optional[torch.Tensor], Optional[str]], Awaitable[None]]] = None,
    ) -> str:
        """
        Add a new request to the queue.

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
        request_id = str(uuid.uuid4())

        request = Request(
            request_id=request_id,
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

        async with self.lock:
            if priority not in self.waiting_requests:
                self.waiting_requests[priority] = []

            self.waiting_requests[priority].append(request)

            # Set the event to notify scheduler
            self.new_request_event.set()

        return request_id

    async def get_request_status(self, request_id: str) -> Optional[Dict[str, Any]]:
        """Get the status of a request."""
        async with self.lock:
            # Check active requests
            if request_id in self.active_requests:
                request = self.active_requests[request_id]

            # Check waiting requests
            else:
                request = None
                for priority, requests in self.waiting_requests.items():
                    for req in requests:
                        if req.request_id == request_id:
                            request = req
                            break
                    if request:
                        break

            # Check completed requests
            if not request and request_id in self.completed_requests:
                request = self.completed_requests[request_id]

            if not request:
                return None

            return {
                "request_id": request.request_id,
                "status": request.status.name,
                "create_time": request.create_time,
                "start_time": request.start_time,
                "end_time": request.end_time,
                "waiting_time": request.waiting_time,
                "running_time": request.running_time,
                "total_time": request.total_time,
                "error": request.error,
            }

    async def cancel_request(self, request_id: str) -> bool:
        """Cancel a request if it's still waiting."""
        async with self.lock:
            # Check waiting requests
            for priority, requests in self.waiting_requests.items():
                for i, request in enumerate(requests):
                    if request.request_id == request_id:
                        # Remove from waiting queue
                        request.mark_cancelled()
                        del self.waiting_requests[priority][i]

                        # Move to completed
                        self.completed_requests[request_id] = request

                        # Notify completion
                        asyncio.create_task(request.notify_finished())

                        return True

            # Cannot cancel if already running or completed
            return False

    async def get_next_batch(self, max_batch_size: int) -> List[Request]:
        """
        Get the next batch of requests to process.

        Args:
            max_batch_size: Maximum number of requests to include in the batch

        Returns:
            List of Request objects to process
        """
        batch = []

        async with self.lock:
            # Check if we have any waiting requests
            if not any(self.waiting_requests.values()):
                # Reset the event since there are no requests
                self.new_request_event.clear()
                return []

            # Get requests in priority order
            priorities = sorted(self.waiting_requests.keys())

            for priority in priorities:
                # Get requests for this priority
                priority_requests = self.waiting_requests[priority]

                # Take as many as we can fit in the batch
                available_slots = max_batch_size - len(batch)
                to_take = min(available_slots, len(priority_requests))

                if to_take > 0:
                    # Take the oldest requests first
                    for _ in range(to_take):
                        request = priority_requests.pop(0)
                        request.mark_running()
                        batch.append(request)

                        # Add to active requests
                        self.active_requests[request.request_id] = request

                    # If we've taken all requests at this priority, clean up
                    if not priority_requests:
                        del self.waiting_requests[priority]

                # If we've filled the batch, we're done
                if len(batch) >= max_batch_size:
                    break

            # If we took some requests but not all, keep the event set
            if batch and any(self.waiting_requests.values()):
                self.new_request_event.set()
            else:
                self.new_request_event.clear()

        return batch

    async def wait_for_requests(self, timeout: Optional[float] = None) -> bool:
        """
        Wait for new requests to arrive.

        Args:
            timeout: Maximum time to wait (in seconds)

        Returns:
            True if new requests are available, False if timed out
        """
        try:
            await asyncio.wait_for(self.new_request_event.wait(), timeout)
            return True
        except asyncio.TimeoutError:
            return False

    async def mark_request_finished(self, request_id: str, result: torch.Tensor):
        """Mark a request as finished successfully."""
        async with self.lock:
            if request_id not in self.active_requests:
                return

            request = self.active_requests[request_id]
            request.mark_finished(result)

            # Move from active to completed
            del self.active_requests[request_id]
            self.completed_requests[request_id] = request

            # Notify completion
            asyncio.create_task(request.notify_finished())

    async def mark_request_error(self, request_id: str, error: str):
        """Mark a request as failed with an error."""
        async with self.lock:
            if request_id not in self.active_requests:
                return

            request = self.active_requests[request_id]
            request.mark_error(error)

            # Move from active to completed
            del self.active_requests[request_id]
            self.completed_requests[request_id] = request

            # Notify completion
            asyncio.create_task(request.notify_finished())

    async def notify_audio_chunk(self, request_id: str, chunk: torch.Tensor):
        """Notify a streaming audio chunk for a request."""
        async with self.lock:
            if request_id not in self.active_requests:
                return

            request = self.active_requests[request_id]
            await request.notify_audio_chunk(chunk)

    def cleanup_old_requests(self, max_age_seconds: float = 3600.0):
        """Clean up completed requests older than the specified age."""
        now = time.time()
        to_remove = []

        for request_id, request in self.completed_requests.items():
            if now - request.end_time > max_age_seconds:
                to_remove.append(request_id)

        for request_id in to_remove:
            del self.completed_requests[request_id]

    async def shutdown(self):
        """Shutdown the batch manager."""
        self.shutdown_flag = True

        # Cancel all waiting requests
        async with self.lock:
            for priority, requests in self.waiting_requests.items():
                for request in requests:
                    request.mark_cancelled()
                    asyncio.create_task(request.notify_finished())

            self.waiting_requests.clear()

            # Mark all active requests as error
            for request_id, request in self.active_requests.items():
                request.mark_error("Server shutting down")
                asyncio.create_task(request.notify_finished())
                self.completed_requests[request_id] = request

            self.active_requests.clear()