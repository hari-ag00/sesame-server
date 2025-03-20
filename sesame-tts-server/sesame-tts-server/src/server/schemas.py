from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field

class GenerationRequest(BaseModel):
    text: str = Field(..., description="Text to synthesize")
    speaker: int = Field(0, description="Speaker ID")
    context: List[Dict[str, Any]] = Field(default_factory=list, description="Previous segments for context")
    max_audio_length_ms: Optional[float] = Field(None, description="Maximum audio length in milliseconds")
    temperature: Optional[float] = Field(None, description="Sampling temperature")
    topk: Optional[int] = Field(None, description="Top-k sampling parameter")
    priority: int = Field(0, description="Request priority (lower = higher priority)")
    stream: bool = Field(False, description="Stream audio chunks as they are generated")

class GenerationResponse(BaseModel):
    request_id: str = Field(..., description="Request ID")
    status: str = Field(..., description="Request status")

class RequestStatusResponse(BaseModel):
    request_id: str = Field(..., description="Request ID")
    status: str = Field(..., description="Request status")
    create_time: float = Field(..., description="Time when request was created")
    start_time: float = Field(0.0, description="Time when request started processing")
    end_time: float = Field(0.0, description="Time when request finished")
    waiting_time: float = Field(..., description="Time spent waiting in the queue")
    running_time: float = Field(..., description="Time spent processing")
    total_time: float = Field(..., description="Total time from creation to completion")
    error: Optional[str] = Field(None, description="Error message if request failed")

class CancelRequestResponse(BaseModel):
    request_id: str = Field(..., description="Request ID")
    cancelled: bool = Field(..., description="Whether the request was cancelled")

class ServerHealthResponse(BaseModel):
    status: str = Field(..., description="Server status")
    version: str = Field(..., description="Server version")
    model: str = Field(..., description="Model name")
    queue_size: int = Field(..., description="Number of requests in the queue")
    active_requests: int = Field(..., description="Number of requests being processed")
    uptime: float = Field(..., description="Server uptime in seconds")