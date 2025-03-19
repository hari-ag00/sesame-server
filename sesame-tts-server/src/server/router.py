import asyncio
import time
from typing import Dict, List, Optional, Any
import io
import wave
import numpy as np
import torch
import torchaudio
from fastapi import APIRouter, HTTPException, Response, WebSocket, WebSocketDisconnect, BackgroundTasks, Query
from fastapi.responses import StreamingResponse
from .schemas import (
    GenerationRequest,
    GenerationResponse,
    RequestStatusResponse,
    CancelRequestResponse,
    ServerHealthResponse,
)
from ..engine.model_runner import ModelRunner, Segment
from ..utils.config import config

router = APIRouter()

# Global model runner instance
model_runner: Optional[ModelRunner] = None
start_time = time.time()


async def stream_audio_chunk(request_id: str, chunk: torch.Tensor, websocket: WebSocket):
    """Stream an audio chunk to a WebSocket client."""
    if chunk.numel() == 0:
        return

    # Convert to WAV bytes
    buffer = io.BytesIO()
    torchaudio.save(buffer, chunk.unsqueeze(0).cpu(), model_runner.sample_rate, format="wav")
    buffer.seek(0)

    # Send chunk
    await websocket.send_bytes(buffer.read())


async def finish_request(request_id: str, audio: Optional[torch.Tensor], error: Optional[str], websocket: WebSocket):
    """Send final status to WebSocket client."""
    if websocket:
        if error:
            await websocket.send_json({"status": "error", "error": error})
        else:
            await websocket.send_json({"status": "finished"})
        await websocket.close()


def audio_to_wav_bytes(audio: torch.Tensor, sample_rate: int) -> bytes:
    """Convert audio tensor to WAV bytes."""
    buffer = io.BytesIO()
    torchaudio.save(buffer, audio.unsqueeze(0).cpu(), sample_rate, format="wav")
    buffer.seek(0)
    return buffer.read()


@router.get("/health", response_model=ServerHealthResponse)
async def health_check():
    """Check server health."""
    if model_runner is None:
        raise HTTPException(status_code=503, detail="Server initializing")

    # Get queue stats
    queue_size = 0
    active_requests = 0

    for priority, requests in model_runner.batch_manager.waiting_requests.items():
        queue_size += len(requests)

    active_requests = len(model_runner.batch_manager.active_requests)

    return {
        "status": "healthy",
        "version": "0.1.0",
        "model": config.model.model_path,
        "queue_size": queue_size,
        "active_requests": active_requests,
        "uptime": time.time() - start_time,
    }


@router.post("/generate", response_model=GenerationResponse)
async def generate(request: GenerationRequest, background_tasks: BackgroundTasks):
    """Generate audio from text."""
    if model_runner is None:
        raise HTTPException(status_code=503, detail="Server initializing")

    # Process context if provided
    context = []
    for ctx in request.context:
        # Convert context dict to Segment object
        if "text" in ctx and "speaker" in ctx and "audio" in ctx:
            # Convert audio data from base64 or handle as needed
            # For simplicity, we'll skip complex audio conversion here
            # In a real implementation, you would decode audio from the request
            audio = torch.zeros(1, dtype=torch.float32)  # Placeholder
            context.append(Segment(speaker=ctx["speaker"], text=ctx["text"], audio=audio))

    # Add request to queue
    request_id = await model_runner.add_request(
        text=request.text,
        speaker=request.speaker,
        context=context,
        max_audio_length_ms=request.max_audio_length_ms,
        temperature=request.temperature,
        topk=request.topk,
        priority=request.priority,
    )

    return {
        "request_id": request_id,
        "status": "queued",
    }


@router.get("/status/{request_id}", response_model=RequestStatusResponse)
async def get_status(request_id: str):
    """Get the status of a request."""
    if model_runner is None:
        raise HTTPException(status_code=503, detail="Server initializing")

    status = await model_runner.get_request_status(request_id)
    if status is None:
        raise HTTPException(status_code=404, detail="Request not found")

    return status


@router.post("/cancel/{request_id}", response_model=CancelRequestResponse)
async def cancel_request(request_id: str):
    """Cancel a queued request."""
    if model_runner is None:
        raise HTTPException(status_code=503, detail="Server initializing")

    cancelled = await model_runner.cancel_request(request_id)

    return {
        "request_id": request_id,
        "cancelled": cancelled,
    }


@router.get("/audio/{request_id}")
async def get_audio(request_id: str):
    """Get the generated audio for a request."""
    if model_runner is None:
        raise HTTPException(status_code=503, detail="Server initializing")

    status = await model_runner.get_request_status(request_id)
    if status is None:
        raise HTTPException(status_code=404, detail="Request not found")

    if status["status"] != "FINISHED":
        raise HTTPException(status_code=400, detail=f"Request not finished (status: {status['status']})")

    # Get the result from completed_requests
    request = model_runner.batch_manager.completed_requests.get(request_id)
    if request is None or request.result is None:
        raise HTTPException(status_code=404, detail="Audio not found")

    # Convert audio to WAV
    audio_bytes = audio_to_wav_bytes(request.result, model_runner.sample_rate)

    # Return as streaming response
    return Response(
        content=audio_bytes,
        media_type="audio/wav",
        headers={
            "Content-Disposition": f"attachment; filename={request_id}.wav",
        },
    )


@router.websocket("/stream/{request_id}")
async def stream_audio(websocket: WebSocket, request_id: str):
    """Stream audio for a request that's already been submitted."""
    if model_runner is None:
        await websocket.close(code=1013, reason="Server initializing")
        return

    await websocket.accept()

    # Check if request exists
    status = await model_runner.get_request_status(request_id)
    if status is None:
        await websocket.send_json({"status": "error", "error": "Request not found"})
        await websocket.close()
        return

    # If request is already finished, send the audio directly
    if status["status"] == "FINISHED":
        request = model_runner.batch_manager.completed_requests.get(request_id)
        if request is not None and request.result is not None:
            # Send audio as WAV
            audio_bytes = audio_to_wav_bytes(request.result, model_runner.sample_rate)
            await websocket.send_bytes(audio_bytes)
            await websocket.send_json({"status": "finished"})
        else:
            await websocket.send_json({"status": "error", "error": "Audio not found"})

        await websocket.close()
        return

    # If request is in error state, send the error
    if status["status"] == "ERROR":
        await websocket.send_json({"status": "error", "error": status["error"]})
        await websocket.close()
        return

    # If request is cancelled, send cancelled status
    if status["status"] == "CANCELLED":
        await websocket.send_json({"status": "cancelled"})
        await websocket.close()
        return

    # Register callbacks for streaming if request is waiting or running
    request = None

    if status["status"] == "WAITING":
        # Find request in waiting queue
        for priority, requests in model_runner.batch_manager.waiting_requests.items():
            for req in requests:
                if req.request_id == request_id:
                    request = req
                    break
            if request:
                break
    elif status["status"] == "RUNNING":
        # Get from active requests
        request = model_runner.batch_manager.active_requests.get(request_id)

    if request is None:
        await websocket.send_json({"status": "error", "error": "Request not found"})
        await websocket.close()
        return

    # Register callbacks
    request.on_audio_chunk = lambda rid, chunk: stream_audio_chunk(rid, chunk, websocket)
    request.on_finished = lambda rid, audio, error: finish_request(rid, audio, error, websocket)

    # Wait for client to disconnect or request to finish
    try:
        while True:
            # Check for client messages or disconnection
            await websocket.receive_text()
    except WebSocketDisconnect:
        # Client disconnected, but don't cancel the request
        pass
    except Exception as e:
        print(f"WebSocket error: {e}")


@router.websocket("/generate-stream")
async def generate_and_stream(websocket: WebSocket):
    """Generate audio and stream it in real-time."""
    if model_runner is None:
        await websocket.close(code=1013, reason="Server initializing")
        return

    await websocket.accept()

    try:
        # Receive generation parameters
        data = await websocket.receive_json()

        # Validate request
        try:
            request = GenerationRequest(**data)
        except Exception as e:
            await websocket.send_json({"status": "error", "error": f"Invalid request: {str(e)}"})
            await websocket.close()
            return

        # Process context if provided
        context = []
        for ctx in request.context:
            # Convert context dict to Segment object
            if "text" in ctx and "speaker" in ctx and "audio" in ctx:
                # In a real implementation, you would decode audio from the request
                audio = torch.zeros(1, dtype=torch.float32)  # Placeholder
                context.append(Segment(speaker=ctx["speaker"], text=ctx["text"], audio=audio))

        # Create callbacks for streaming
        async def on_audio_chunk(request_id: str, chunk: torch.Tensor):
            if chunk.numel() == 0:
                return

            # Convert to WAV bytes
            audio_bytes = audio_to_wav_bytes(chunk, model_runner.sample_rate)

            # Send chunk
            await websocket.send_bytes(audio_bytes)

        async def on_finished(request_id: str, audio: Optional[torch.Tensor], error: Optional[str]):
            if error:
                await websocket.send_json({"status": "error", "error": error})
            else:
                await websocket.send_json({"status": "finished"})

        # Add request to queue
        request_id = await model_runner.add_request(
            text=request.text,
            speaker=request.speaker,
            context=context,
            max_audio_length_ms=request.max_audio_length_ms,
            temperature=request.temperature,
            topk=request.topk,
            priority=request.priority,
            on_audio_chunk=on_audio_chunk if request.stream else None,
            on_finished=on_finished,
        )

        # Send request ID to client
        await websocket.send_json({"status": "queued", "request_id": request_id})

        # Wait for client to disconnect
        while True:
            # Check for client messages or disconnection
            await websocket.receive_text()

    except WebSocketDisconnect:
        # Client disconnected
        pass
    except Exception as e:
        print(f"WebSocket error: {e}")
        try:
            await websocket.send_json({"status": "error", "error": str(e)})
            await websocket.close()
        except:
            pass
