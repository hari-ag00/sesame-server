import asyncio
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from .router import router, model_runner
from ..engine.model_runner import ModelRunner
from ..utils.config import config

app = FastAPI(
    title="Sesame TTS Server",
    description="A server for Sesame TTS model inference",
    version="0.1.0",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add router
app.include_router(router)


@app.on_event("startup")
async def startup_event():
    """Initialize the model on server startup."""
    global model_runner

    # Create model runner
    model_runner.router.model_runner = ModelRunner()

    # Initialize model
    try:
        await model_runner.router.model_runner.initialize()
    except Exception as e:
        print(f"Error initializing model: {e}")
        raise HTTPException(status_code=500, detail=f"Error initializing model: {str(e)}")


@app.on_event("shutdown")
async def shutdown_event():
    """Shutdown the model on server shutdown."""
    global model_runner

    if model_runner.router.model_runner:
        # Shutdown model runner
        await model_runner.router.model_runner.shutdown()
        model_runner.router.model_runner = None


def start_server():
    """Start the FastAPI server."""
    uvicorn.run(
        "src.server.app:app",
        host=config.server.host,
        port=config.server.port,
        reload=config.server.debug,
        workers=config.server.workers,
    )