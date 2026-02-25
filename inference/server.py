"""
server.py — FastAPI Inference Server
=======================================
REST API for text generation with:
  - /generate endpoint (POST)
  - /health endpoint (GET)
  - Streaming response support
  - Configurable sampling parameters
"""

import os
import sys
import json
import time
import torch
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from typing import Optional
import asyncio

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import ModelConfig, small_config, medium_config
from model.transformer import GPT
from tokenizer.sp_wrapper import load_tokenizer
from inference.generate import generate


# ─── Configuration ──────────────────────────────────────

MODEL_CONFIG = os.environ.get("MODEL_CONFIG", "small")
CHECKPOINT_PATH = os.environ.get("CHECKPOINT_PATH", "checkpoints/best.pt")
TOKENIZER_PATH = os.environ.get("TOKENIZER_PATH", "tokenizer/sp_bpe.model")
DEVICE = os.environ.get("DEVICE", "auto")

app = FastAPI(
    title="GPT Inference Server",
    description="Autoregressive text generation API",
    version="1.0.0",
)


# ─── Request / Response Models ──────────────────────────

class GenerateRequest(BaseModel):
    prompt: str = Field(..., description="Input text prompt")
    max_new_tokens: int = Field(100, ge=1, le=1024)
    temperature: float = Field(0.8, ge=0.0, le=2.0)
    top_k: Optional[int] = Field(50, ge=1)
    top_p: Optional[float] = Field(0.9, ge=0.0, le=1.0)
    greedy: bool = Field(False)


class GenerateResponse(BaseModel):
    generated_text: str
    prompt: str
    tokens_generated: int
    time_seconds: float
    tokens_per_second: float


class HealthResponse(BaseModel):
    status: str
    model_config: str
    device: str
    vocab_size: int
    parameters_M: float


# ─── Model Loading ──────────────────────────────────────

model = None
tokenizer = None
model_cfg = None
device = None


@app.on_event("startup")
async def load_model():
    """Load model and tokenizer on startup."""
    global model, tokenizer, model_cfg, device

    # Select device
    if DEVICE == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(DEVICE)

    print(f"Device: {device}")

    # Load config
    config_fn = small_config if MODEL_CONFIG == "small" else medium_config
    model_cfg, _ = config_fn()

    # Load tokenizer
    if os.path.exists(TOKENIZER_PATH):
        tokenizer = load_tokenizer(TOKENIZER_PATH)
        print(f"Tokenizer loaded from {TOKENIZER_PATH}")
    else:
        print(f"ERROR: Tokenizer not found at {TOKENIZER_PATH}")
        print("Train one first — see README.md")

    # Load model
    model = GPT(model_cfg)
    if os.path.exists(CHECKPOINT_PATH):
        checkpoint = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state"])
        print(f"Model loaded from {CHECKPOINT_PATH}")
    else:
        print("WARNING: No checkpoint found, using random weights")

    model = model.to(device)
    model.eval()
    print("Server ready!")


# ─── Endpoints ──────────────────────────────────────────

@app.get("/health", response_model=HealthResponse)
async def health():
    """Health check endpoint."""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    n_params = sum(p.numel() for p in model.parameters())
    return HealthResponse(
        status="healthy",
        model_config=MODEL_CONFIG,
        device=str(device),
        vocab_size=model_cfg.vocab_size,
        parameters_M=n_params / 1e6,
    )


@app.post("/generate", response_model=GenerateResponse)
async def generate_text(request: GenerateRequest):
    """Generate text from a prompt."""
    if model is None or tokenizer is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        # Encode prompt
        input_ids = tokenizer.encode(request.prompt, add_special_tokens=True)
        input_tensor = torch.tensor([input_ids], dtype=torch.long, device=device)

        # Generate
        output_ids, stats = generate(
            model,
            input_tensor,
            max_new_tokens=request.max_new_tokens,
            temperature=request.temperature,
            top_k=request.top_k,
            top_p=request.top_p,
            greedy=request.greedy,
            eos_token_id=tokenizer.eos_id,
            use_cache=True,
        )

        # Decode
        generated = tokenizer.decode(
            output_ids[0].tolist(), skip_special_tokens=True
        )

        return GenerateResponse(
            generated_text=generated,
            prompt=request.prompt,
            tokens_generated=stats["tokens_generated"],
            time_seconds=stats["total_time_s"],
            tokens_per_second=stats["tokens_per_second"],
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
