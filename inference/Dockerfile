# Multi-stage Dockerfile for GPT inference server
# Stage 1: Install dependencies
# Stage 2: Copy code and run

FROM python:3.11-slim AS base

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

# Run server
ENV MODEL_CONFIG=small
ENV CHECKPOINT_PATH=checkpoints/best.pt
ENV TOKENIZER_PATH=tokenizer/bpe_tokenizer.json
ENV DEVICE=auto

CMD ["python", "-m", "uvicorn", "inference.server:app", "--host", "0.0.0.0", "--port", "8000"]
