
FROM nvidia/cuda:13.0.0-runtime-ubuntu22.04

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    python3-dev \
    curl \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libgtk-3-0 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Install UV
COPY --from=ghcr.io/astral-sh/uv:0.8.15 /uv /uvx /bin/

# Set working directory
WORKDIR /app

# Copy project files
COPY pyproject.toml uv.lock ./
COPY ./ml-processor/src/ ./src/

# Create virtual environment and install dependencies
RUN uv sync --locked

# Create necessary directories
RUN mkdir -p /app/logs /app/data /app/models

# Set environment variables
ENV PATH="/app/.venv/bin:$PATH"
ENV PYTHONPATH=/app/src
ENV NVIDIA_VISIBLE_DEVICES=all

CMD ["uv", "run", "python", "src/main.py"]
