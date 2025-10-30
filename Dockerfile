# 3D Measurement System - CPU-Only Dockerfile (Not Recommended)
# This is a fallback Dockerfile for systems without GPU
# WARNING: Performance will be significantly degraded without GPU acceleration

FROM ubuntu:22.04

# Metadata
LABEL maintainer="3D Measurement Team"
LABEL description="3D measurement system (CPU-only, for testing/development)"
LABEL version="2.0.0"

# Prevent interactive prompts
ENV DEBIAN_FRONTEND=noninteractive

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    # Python
    python3.10 \
    python3-pip \
    python3-dev \
    # Build tools
    build-essential \
    cmake \
    git \
    wget \
    curl \
    ca-certificates \
    # COLMAP (CPU version)
    colmap \
    # OpenCV dependencies
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libjpeg-dev \
    libpng-dev \
    libtiff-dev \
    # Cleanup
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Set Python3 as default
RUN ln -sf /usr/bin/python3.10 /usr/bin/python && \
    ln -sf /usr/bin/python3.10 /usr/bin/python3

# Upgrade pip
RUN python -m pip install --upgrade pip setuptools wheel

# Set working directory
WORKDIR /app

# Copy requirements (base only, no GPU dependencies)
COPY requirements/base.txt ./requirements/

# Install Python dependencies (CPU versions)
RUN pip install --no-cache-dir -r requirements/base.txt && \
    pip install --no-cache-dir \
    torch>=2.2.0 \
    torchvision>=0.17.0 \
    pycolmap>=0.5.0 \
    transformers>=4.35.0 \
    fastapi>=0.105.0 \
    uvicorn[standard]>=0.25.0 \
    opencv-contrib-python>=4.8.0

# Copy application source
COPY src/ ./src/
COPY configs/ ./configs/
COPY main.py ./main.py
COPY calibrate_scale.py ./calibrate_scale.py
COPY calibrate_depth_scale.py ./calibrate_depth_scale.py
COPY resize_images.py ./resize_images.py

# Create directories
RUN mkdir -p /app/output /app/logs /app/data /app/examples && \
    chmod -R 755 /app

# Make scripts executable
RUN chmod +x main.py calibrate_scale.py calibrate_depth_scale.py resize_images.py

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Start API server
CMD ["python", "main.py", "serve", "--host", "0.0.0.0", "--port", "8000"]
