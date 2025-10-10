FROM nvidia/cuda:11.8-devel-ubuntu20.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV CUDA_HOME=/usr/local/cuda

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-dev \
    git \
    wget \
    curl \
    build-essential \
    cmake \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Set Python3 as default python
RUN ln -s /usr/bin/python3 /usr/bin/python

# Upgrade pip
RUN pip3 install --upgrade pip

# Set working directory
WORKDIR /app

# Copy requirements first for better Docker layer caching
COPY requirements.txt ./
COPY server/requirements.txt ./server/

# Install Python dependencies
RUN pip install -r requirements.txt
RUN pip install -r server/requirements.txt

# Copy project files
COPY . .

# Create necessary directories
RUN mkdir -p /app/data/models /app/output /app/logs

# Install DUSt3R and MASt3R
RUN git clone --recursive https://github.com/naver/dust3r.git && \
    cd dust3r && \
    pip install -e .

RUN git clone --recursive https://github.com/naver/mast3r.git && \
    cd mast3r && \
    pip install -e .

# Download models (if needed)
RUN python scripts/download_models.py

# Set permissions
RUN chmod +x scripts/*.py

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run the application
CMD ["python", "server/main.py"]
