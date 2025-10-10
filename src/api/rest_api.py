"""
FastAPI REST API for GPU-accelerated 3D measurement.

Provides endpoints for measurement, health checks, and system info.
"""

import torch
import numpy as np
import logging
import tempfile
import shutil
from typing import List, Optional, Dict
from pathlib import Path
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import cv2

from ..core.measurement_system_gpu import MeasurementSystemGPU
from ..core.config import SystemConfig, get_gpu_info

logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="3D Measurement API",
    description="GPU-accelerated 3D measurement system using COLMAP and Metric3D",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global measurement system instance
measurement_system: Optional[MeasurementSystemGPU] = None


# Pydantic models
class MeasurementResponse(BaseModel):
    """Response model for measurement endpoint."""
    
    success: bool
    measurements: Dict[str, float]
    confidence: float
    processing_times: Dict[str, float]
    scale_recovery: Dict
    reconstruction_stats: Dict
    pointcloud_path: Optional[str] = None


class HealthResponse(BaseModel):
    """Response model for health check."""
    
    status: str
    gpu_available: bool
    gpu_info: Dict
    system_ready: bool


class BenchmarkRequest(BaseModel):
    """Request model for benchmark endpoint."""
    
    num_images: int = Field(default=5, ge=3, le=20)
    image_size: tuple = Field(default=(1024, 1024))
    num_runs: int = Field(default=3, ge=1, le=10)


# Startup and shutdown events
@app.on_event("startup")
async def startup_event():
    """Initialize measurement system on startup."""
    global measurement_system
    
    try:
        logger.info("Initializing 3D measurement system...")
        
        # Check GPU availability
        if not torch.cuda.is_available():
            raise RuntimeError("GPU not available - system requires CUDA")
        
        # Initialize system
        config = SystemConfig()
        measurement_system = MeasurementSystemGPU(config)
        
        logger.info("System initialized successfully")
        
    except Exception as e:
        logger.error(f"Failed to initialize system: {e}")
        measurement_system = None
        raise


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown."""
    global measurement_system
    
    logger.info("Shutting down system...")
    
    if measurement_system:
        # Clear GPU memory
        torch.cuda.empty_cache()
        measurement_system = None
    
    logger.info("Shutdown complete")


# API endpoints
@app.get("/", tags=["General"])
async def root():
    """Root endpoint."""
    return {
        "service": "3D Measurement API",
        "version": "2.0.0",
        "status": "running",
        "docs": "/docs"
    }


@app.get("/health", response_model=HealthResponse, tags=["General"])
async def health_check():
    """
    Check system health and GPU status.
    
    Returns:
        Health status with GPU information
    """
    gpu_info = get_gpu_info()
    
    return HealthResponse(
        status="healthy" if measurement_system else "degraded",
        gpu_available=gpu_info['available'],
        gpu_info=gpu_info,
        system_ready=measurement_system is not None
    )


@app.post("/measure", response_model=MeasurementResponse, tags=["Measurement"])
async def measure_endpoint(
    files: List[UploadFile] = File(..., description="Image files to process"),
    imu_data: Optional[str] = Form(None, description="IMU data as JSON string"),
    metadata: Optional[str] = Form(None, description="Image metadata as JSON string")
):
    """
    Measure dimensions from uploaded images.
    
    Args:
        files: List of image files (minimum 3 images)
        imu_data: Optional IMU sensor data as JSON string
        metadata: Optional image metadata (EXIF) as JSON string
        
    Returns:
        Measurement results with dimensions in centimeters
        
    Raises:
        HTTPException: If system not ready or processing fails
    """
    if measurement_system is None:
        raise HTTPException(
            status_code=503,
            detail="Measurement system not initialized"
        )
    
    # Validate inputs
    if len(files) < 3:
        raise HTTPException(
            status_code=400,
            detail=f"At least 3 images required, got {len(files)}"
        )
    
    if len(files) > 50:
        raise HTTPException(
            status_code=400,
            detail=f"Maximum 50 images allowed, got {len(files)}"
        )
    
    temp_dir = None
    
    try:
        # Create temporary directory for images
        temp_dir = tempfile.mkdtemp()
        temp_path = Path(temp_dir)
        
        # Load images
        images = []
        image_paths = []
        
        for i, file in enumerate(files):
            # Validate file type
            if not file.content_type.startswith('image/'):
                raise HTTPException(
                    status_code=400,
                    detail=f"File {file.filename} is not an image"
                )
            
            # Read and decode image
            contents = await file.read()
            nparr = np.frombuffer(contents, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if img is None:
                raise HTTPException(
                    status_code=400,
                    detail=f"Failed to decode image {file.filename}"
                )
            
            # Convert BGR to RGB
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            images.append(img)
            
            # Save to temp directory
            img_path = temp_path / f"image_{i:04d}.jpg"
            cv2.imwrite(str(img_path), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
            image_paths.append(img_path)
        
        # Parse optional data
        imu_parsed = None
        if imu_data:
            import json
            imu_parsed = json.loads(imu_data)
        
        metadata_parsed = None
        if metadata:
            import json
            metadata_parsed = json.loads(metadata)
        
        # Run measurement
        logger.info(f"Processing {len(images)} images...")
        result = measurement_system.measure(
            images=images,
            image_paths=image_paths,
            imu_data=imu_parsed,
            metadata=metadata_parsed
        )
        
        # Convert to response model
        response = MeasurementResponse(**result.to_dict())
        
        logger.info("Measurement completed successfully")
        return response
        
    except HTTPException:
        raise
    
    except Exception as e:
        logger.error(f"Measurement failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Measurement processing failed: {str(e)}"
        )
    
    finally:
        # Cleanup temporary directory
        if temp_dir and Path(temp_dir).exists():
            shutil.rmtree(temp_dir, ignore_errors=True)


@app.post("/benchmark", tags=["Performance"])
async def benchmark_endpoint(request: BenchmarkRequest):
    """
    Run performance benchmark.
    
    Args:
        request: Benchmark parameters
        
    Returns:
        Benchmark results with timing statistics
        
    Raises:
        HTTPException: If system not ready
    """
    if measurement_system is None:
        raise HTTPException(
            status_code=503,
            detail="Measurement system not initialized"
        )
    
    try:
        logger.info(f"Running benchmark: {request.dict()}")
        
        results = measurement_system.benchmark(
            num_images=request.num_images,
            image_size=request.image_size,
            num_runs=request.num_runs
        )
        
        return {
            "success": True,
            "results": results
        }
        
    except Exception as e:
        logger.error(f"Benchmark failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Benchmark failed: {str(e)}"
        )


@app.get("/gpu-stats", tags=["Performance"])
async def gpu_stats():
    """
    Get current GPU statistics.
    
    Returns:
        GPU memory and utilization statistics
    """
    if not torch.cuda.is_available():
        raise HTTPException(
            status_code=503,
            detail="GPU not available"
        )
    
    gpu_info = get_gpu_info()
    
    return {
        "success": True,
        "gpu_stats": gpu_info
    }


# Error handlers
@app.exception_handler(torch.cuda.OutOfMemoryError)
async def cuda_oom_handler(request, exc):
    """Handle CUDA out of memory errors."""
    logger.error("GPU out of memory")
    torch.cuda.empty_cache()
    
    return HTTPException(
        status_code=507,
        detail="GPU out of memory. Try reducing image size or number of images."
    )


# Main entry point
def start_server(
    host: str = "0.0.0.0",
    port: int = 8000,
    reload: bool = False
):
    """
    Start the FastAPI server.
    
    Args:
        host: Host address
        port: Port number
        reload: Enable auto-reload for development
    """
    import uvicorn
    
    uvicorn.run(
        "src.api.rest_api:app",
        host=host,
        port=port,
        reload=reload,
        log_level="info"
    )


if __name__ == "__main__":
    start_server()

