"""
Server configuration settings
"""

import os
from pathlib import Path
from typing import List, Optional

class ServerConfig:
    # Server Settings
    HOST: str = os.getenv("HOST", "0.0.0.0")
    PORT: int = int(os.getenv("PORT", "8000"))
    DEBUG: bool = os.getenv("DEBUG", "false").lower() == "true"
    
    # Device Settings
    DEVICE: str = os.getenv("DEVICE", "cuda")
    MAX_WORKERS: int = int(os.getenv("MAX_WORKERS", "4"))
    
    # Processing Settings
    BATCH_SIZE: int = int(os.getenv("BATCH_SIZE", "1"))
    MAX_IMAGES: int = int(os.getenv("MAX_IMAGES", "25"))
    MAX_IMAGE_SIZE: int = int(os.getenv("MAX_IMAGE_SIZE", "4096"))
    
    # File Paths
    BASE_DIR: Path = Path(__file__).parent.parent
    MODEL_PATH: Path = Path(os.getenv("MODEL_PATH", BASE_DIR / "data" / "models"))
    UPLOAD_DIR: Path = Path(os.getenv("UPLOAD_DIR", BASE_DIR / "data" / "uploads"))
    OUTPUT_DIR: Path = Path(os.getenv("OUTPUT_DIR", BASE_DIR / "output"))
    LOG_DIR: Path = Path(os.getenv("LOG_DIR", BASE_DIR / "logs"))
    
    # Security
    SECRET_KEY: str = os.getenv("SECRET_KEY", "your-secret-key-change-in-production")
    ALLOWED_ORIGINS: List[str] = os.getenv("ALLOWED_ORIGINS", "*").split(",")
    
    # Database (optional)
    DATABASE_URL: Optional[str] = os.getenv("DATABASE_URL")
    REDIS_URL: str = os.getenv("REDIS_URL", "redis://localhost:6379")
    
    # External Services
    AWS_ACCESS_KEY_ID: Optional[str] = os.getenv("AWS_ACCESS_KEY_ID")
    AWS_SECRET_ACCESS_KEY: Optional[str] = os.getenv("AWS_SECRET_ACCESS_KEY")
    AWS_BUCKET_NAME: Optional[str] = os.getenv("AWS_BUCKET_NAME")
    
    # Monitoring
    SENTRY_DSN: Optional[str] = os.getenv("SENTRY_DSN")
    ENABLE_METRICS: bool = os.getenv("ENABLE_METRICS", "false").lower() == "true"
    
    # Logging
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    LOG_FORMAT: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    @classmethod
    def create_directories(cls):
        """Create necessary directories"""
        for directory in [cls.MODEL_PATH, cls.UPLOAD_DIR, cls.OUTPUT_DIR, cls.LOG_DIR]:
            directory.mkdir(parents=True, exist_ok=True)
    
    @classmethod
    def validate(cls):
        """Validate configuration"""
        errors = []
        
        if not cls.MODEL_PATH.exists():
            errors.append(f"Model path does not exist: {cls.MODEL_PATH}")
        
        if cls.DEVICE == "cuda":
            try:
                import torch
                if not torch.cuda.is_available():
                    errors.append("CUDA device specified but not available")
            except ImportError:
                errors.append("CUDA device specified but PyTorch not installed")
        
        if cls.MAX_WORKERS < 1:
            errors.append("MAX_WORKERS must be at least 1")
        
        if cls.MAX_IMAGES < 2:
            errors.append("MAX_IMAGES must be at least 2")
        
        return errors

# Create singleton instance
config = ServerConfig()
