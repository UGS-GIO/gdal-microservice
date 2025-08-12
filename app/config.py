"""Configuration management for GDAL API Service"""

import os
from pathlib import Path
from typing import Optional, List
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """
    Application settings with environment variable support.
    
    These can be overridden by setting environment variables with the same name.
    For example, setting PORT=8080 will override the default port.
    """
    
    # ==========================================
    # Server Configuration
    # ==========================================
    port: int = int(os.getenv("PORT", 8080))
    host: str = "0.0.0.0"
    workers: int = 1
    reload: bool = False  # Set to True for development
    
    # ==========================================
    # GCP Configuration
    # ==========================================
    project_id: str = os.getenv("PROJECT_ID", "ut-dnr-ugs-backend-tools")
    gcs_bucket: str = os.getenv("GCS_BUCKET", "ut-dnr-ugs-gdal-workspace")
    output_bucket: str = os.getenv("OUTPUT_BUCKET", "ut-dnr-ugs-gdal-output")
    region: str = os.getenv("REGION", "us-central1")
    
    # ==========================================
    # File Handling Configuration
    # ==========================================
    max_file_size: int = int(os.getenv("MAX_FILE_SIZE", 524288000))  # 500MB default
    upload_dir: Path = Path(os.getenv("UPLOAD_DIR", "/tmp/gdal-uploads"))
    output_dir: Path = Path(os.getenv("OUTPUT_DIR", "/tmp/gdal-outputs"))
    workspace_dir: Path = Path(os.getenv("WORKSPACE_DIR", "/tmp/gdal-workspace"))
    allowed_extensions: List[str] = [
        ".shp", ".dbf", ".shx", ".prj",  # Shapefile
        ".gpkg",  # GeoPackage
        ".geojson", ".json",  # GeoJSON
        ".kml", ".kmz",  # KML
        ".gdb",  # File Geodatabase
        ".csv",  # CSV
        ".fgb",  # FlatGeobuf
        ".parquet",  # Parquet
        ".gml",  # GML
        ".tab", ".mif", ".mid"  # MapInfo
    ]
    
    # ==========================================
    # GDAL Configuration
    # ==========================================
    gdal_cachemax: int = int(os.getenv("GDAL_CACHEMAX", 1024))  # MB
    gdal_num_threads: str = os.getenv("GDAL_NUM_THREADS", "ALL_CPUS")
    cpl_tmpdir: str = os.getenv("CPL_TMPDIR", "/tmp/gdal-workspace")
    gdal_data: str = os.getenv("GDAL_DATA", "/usr/share/gdal")
    proj_lib: str = os.getenv("PROJ_LIB", "/usr/share/proj")
    
    # ==========================================
    # Job Management
    # ==========================================
    job_timeout_seconds: int = int(os.getenv("JOB_TIMEOUT", 3600))  # 1 hour
    job_cleanup_hours: int = int(os.getenv("JOB_CLEANUP_HOURS", 24))
    max_concurrent_jobs: int = int(os.getenv("MAX_CONCURRENT_JOBS", 10))
    
    # ==========================================
    # API Configuration
    # ==========================================
    api_title: str = "GDAL API Service"
    api_description: str = "Serverless GDAL processing API for vector geospatial data"
    api_version: str = "1.0.0"
    api_prefix: str = "/api/v1"
    
    # ==========================================
    # Security Configuration
    # ==========================================
    cors_enabled: bool = True
    cors_origins: List[str] = ["*"]  # Configure for production
    cors_credentials: bool = True
    cors_methods: List[str] = ["GET", "POST", "PUT", "DELETE", "OPTIONS"]
    cors_headers: List[str] = ["*"]
    
    # Rate limiting (requests per minute)
    rate_limit_enabled: bool = False
    rate_limit_requests: int = 100
    rate_limit_window: int = 60  # seconds
    
    # ==========================================
    # Logging Configuration
    # ==========================================
    log_level: str = os.getenv("LOG_LEVEL", "INFO")
    log_format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    log_file: Optional[str] = None  # Set to enable file logging
    
    # ==========================================
    # Monitoring Configuration
    # ==========================================
    enable_metrics: bool = True
    enable_tracing: bool = False
    
    # ==========================================
    # Environment Configuration
    # ==========================================
    environment: str = os.getenv("ENVIRONMENT", "development")
    debug: bool = environment == "development"
    testing: bool = False
    
    class Config:
        """Pydantic configuration"""
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False
        
    def __init__(self, **kwargs):
        """Initialize settings and create directories"""
        super().__init__(**kwargs)
        
        # Create necessary directories
        self.upload_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.workspace_dir.mkdir(parents=True, exist_ok=True)
        
        # Set GDAL environment variables
        os.environ["GDAL_CACHEMAX"] = str(self.gdal_cachemax)
        os.environ["GDAL_NUM_THREADS"] = self.gdal_num_threads
        os.environ["CPL_TMPDIR"] = self.cpl_tmpdir
        if self.gdal_data:
            os.environ["GDAL_DATA"] = self.gdal_data
        if self.proj_lib:
            os.environ["PROJ_LIB"] = self.proj_lib
    
    def is_production(self) -> bool:
        """Check if running in production environment"""
        return self.environment.lower() == "production"
    
    def is_development(self) -> bool:
        """Check if running in development environment"""
        return self.environment.lower() == "development"
    
    def get_storage_path(self, path_type: str = "workspace") -> Path:
        """Get the appropriate storage path based on type"""
        paths = {
            "upload": self.upload_dir,
            "output": self.output_dir,
            "workspace": self.workspace_dir
        }
        return paths.get(path_type, self.workspace_dir)
    
    def validate_file_extension(self, filename: str) -> bool:
        """Check if file extension is allowed"""
        return any(filename.lower().endswith(ext) for ext in self.allowed_extensions)
    
    def get_gcs_url(self, bucket: str, path: str) -> str:
        """Construct a GCS URL"""
        return f"gs://{bucket}/{path}"
    
    def get_max_file_size_mb(self) -> float:
        """Get max file size in megabytes"""
        return self.max_file_size / (1024 * 1024)


# Create a singleton settings instance
settings = Settings()


# Logging configuration
import logging
import sys

def setup_logging():
    """Configure logging for the application"""
    log_level = getattr(logging, settings.log_level.upper(), logging.INFO)
    
    # Configure root logger
    logging.basicConfig(
        level=log_level,
        format=settings.log_format,
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    # Set specific loggers
    logging.getLogger("uvicorn").setLevel(log_level)
    logging.getLogger("fastapi").setLevel(log_level)
    
    # Quiet down some noisy loggers
    if not settings.debug:
        logging.getLogger("googleapiclient").setLevel(logging.WARNING)
        logging.getLogger("google.cloud").setLevel(logging.WARNING)
    
    logger = logging.getLogger(__name__)
    logger.info(f"Logging configured at level: {settings.log_level}")
    logger.info(f"Environment: {settings.environment}")
    logger.info(f"Project ID: {settings.project_id}")
    
    return logger


# Initialize logging when module is imported
logger = setup_logging()