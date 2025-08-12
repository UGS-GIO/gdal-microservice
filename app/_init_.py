"""GDAL API Service - Serverless geospatial processing"""

__version__ = "1.0.0"

---

# app/config.py
"""Configuration management for GDAL API"""

import os
from pathlib import Path
from pydantic_settings import BaseSettings
from typing import Optional


class Settings(BaseSettings):
    """Application settings"""
    
    # Server configuration
    port: int = 8080
    host: str = "0.0.0.0"
    workers: int = 1
    
    # GCP configuration
    project_id: Optional[str] = os.getenv("GCP_PROJECT")
    gcs_bucket: Optional[str] = os.getenv("GCS_BUCKET")
    
    # File handling
    max_file_size: int = 524288000  # 500MB
    upload_dir: Path = Path("/tmp/gdal-uploads")
    output_dir: Path = Path("/tmp/gdal-outputs")
    workspace_dir: Path = Path("/tmp/gdal-workspace")
    
    # GDAL configuration
    gdal_cachemax: int = 512
    gdal_num_threads: str = "ALL_CPUS"
    cpl_tmpdir: str = "/tmp/gdal-workspace"
    
    # Job management
    job_timeout_seconds: int = 3600  # 1 hour
    job_cleanup_hours: int = 24
    
    # Security
    cors_origins: list = ["*"]
    cors_credentials: bool = True
    cors_methods: list = ["*"]
    cors_headers: list = ["*"]
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


settings = Settings()

---

# app/models.py
"""Pydantic models for request/response validation"""

from datetime import datetime
from typing import Optional, Dict, List, Any
from enum import Enum
from pydantic import BaseModel, Field, validator


class JobStatus(str, Enum):
    """Job processing status"""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class OutputFormat(str, Enum):
    """Supported output formats"""
    SHAPEFILE = "shapefile"
    GEOJSON = "geojson"
    GEOPACKAGE = "gpkg"
    CSV = "csv"
    KML = "kml"
    GML = "gml"
    FLATGEOBUF = "fgb"
    PARQUET = "parquet"
    TAB = "tab"
    MIF = "mif"


class ConversionOptions(BaseModel):
    """GDAL conversion options"""
    t_srs: Optional[str] = Field(None, description="Target SRS")
    s_srs: Optional[str] = Field(None, description="Source SRS")
    select: Optional[str] = Field(None, description="Select fields")
    where: Optional[str] = Field(None, description="SQL WHERE clause")
    sql: Optional[str] = Field(None, description="SQL statement")
    clipsrc: Optional[str] = Field(None, description="Clip source")
    simplify: Optional[float] = Field(None, description="Simplification tolerance")
    lco: Optional[Dict[str, str]] = Field(None, description="Layer creation options")
    dsco: Optional[Dict[str, str]] = Field(None, description="Dataset creation options")


class ConversionRequest(BaseModel):
    """Conversion request model"""
    input_url: str = Field(..., description="GCS URL of input file")
    output_format: OutputFormat = Field(..., description="Target output format")
    output_bucket: Optional[str] = Field(None, description="GCS bucket for output")
    output_path: Optional[str] = Field(None, description="GCS path for output")
    options: Optional[ConversionOptions] = Field(None, description="Conversion options")
    
    @validator('input_url')
    def validate_gcs_url(cls, v):
        if not v.startswith('gs://'):
            raise ValueError('Input URL must be a GCS URL (gs://...)')
        return v


class MergeRequest(BaseModel):
    """Merge multiple datasets request"""
    input_urls: List[str] = Field(..., min_items=2, description="List of GCS URLs to merge")
    output_format: OutputFormat = Field(..., description="Target output format")
    output_bucket: Optional[str] = Field(None, description="GCS bucket for output")
    output_path: Optional[str] = Field(None, description="GCS path for output")
    options: Optional[ConversionOptions] = Field(None, description="Merge options")


class DissolveRequest(BaseModel):
    """Dissolve features request"""
    input_url: str = Field(..., description="GCS URL of input file")
    dissolve_field: str = Field(..., description="Field to dissolve by")
    output_format: Optional[OutputFormat] = Field(None, description="Target output format")
    output_bucket: Optional[str] = Field(None, description="GCS bucket for output")
    output_path: Optional[str] = Field(None, description="GCS path for output")
    statistics: Optional[List[str]] = Field(None, description="Statistics to calculate")


class FileInfoResponse(BaseModel):
    """File information response"""
    driver: str
    layer_count: int
    layers: List[Dict[str, Any]]
    metadata: Optional[Dict[str, Any]] = None


class JobResponse(BaseModel):
    """Job response model"""
    id: str
    status: JobStatus
    input_format: Optional[str] = None
    output_format: Optional[str] = None
    input_source: str
    input_path: str
    output_path: Optional[str] = None
    error: Optional[str] = None
    created_at: datetime
    updated_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    options: Optional[Dict[str, Any]] = None
    progress: Optional[float] = None
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }

---

# app/utils.py
"""Utility functions for GDAL operations"""

import hashlib
import os
from pathlib import Path
from typing import Optional, Dict, Any
import tempfile
import shutil
from osgeo import ogr, osr, gdal


def get_file_hash(filepath: Path) -> str:
    """Calculate SHA256 hash of a file"""
    sha256_hash = hashlib.sha256()
    with open(filepath, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()


def validate_vector_file(filepath: Path) -> bool:
    """Validate that a file can be opened as vector data"""
    try:
        dataset = ogr.Open(str(filepath))
        if dataset is None:
            return False
        dataset = None  # Close
        return True
    except Exception:
        return False


def get_dataset_info(filepath: Path) -> Dict[str, Any]:
    """Get detailed information about a vector dataset"""
    dataset = ogr.Open(str(filepath))
    if not dataset:
        raise ValueError(f"Cannot open file: {filepath}")
    
    info = {
        "driver": dataset.GetDriver().GetName(),
        "layer_count": dataset.GetLayerCount(),
        "layers": [],
        "size_bytes": filepath.stat().st_size if filepath.exists() else None
    }
    
    for i in range(dataset.GetLayerCount()):
        layer = dataset.GetLayerByIndex(i)
        layer_info = {
            "index": i,
            "name": layer.GetName(),
            "feature_count": layer.GetFeatureCount(),
            "geometry_type": ogr.GeometryTypeToName(layer.GetGeomType()),
            "fields": [],
            "extent": None,
            "srs": None
        }
        
        # Get extent
        extent = layer.GetExtent()
        if extent:
            layer_info["extent"] = {
                "xmin": extent[0],
                "xmax": extent[1],
                "ymin": extent[2],
                "ymax": extent[3]
            }
        
        # Get SRS
        srs = layer.GetSpatialRef()
        if srs:
            layer_info["srs"] = {
                "proj4": srs.ExportToProj4(),
                "wkt": srs.ExportToWkt(),
                "epsg": None
            }
            
            # Try to get EPSG code
            if srs.GetAttrValue("AUTHORITY", 0) == "EPSG":
                layer_info["srs"]["epsg"] = srs.GetAttrValue("AUTHORITY", 1)
        
        # Get fields
        layer_defn = layer.GetLayerDefn()
        for j in range(layer_defn.GetFieldCount()):
            field = layer_defn.GetFieldDefn(j)
            layer_info["fields"].append({
                "name": field.GetName(),
                "type": field.GetTypeName(),
                "width": field.GetWidth(),
                "precision": field.GetPrecision()
            })
        
        info["layers"].append(layer_info)
    
    dataset = None  # Close
    return info


def create_vrt(input_files: list, output_path: Path) -> Path:
    """Create a VRT file for multiple inputs"""
    vrt_path = output_path.with_suffix('.vrt')
    
    # Build VRT
    vrt_options = gdal.BuildVRTOptions(separate=False)
    vrt = gdal.BuildVRT(str(vrt_path), input_files, options=vrt_options)
    vrt = None  # Close
    
    return vrt_path


def estimate_processing_time(file_size: int, operation: str) -> float:
    """Estimate processing time in seconds based on file size and operation"""
    # Basic estimation: 10MB/second for simple operations
    base_rate = 10 * 1024 * 1024  # 10MB/s
    
    # Adjust rate based on operation complexity
    operation_multipliers = {
        "convert": 1.0,
        "reproject": 1.5,
        "clip": 1.2,
        "merge": 2.0,
        "dissolve": 3.0,
        "simplify": 1.8
    }
    
    multiplier = operation_multipliers.get(operation, 1.0)
    estimated_seconds = (file_size / base_rate) * multiplier
    
    # Add overhead
    return estimated_seconds + 5.0


def cleanup_old_files(directory: Path, hours: int = 24):
    """Clean up files older than specified hours"""
    import time
    current_time = time.time()
    
    for filepath in directory.glob("*"):
        if filepath.is_file():
            file_age_hours = (current_time - filepath.stat().st_mtime) / 3600
            if file_age_hours > hours:
                try:
                    filepath.unlink()
                except Exception:
                    pass  # Ignore errors during cleanup

---

# app/exceptions.py
"""Custom exceptions for GDAL API"""

class GDALAPIException(Exception):
    """Base exception for GDAL API"""
    pass


class InvalidFileFormatError(GDALAPIException):
    """Raised when file format is not supported"""
    pass


class ProcessingError(GDALAPIException):
    """Raised when GDAL processing fails"""
    pass


class StorageError(GDALAPIException):
    """Raised when storage operations fail"""
    pass


class JobNotFoundError(GDALAPIException):
    """Raised when job is not found"""
    pass


class QuotaExceededError(GDALAPIException):
    """Raised when user exceeds quota"""
    pass

---

# tests/test_api.py
"""Test suite for GDAL API"""

import pytest
from fastapi.testclient import TestClient
from pathlib import Path
import json

from app.main import app


@pytest.fixture
def client():
    """Create test client"""
    return TestClient(app)


def test_health_check(client):
    """Test health check endpoint"""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "gdal_version" in data


def test_list_formats(client):
    """Test formats listing"""
    response = client.get("/api/v1/formats")
    assert response.status_code == 200
    formats = response.json()
    assert len(formats) > 0
    assert any(f["driver"] == "ESRI Shapefile" for f in formats)


def test_convert_missing_params(client):
    """Test conversion with missing parameters"""
    response = client.post("/api/v1/convert", json={})
    assert response.status_code == 422


def test_job_not_found(client):
    """Test getting non-existent job"""
    response = client.get("/api/v1/jobs/non-existent-id")
    assert response.status_code == 404


@pytest.mark.asyncio
async def test_file_upload(client, tmp_path):
    """Test file upload and conversion"""
    # Create a test file
    test_file = tmp_path / "test.geojson"
    test_file.write_text(json.dumps({
        "type": "FeatureCollection",
        "features": [{
            "type": "Feature",
            "geometry": {"type": "Point", "coordinates": [0, 0]},
            "properties": {"name": "Test"}
        }]
    }))
    
    with open(test_file, "rb") as f:
        response = client.post(
            "/api/v1/convert/upload",
            files={"file": ("test.geojson", f, "application/geo+json")},
            data={"output_format": "shapefile"}
        )
    
    assert response.status_code == 202
    job = response.json()
    assert job["status"] == "pending"
    assert job["id"] is not None