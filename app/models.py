"""Pydantic models for request/response validation in GDAL API Service"""

from datetime import datetime
from typing import Optional, Dict, List, Any, Union
from enum import Enum
from pydantic import BaseModel, Field, validator, root_validator
import re


# ==========================================
# Enums
# ==========================================

class JobStatus(str, Enum):
    """Job processing status enum"""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class OutputFormat(str, Enum):
    """Supported output format enum"""
    SHAPEFILE = "shapefile"
    SHP = "shp"
    GEOJSON = "geojson"
    JSON = "json"
    GEOPACKAGE = "gpkg"
    GPKG = "gpkg"
    CSV = "csv"
    KML = "kml"
    KMZ = "kmz"
    GML = "gml"
    FLATGEOBUF = "fgb"
    FGB = "fgb"
    PARQUET = "parquet"
    TAB = "tab"
    MIF = "mif"
    GDB = "gdb"


class GeometryType(str, Enum):
    """Geometry type enum"""
    POINT = "Point"
    MULTIPOINT = "MultiPoint"
    LINESTRING = "LineString"
    MULTILINESTRING = "MultiLineString"
    POLYGON = "Polygon"
    MULTIPOLYGON = "MultiPolygon"
    GEOMETRYCOLLECTION = "GeometryCollection"
    UNKNOWN = "Unknown"


# ==========================================
# Request Models
# ==========================================

class ConversionOptions(BaseModel):
    """GDAL conversion options"""
    t_srs: Optional[str] = Field(None, description="Target SRS (e.g., EPSG:4326)")
    s_srs: Optional[str] = Field(None, description="Source SRS override")
    select: Optional[str] = Field(None, description="Comma-separated list of fields to select")
    where: Optional[str] = Field(None, description="SQL WHERE clause for filtering")
    sql: Optional[str] = Field(None, description="Complete SQL statement")
    clipsrc: Optional[Union[List[float], str]] = Field(None, description="Clip source (bbox or file)")
    simplify: Optional[float] = Field(None, description="Simplification tolerance")
    skipfailures: Optional[bool] = Field(False, description="Skip failures")
    limit: Optional[int] = Field(None, description="Limit number of features")
    preserve_fid: Optional[bool] = Field(False, description="Preserve feature IDs")
    lco: Optional[Dict[str, str]] = Field(None, description="Layer creation options")
    dsco: Optional[Dict[str, str]] = Field(None, description="Dataset creation options")
    
    @validator('t_srs', 's_srs')
    def validate_srs(cls, v):
        """Validate SRS format"""
        if v is None:
            return v
        
        # Check for EPSG format
        if v.upper().startswith("EPSG:"):
            try:
                epsg_code = int(v.split(":")[1])
                if epsg_code < 1 or epsg_code > 100000:
                    raise ValueError("Invalid EPSG code")
            except (IndexError, ValueError):
                raise ValueError("Invalid EPSG format. Use EPSG:XXXX")
        
        return v
    
    @validator('clipsrc')
    def validate_clipsrc(cls, v):
        """Validate clip source"""
        if v is None:
            return v
        
        if isinstance(v, list):
            if len(v) != 4:
                raise ValueError("Bounding box must have 4 values: [xmin, ymin, xmax, ymax]")
            if v[0] >= v[2] or v[1] >= v[3]:
                raise ValueError("Invalid bounding box: min values must be less than max values")
        
        return v


class ConversionRequest(BaseModel):
    """Vector format conversion request"""
    input_url: str = Field(..., description="GCS URL of input file (gs://...)")
    output_format: OutputFormat = Field(..., description="Target output format")
    output_bucket: Optional[str] = Field(None, description="GCS bucket for output")
    output_path: Optional[str] = Field(None, description="Path within bucket for output")
    options: Optional[ConversionOptions] = Field(None, description="Conversion options")
    
    @validator('input_url')
    def validate_gcs_url(cls, v):
        """Validate GCS URL format"""
        if not v.startswith('gs://'):
            raise ValueError('Input URL must be a GCS URL (gs://bucket/path)')
        
        # Basic validation of bucket/path structure
        parts = v[5:].split('/', 1)
        if len(parts) != 2 or not parts[0] or not parts[1]:
            raise ValueError('Invalid GCS URL format. Expected: gs://bucket/path/to/file')
        
        return v
    
    @validator('output_path')
    def validate_output_path(cls, v):
        """Validate output path"""
        if v is None:
            return v
        
        # Remove leading/trailing slashes
        v = v.strip('/')
        
        # Check for invalid characters
        if not re.match(r'^[a-zA-Z0-9/_\-\.]+$', v):
            raise ValueError('Output path contains invalid characters')
        
        return v


class ReprojectionRequest(BaseModel):
    """Reprojection request"""
    input_url: str = Field(..., description="GCS URL of input file")
    target_srs: str = Field(..., description="Target SRS (e.g., EPSG:4326)")
    source_srs: Optional[str] = Field(None, description="Source SRS override")
    output_format: Optional[OutputFormat] = Field(None, description="Output format (defaults to input format)")
    output_bucket: Optional[str] = Field(None, description="GCS bucket for output")
    output_path: Optional[str] = Field(None, description="Path within bucket for output")


class ClipRequest(BaseModel):
    """Clip operation request"""
    input_url: str = Field(..., description="GCS URL of input file")
    clip_bounds: Optional[List[float]] = Field(None, description="Bounding box [xmin, ymin, xmax, ymax]")
    clip_geometry_url: Optional[str] = Field(None, description="GCS URL of clip geometry file")
    output_format: Optional[OutputFormat] = Field(None, description="Output format")
    output_bucket: Optional[str] = Field(None, description="GCS bucket for output")
    output_path: Optional[str] = Field(None, description="Path within bucket for output")
    
    @root_validator
    def validate_clip_source(cls, values):
        """Ensure at least one clip source is provided"""
        bounds = values.get('clip_bounds')
        geom_url = values.get('clip_geometry_url')
        
        if not bounds and not geom_url:
            raise ValueError('Either clip_bounds or clip_geometry_url must be provided')
        
        if bounds and geom_url:
            raise ValueError('Provide either clip_bounds or clip_geometry_url, not both')
        
        return values


class MergeRequest(BaseModel):
    """Merge multiple datasets request"""
    input_urls: List[str] = Field(..., min_items=2, description="List of GCS URLs to merge")
    output_format: OutputFormat = Field(..., description="Target output format")
    output_bucket: Optional[str] = Field(None, description="GCS bucket for output")
    output_path: Optional[str] = Field(None, description="Path within bucket for output")
    single_layer: bool = Field(True, description="Merge into single layer")
    options: Optional[ConversionOptions] = Field(None, description="Merge options")
    
    @validator('input_urls')
    def validate_input_urls(cls, v):
        """Validate all input URLs"""
        for url in v:
            if not url.startswith('gs://'):
                raise ValueError(f'All input URLs must be GCS URLs: {url}')
        return v


class DissolveRequest(BaseModel):
    """Dissolve features request"""
    input_url: str = Field(..., description="GCS URL of input file")
    dissolve_field: str = Field(..., description="Field name to dissolve by")
    output_format: Optional[OutputFormat] = Field(None, description="Output format")
    output_bucket: Optional[str] = Field(None, description="GCS bucket for output")
    output_path: Optional[str] = Field(None, description="Path within bucket for output")
    statistics: Optional[List[str]] = Field(None, description="Statistics to calculate (sum, mean, min, max, count)")
    statistics_fields: Optional[List[str]] = Field(None, description="Fields to calculate statistics on")


class SimplifyRequest(BaseModel):
    """Simplify geometry request"""
    input_url: str = Field(..., description="GCS URL of input file")
    tolerance: float = Field(..., gt=0, description="Simplification tolerance")
    preserve_topology: bool = Field(True, description="Preserve topology")
    output_format: Optional[OutputFormat] = Field(None, description="Output format")
    output_bucket: Optional[str] = Field(None, description="GCS bucket for output")
    output_path: Optional[str] = Field(None, description="Path within bucket for output")


# ==========================================
# Response Models
# ==========================================

class Job(BaseModel):
    """Job model for async processing"""
    id: str = Field(..., description="Unique job ID")
    status: JobStatus = Field(..., description="Current job status")
    input_format: Optional[str] = Field(None, description="Detected input format")
    output_format: Optional[str] = Field(None, description="Target output format")
    input_source: str = Field(..., description="Input source type (gcs/upload)")
    input_path: str = Field(..., description="Input file path or URL")
    output_path: Optional[str] = Field(None, description="Output file path or URL")
    error: Optional[str] = Field(None, description="Error message if failed")
    created_at: datetime = Field(..., description="Job creation timestamp")
    updated_at: Optional[datetime] = Field(None, description="Last update timestamp")
    completed_at: Optional[datetime] = Field(None, description="Completion timestamp")
    options: Optional[Dict[str, Any]] = Field(None, description="Processing options")
    progress: Optional[float] = Field(None, ge=0, le=100, description="Progress percentage")
    file_size: Optional[int] = Field(None, description="Input file size in bytes")
    feature_count: Optional[int] = Field(None, description="Number of features processed")
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class JobListResponse(BaseModel):
    """Response for job listing"""
    jobs: List[Job] = Field(..., description="List of jobs")
    count: int = Field(..., description="Total number of jobs")
    page: Optional[int] = Field(1, description="Current page")
    per_page: Optional[int] = Field(50, description="Items per page")
    total_pages: Optional[int] = Field(1, description="Total number of pages")


class FormatInfo(BaseModel):
    """Information about a supported format"""
    name: str = Field(..., description="Format display name")
    driver: str = Field(..., description="GDAL driver name")
    extensions: List[str] = Field(..., description="File extensions")
    capabilities: List[str] = Field(..., description="Driver capabilities (read/write/update)")
    vector_support: bool = Field(True, description="Supports vector data")
    creation_options: Optional[List[str]] = Field(None, description="Available creation options")


class LayerInfo(BaseModel):
    """Information about a vector layer"""
    name: str = Field(..., description="Layer name")
    feature_count: int = Field(..., description="Number of features")
    geometry_type: str = Field(..., description="Geometry type")
    extent: Optional[Dict[str, float]] = Field(None, description="Layer extent")
    srs: Optional[Dict[str, Any]] = Field(None, description="Spatial reference system")
    fields: List[Dict[str, Any]] = Field(..., description="Field definitions")


class FileInfoResponse(BaseModel):
    """Response with file information"""
    driver: str = Field(..., description="Detected driver/format")
    layer_count: int = Field(..., description="Number of layers")
    layers: List[LayerInfo] = Field(..., description="Layer information")
    size_bytes: Optional[int] = Field(None, description="File size in bytes")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata")


class HealthResponse(BaseModel):
    """Health check response"""
    status: str = Field(..., description="Service status")
    gdal_available: bool = Field(..., description="GDAL availability")
    gdal_version: str = Field(..., description="GDAL version")
    driver_count: int = Field(..., description="Number of available drivers")
    timestamp: datetime = Field(..., description="Response timestamp")
    environment: Optional[str] = Field(None, description="Environment name")
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class ErrorResponse(BaseModel):
    """Error response model"""
    error: str = Field(..., description="Error type")
    message: str = Field(..., description="Error message")
    details: Optional[Dict[str, Any]] = Field(None, description="Additional error details")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Error timestamp")
    request_id: Optional[str] = Field(None, description="Request ID for tracking")
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class SuccessResponse(BaseModel):
    """Generic success response"""
    success: bool = Field(True, description="Success flag")
    message: str = Field(..., description="Success message")
    data: Optional[Dict[str, Any]] = Field(None, description="Additional data")


# ==========================================
# Utility Models
# ==========================================

class BoundingBox(BaseModel):
    """Bounding box model"""
    xmin: float = Field(..., description="Minimum X coordinate")
    ymin: float = Field(..., description="Minimum Y coordinate")
    xmax: float = Field(..., description="Maximum X coordinate")
    ymax: float = Field(..., description="Maximum Y coordinate")
    
    @root_validator
    def validate_bounds(cls, values):
        """Validate bounding box coordinates"""
        xmin = values.get('xmin')
        xmax = values.get('xmax')
        ymin = values.get('ymin')
        ymax = values.get('ymax')
        
        if xmin >= xmax:
            raise ValueError('xmin must be less than xmax')
        if ymin >= ymax:
            raise ValueError('ymin must be less than ymax')
        
        return values
    
    def to_list(self) -> List[float]:
        """Convert to list format"""
        return [self.xmin, self.ymin, self.xmax, self.ymax]
    
    def to_wkt(self) -> str:
        """Convert to WKT polygon"""
        return f"POLYGON(({self.xmin} {self.ymin}, {self.xmax} {self.ymin}, {self.xmax} {self.ymax}, {self.xmin} {self.ymax}, {self.xmin} {self.ymin}))"


class SpatialReference(BaseModel):
    """Spatial reference system model"""
    epsg: Optional[int] = Field(None, description="EPSG code")
    proj4: Optional[str] = Field(None, description="PROJ4 string")
    wkt: Optional[str] = Field(None, description="Well-Known Text")
    name: Optional[str] = Field(None, description="SRS name")