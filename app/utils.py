"""Utility functions for GDAL operations and file handling"""

import hashlib
import os
import re
import json
import tempfile
import shutil
import asyncio
import subprocess
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple, Union
from datetime import datetime, timedelta
import logging
import uuid

from osgeo import ogr, osr, gdal
from google.cloud import storage

# Configure GDAL
gdal.UseExceptions()
ogr.UseExceptions()

logger = logging.getLogger(__name__)


# ==========================================
# File Operations
# ==========================================

def get_file_hash(filepath: Path, algorithm: str = "sha256") -> str:
    """
    Calculate hash of a file.
    
    Args:
        filepath: Path to file
        algorithm: Hash algorithm (sha256, md5, sha1)
    
    Returns:
        Hex digest of file hash
    """
    hash_func = getattr(hashlib, algorithm)()
    
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            hash_func.update(chunk)
    
    return hash_func.hexdigest()


def get_file_size_mb(filepath: Path) -> float:
    """Get file size in megabytes"""
    if not filepath.exists():
        return 0.0
    return filepath.stat().st_size / (1024 * 1024)


def validate_file_extension(filename: str, allowed_extensions: List[str]) -> bool:
    """
    Check if file has an allowed extension.
    
    Args:
        filename: Name of the file
        allowed_extensions: List of allowed extensions (with dots)
    
    Returns:
        True if extension is allowed
    """
    return any(filename.lower().endswith(ext.lower()) for ext in allowed_extensions)


def generate_unique_filename(original_filename: str, prefix: str = "") -> str:
    """
    Generate a unique filename with UUID.
    
    Args:
        original_filename: Original file name
        prefix: Optional prefix for the filename
    
    Returns:
        Unique filename
    """
    name, ext = os.path.splitext(original_filename)
    unique_id = str(uuid.uuid4())[:8]
    
    if prefix:
        return f"{prefix}_{unique_id}_{name}{ext}"
    return f"{unique_id}_{name}{ext}"


def cleanup_old_files(directory: Path, hours: int = 24) -> int:
    """
    Clean up files older than specified hours.
    
    Args:
        directory: Directory to clean
        hours: Age threshold in hours
    
    Returns:
        Number of files deleted
    """
    if not directory.exists():
        return 0
    
    current_time = datetime.now()
    deleted_count = 0
    
    for filepath in directory.glob("*"):
        if filepath.is_file():
            file_age = current_time - datetime.fromtimestamp(filepath.stat().st_mtime)
            if file_age > timedelta(hours=hours):
                try:
                    filepath.unlink()
                    deleted_count += 1
                    logger.debug(f"Deleted old file: {filepath}")
                except Exception as e:
                    logger.error(f"Failed to delete {filepath}: {e}")
    
    return deleted_count


# ==========================================
# GDAL/OGR Operations
# ==========================================

def validate_vector_file(filepath: Path) -> Tuple[bool, Optional[str]]:
    """
    Validate that a file can be opened as vector data.
    
    Args:
        filepath: Path to vector file
    
    Returns:
        Tuple of (is_valid, error_message)
    """
    try:
        dataset = ogr.Open(str(filepath))
        if dataset is None:
            return False, "File cannot be opened by GDAL/OGR"
        
        layer_count = dataset.GetLayerCount()
        if layer_count == 0:
            return False, "File contains no vector layers"
        
        # Check if we can read at least one feature
        layer = dataset.GetLayerByIndex(0)
        feature_count = layer.GetFeatureCount()
        
        dataset = None  # Close dataset
        return True, None
        
    except Exception as e:
        return False, str(e)


def get_driver_for_format(format_name: str) -> str:
    """
    Get GDAL driver name for a format.
    
    Args:
        format_name: Format name or extension
    
    Returns:
        GDAL driver name
    """
    driver_map = {
        "shapefile": "ESRI Shapefile",
        "shp": "ESRI Shapefile",
        "gpkg": "GPKG",
        "geopackage": "GPKG",
        "geojson": "GeoJSON",
        "json": "GeoJSON",
        "csv": "CSV",
        "kml": "KML",
        "kmz": "LIBKML",
        "gdb": "OpenFileGDB",
        "fgb": "FlatGeobuf",
        "flatgeobuf": "FlatGeobuf",
        "parquet": "Parquet",
        "gml": "GML",
        "tab": "MapInfo File",
        "mif": "MapInfo File",
        "mid": "MapInfo File"
    }
    
    format_lower = format_name.lower()
    return driver_map.get(format_lower, format_name)


def add_format_extension(filepath: Path, format_name: str) -> Path:
    """
    Add appropriate extension based on format.
    
    Args:
        filepath: Base filepath
        format_name: Output format name
    
    Returns:
        Path with appropriate extension
    """
    ext_map = {
        "shapefile": ".shp",
        "shp": ".shp",
        "gpkg": ".gpkg",
        "geopackage": ".gpkg",
        "geojson": ".geojson",
        "json": ".json",
        "csv": ".csv",
        "kml": ".kml",
        "kmz": ".kmz",
        "gdb": ".gdb",
        "fgb": ".fgb",
        "flatgeobuf": ".fgb",
        "parquet": ".parquet",
        "gml": ".gml",
        "tab": ".tab",
        "mif": ".mif"
    }
    
    ext = ext_map.get(format_name.lower(), "")
    if ext:
        return filepath.with_suffix(ext)
    return filepath


def get_dataset_info(filepath: Path) -> Dict[str, Any]:
    """
    Get detailed information about a vector dataset.
    
    Args:
        filepath: Path to vector file
    
    Returns:
        Dictionary with dataset information
    """
    dataset = ogr.Open(str(filepath))
    if not dataset:
        raise ValueError(f"Cannot open file: {filepath}")
    
    try:
        info = {
            "driver": dataset.GetDriver().GetName(),
            "layer_count": dataset.GetLayerCount(),
            "layers": [],
            "size_bytes": filepath.stat().st_size if filepath.exists() else None,
            "format_name": dataset.GetDriver().GetDescription()
        }
        
        for i in range(dataset.GetLayerCount()):
            layer = dataset.GetLayerByIndex(i)
            layer_info = get_layer_info(layer, i)
            info["layers"].append(layer_info)
        
        return info
        
    finally:
        dataset = None  # Close dataset


def get_layer_info(layer: ogr.Layer, index: int = 0) -> Dict[str, Any]:
    """
    Get information about a specific layer.
    
    Args:
        layer: OGR Layer object
        index: Layer index
    
    Returns:
        Dictionary with layer information
    """
    layer_info = {
        "index": index,
        "name": layer.GetName(),
        "feature_count": layer.GetFeatureCount(),
        "geometry_type": ogr.GeometryTypeToName(layer.GetGeomType()),
        "fields": [],
        "extent": None,
        "srs": None
    }
    
    # Get extent
    try:
        extent = layer.GetExtent()
        if extent:
            layer_info["extent"] = {
                "xmin": extent[0],
                "xmax": extent[1],
                "ymin": extent[2],
                "ymax": extent[3]
            }
    except Exception as e:
        logger.warning(f"Could not get extent: {e}")
    
    # Get SRS
    srs = layer.GetSpatialRef()
    if srs:
        layer_info["srs"] = get_srs_info(srs)
    
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
    
    return layer_info


def get_srs_info(srs: osr.SpatialReference) -> Dict[str, Any]:
    """
    Get information about a spatial reference system.
    
    Args:
        srs: OSR SpatialReference object
    
    Returns:
        Dictionary with SRS information
    """
    srs_info = {
        "proj4": None,
        "wkt": None,
        "epsg": None,
        "name": None
    }
    
    try:
        srs_info["proj4"] = srs.ExportToProj4()
        srs_info["wkt"] = srs.ExportToWkt()
        
        # Try to get EPSG code
        if srs.GetAttrValue("AUTHORITY", 0) == "EPSG":
            srs_info["epsg"] = int(srs.GetAttrValue("AUTHORITY", 1))
        
        # Try to get name
        srs_info["name"] = srs.GetAttrValue("PROJCS") or srs.GetAttrValue("GEOGCS")
        
    except Exception as e:
        logger.warning(f"Error extracting SRS info: {e}")
    
    return srs_info


def estimate_processing_time(file_size: int, operation: str = "convert") -> float:
    """
    Estimate processing time based on file size and operation.
    
    Args:
        file_size: File size in bytes
        operation: Type of operation
    
    Returns:
        Estimated time in seconds
    """
    # Base rate: 10MB/second for simple operations
    base_rate_mbps = 10
    
    # Operation complexity multipliers
    operation_multipliers = {
        "convert": 1.0,
        "reproject": 1.5,
        "clip": 1.2,
        "merge": 2.0,
        "dissolve": 3.0,
        "simplify": 1.8,
        "buffer": 2.5
    }
    
    multiplier = operation_multipliers.get(operation.lower(), 1.0)
    file_size_mb = file_size / (1024 * 1024)
    
    # Calculate time with multiplier
    estimated_seconds = (file_size_mb / base_rate_mbps) * multiplier
    
    # Add base overhead (5 seconds)
    return estimated_seconds + 5.0


# ==========================================
# GCS Operations
# ==========================================

async def download_from_gcs(gcs_url: str, local_dir: Path) -> Path:
    """
    Download file from Google Cloud Storage.
    
    Args:
        gcs_url: GCS URL (gs://bucket/path)
        local_dir: Local directory to save file
    
    Returns:
        Path to downloaded file
    """
    if not gcs_url.startswith("gs://"):
        raise ValueError("Invalid GCS URL format")
    
    # Parse GCS URL
    match = re.match(r"gs://([^/]+)/(.+)", gcs_url)
    if not match:
        raise ValueError("Invalid GCS URL format")
    
    bucket_name, object_name = match.groups()
    
    # Generate local filename
    local_filename = generate_unique_filename(Path(object_name).name)
    local_path = local_dir / local_filename
    
    try:
        # Download using gsutil for better performance
        cmd = ["gsutil", "cp", gcs_url, str(local_path)]
        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        stdout, stderr = await process.communicate()
        
        if process.returncode != 0:
            raise Exception(f"gsutil failed: {stderr.decode()}")
        
        logger.info(f"Downloaded {gcs_url} to {local_path}")
        return local_path
        
    except Exception as e:
        # Fallback to Python client
        logger.warning(f"gsutil failed, using Python client: {e}")
        
        client = storage.Client()
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(object_name)
        blob.download_to_filename(str(local_path))
        
        return local_path


async def upload_to_gcs(local_path: Path, gcs_url: str) -> str:
    """
    Upload file to Google Cloud Storage.
    
    Args:
        local_path: Local file path
        gcs_url: Target GCS URL (gs://bucket/path)
    
    Returns:
        GCS URL of uploaded file
    """
    if not local_path.exists():
        raise FileNotFoundError(f"File not found: {local_path}")
    
    # Parse GCS URL
    match = re.match(r"gs://([^/]+)/(.+)", gcs_url)
    if not match:
        raise ValueError("Invalid GCS URL format")
    
    bucket_name, object_name = match.groups()
    
    try:
        # Upload using gsutil for better performance
        cmd = ["gsutil", "cp", str(local_path), gcs_url]
        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        stdout, stderr = await process.communicate()
        
        if process.returncode != 0:
            raise Exception(f"gsutil failed: {stderr.decode()}")
        
        logger.info(f"Uploaded {local_path} to {gcs_url}")
        return gcs_url
        
    except Exception as e:
        # Fallback to Python client
        logger.warning(f"gsutil failed, using Python client: {e}")
        
        client = storage.Client()
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(object_name)
        blob.upload_from_filename(str(local_path))
        
        return gcs_url


def parse_gcs_url(gcs_url: str) -> Tuple[str, str]:
    """
    Parse GCS URL into bucket and object path.
    
    Args:
        gcs_url: GCS URL (gs://bucket/path)
    
    Returns:
        Tuple of (bucket_name, object_path)
    """
    if not gcs_url.startswith("gs://"):
        raise ValueError("URL must start with gs://")
    
    match = re.match(r"gs://([^/]+)/(.+)", gcs_url)
    if not match:
        raise ValueError("Invalid GCS URL format")
    
    return match.groups()


# ==========================================
# Conversion Helpers
# ==========================================

def build_ogr2ogr_command(
    input_file: str,
    output_file: str,
    output_format: str,
    options: Optional[Dict[str, Any]] = None
) -> List[str]:
    """
    Build ogr2ogr command with options.
    
    Args:
        input_file: Input file path
        output_file: Output file path
        output_format: Output format driver name
        options: Additional options
    
    Returns:
        Command as list of strings
    """
    cmd = [
        "ogr2ogr",
        "-f", output_format,
        output_file,
        input_file
    ]
    
    if options:
        # Target SRS
        if "t_srs" in options:
            cmd.extend(["-t_srs", options["t_srs"]])
        
        # Source SRS
        if "s_srs" in options:
            cmd.extend(["-s_srs", options["s_srs"]])
        
        # SQL query
        if "sql" in options:
            cmd.extend(["-sql", options["sql"]])
        elif "where" in options:
            cmd.extend(["-where", options["where"]])
        
        # Field selection
        if "select" in options:
            cmd.extend(["-select", options["select"]])
        
        # Clipping
        if "clipsrc" in options:
            clip = options["clipsrc"]
            if isinstance(clip, list):
                cmd.extend(["-clipsrc"] + [str(x) for x in clip])
            else:
                cmd.extend(["-clipsrc", str(clip)])
        
        # Simplification
        if "simplify" in options:
            cmd.extend(["-simplify", str(options["simplify"])])
        
        # Skip failures
        if options.get("skipfailures"):
            cmd.append("-skipfailures")
        
        # Limit features
        if "limit" in options:
            cmd.extend(["-limit", str(options["limit"])])
        
        # Layer creation options
        if "lco" in options:
            for key, value in options["lco"].items():
                cmd.extend(["-lco", f"{key}={value}"])
        
        # Dataset creation options
        if "dsco" in options:
            for key, value in options["dsco"].items():
                cmd.extend(["-dsco", f"{key}={value}"])
    
    return cmd


async def run_gdal_command(cmd: List[str], timeout: int = 3600) -> Tuple[bool, str]:
    """
    Run a GDAL command asynchronously.
    
    Args:
        cmd: Command as list of strings
        timeout: Timeout in seconds
    
    Returns:
        Tuple of (success, output/error message)
    """
    try:
        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        
        try:
            stdout, stderr = await asyncio.wait_for(
                process.communicate(),
                timeout=timeout
            )
        except asyncio.TimeoutError:
            process.kill()
            return False, f"Command timed out after {timeout} seconds"
        
        if process.returncode != 0:
            error_msg = stderr.decode() if stderr else stdout.decode()
            return False, error_msg
        
        return True, stdout.decode()
        
    except Exception as e:
        return False, str(e)


# ==========================================
# Validation Helpers
# ==========================================

def validate_epsg_code(epsg_code: Union[str, int]) -> bool:
    """
    Validate EPSG code.
    
    Args:
        epsg_code: EPSG code as string or integer
    
    Returns:
        True if valid EPSG code
    """
    try:
        if isinstance(epsg_code, str):
            if epsg_code.upper().startswith("EPSG:"):
                epsg_code = epsg_code.split(":")[1]
            epsg_code = int(epsg_code)
        
        # Check if code is in valid range
        if epsg_code < 1 or epsg_code > 100000:
            return False
        
        # Try to create SRS with the code
        srs = osr.SpatialReference()
        result = srs.ImportFromEPSG(epsg_code)
        
        return result == 0
        
    except Exception:
        return False


def validate_bbox(bbox: List[float]) -> bool:
    """
    Validate bounding box coordinates.
    
    Args:
        bbox: List of [xmin, ymin, xmax, ymax]
    
    Returns:
        True if valid bounding box
    """
    if len(bbox) != 4:
        return False
    
    xmin, ymin, xmax, ymax = bbox
    
    # Check that min < max
    if xmin >= xmax or ymin >= ymax:
        return False
    
    # Check for reasonable coordinate ranges
    if abs(xmin) > 180 or abs(xmax) > 180:
        # Might be projected coordinates
        if abs(xmin) > 20000000 or abs(xmax) > 20000000:
            return False
    
    if abs(ymin) > 90 or abs(ymax) > 90:
        # Might be projected coordinates
        if abs(ymin) > 20000000 or abs(ymax) > 20000000:
            return False
    
    return True


# ==========================================
# Format Detection
# ==========================================

def detect_vector_format(filepath: Path) -> Optional[str]:
    """
    Detect vector format from file.
    
    Args:
        filepath: Path to file
    
    Returns:
        Detected format name or None
    """
    try:
        dataset = ogr.Open(str(filepath))
        if dataset:
            driver_name = dataset.GetDriver().GetName()
            dataset = None
            return driver_name
    except Exception:
        pass
    
    # Fallback to extension-based detection
    ext = filepath.suffix.lower()
    ext_format_map = {
        ".shp": "ESRI Shapefile",
        ".gpkg": "GPKG",
        ".geojson": "GeoJSON",
        ".json": "GeoJSON",
        ".kml": "KML",
        ".kmz": "LIBKML",
        ".gml": "GML",
        ".csv": "CSV",
        ".fgb": "FlatGeobuf",
        ".parquet": "Parquet"
    }
    
    return ext_format_map.get(ext)