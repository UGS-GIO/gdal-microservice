# app/main.py
import os
import asyncio
import subprocess
import shutil
import json
import csv
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, List, Any
from enum import Enum
import uuid
import zipfile

from fastapi import FastAPI, File, UploadFile, Form, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field
from google.cloud import storage
from contextlib import asynccontextmanager
import aiofiles
from osgeo import ogr, osr, gdal

# Enable GDAL exceptions
gdal.UseExceptions()

# Configuration
WORKSPACE_DIR = Path("/tmp/gdal-workspace")
UPLOAD_DIR = Path("/tmp/gdal-uploads")
OUTPUT_DIR = Path("/tmp/gdal-outputs")
MAX_FILE_SIZE = int(os.getenv("MAX_FILE_SIZE", 524288000))  # 500MB default
GCS_BUCKET = os.getenv("GCS_BUCKET", "")

# Ensure directories exist
for dir_path in [WORKSPACE_DIR, UPLOAD_DIR, OUTPUT_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# Initialize GCS client
try:
    gcs_client = storage.Client()
except Exception as e:
    print(f"Warning: Could not create GCS client: {e}")
    gcs_client = None


class JobStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class ConversionRequest(BaseModel):
    input_url: str = Field(..., description="GCS URL of input file")
    output_format: str = Field(..., description="Target output format")
    output_bucket: Optional[str] = Field(None, description="GCS bucket for output")
    output_path: Optional[str] = Field(None, description="GCS path for output")
    options: Optional[Dict[str, str]] = Field(default_factory=dict, description="GDAL conversion options")


class ReprojectionRequest(BaseModel):
    input_url: str = Field(..., description="GCS URL of input file")
    target_srs: str = Field(..., description="Target spatial reference system (EPSG code or PROJ string)")
    output_format: Optional[str] = Field(None, description="Output format (defaults to input format)")
    output_bucket: Optional[str] = Field(None, description="GCS bucket for output")
    output_path: Optional[str] = Field(None, description="GCS path for output")


class ClipRequest(BaseModel):
    input_url: str = Field(..., description="GCS URL of input file")
    clip_bounds: Optional[List[float]] = Field(None, description="Bounding box [xmin, ymin, xmax, ymax]")
    clip_geometry_url: Optional[str] = Field(None, description="GCS URL of clip geometry file")
    output_format: Optional[str] = Field(None, description="Output format")
    output_bucket: Optional[str] = Field(None, description="GCS bucket for output")
    output_path: Optional[str] = Field(None, description="GCS path for output")


class Job(BaseModel):
    id: str
    status: JobStatus
    input_format: Optional[str] = None
    output_format: Optional[str] = None
    input_source: str
    input_path: str
    output_path: Optional[str] = None
    error: Optional[str] = None
    created_at: datetime
    completed_at: Optional[datetime] = None
    options: Optional[Dict[str, str]] = None


class FormatInfo(BaseModel):
    name: str
    driver: str
    extensions: List[str]
    capabilities: List[str]


# In-memory job storage (replace with Redis/database in production)
jobs_store: Dict[str, Job] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    print(f"GDAL version: {gdal.VersionInfo()}")
    print(f"Available drivers: {gdal.GetDriverCount()}")
    yield
    # Shutdown
    print("Shutting down...")


app = FastAPI(
    title="GDAL API Service",
    description="Serverless GDAL processing API for vector data",
    version="1.0.0",
    lifespan=lifespan
)


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    try:
        # Check GDAL availability
        gdal_version = gdal.VersionInfo()
        driver_count = gdal.GetDriverCount()
        
        return {
            "status": "healthy",
            "gdal_available": True,
            "gdal_version": gdal_version,
            "driver_count": driver_count,
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        return JSONResponse(
            status_code=503,
            content={
                "status": "unhealthy",
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            }
        )


@app.get("/api/v1/formats", response_model=List[FormatInfo])
async def get_formats():
    """Get list of supported vector formats"""
    formats = [
        FormatInfo(
            name="ESRI Shapefile",
            driver="ESRI Shapefile",
            extensions=[".shp", ".dbf", ".shx", ".prj"],
            capabilities=["read", "write"]
        ),
        FormatInfo(
            name="GeoPackage",
            driver="GPKG",
            extensions=[".gpkg"],
            capabilities=["read", "write", "update"]
        ),
        FormatInfo(
            name="GeoJSON",
            driver="GeoJSON",
            extensions=[".geojson", ".json"],
            capabilities=["read", "write"]
        ),
        FormatInfo(
            name="CSV",
            driver="CSV",
            extensions=[".csv"],
            capabilities=["read", "write"]
        ),
        FormatInfo(
            name="File Geodatabase",
            driver="OpenFileGDB",
            extensions=[".gdb"],
            capabilities=["read"]
        ),
        FormatInfo(
            name="KML",
            driver="KML",
            extensions=[".kml", ".kmz"],
            capabilities=["read", "write"]
        ),
        FormatInfo(
            name="FlatGeobuf",
            driver="FlatGeobuf",
            extensions=[".fgb"],
            capabilities=["read", "write"]
        ),
        FormatInfo(
            name="Parquet",
            driver="Parquet",
            extensions=[".parquet"],
            capabilities=["read", "write"]
        ),
        FormatInfo(
            name="GML",
            driver="GML",
            extensions=[".gml"],
            capabilities=["read", "write"]
        ),
        FormatInfo(
            name="MapInfo File",
            driver="MapInfo File",
            extensions=[".tab", ".mif", ".mid"],
            capabilities=["read", "write"]
        )
    ]
    
    return formats


@app.post("/api/v1/convert", status_code=202)
async def convert_from_url(
    request: ConversionRequest,
    background_tasks: BackgroundTasks
):
    """Convert geospatial data from a GCS URL"""
    job_id = str(uuid.uuid4())
    
    job = Job(
        id=job_id,
        status=JobStatus.PENDING,
        output_format=request.output_format,
        input_source="gcs",
        input_path=request.input_url,
        created_at=datetime.utcnow(),
        options=request.options
    )
    
    if request.output_bucket and request.output_path:
        job.output_path = f"gs://{request.output_bucket}/{request.output_path}"
    
    jobs_store[job_id] = job
    
    # Process in background
    background_tasks.add_task(process_conversion, job_id)
    
    return job


@app.post("/api/v1/convert/upload", status_code=202)
async def convert_upload(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    output_format: str = Form(...),
    input_srid: Optional[str] = Form(None, description="Input SRID/EPSG code (e.g., 4326, 3857, 26912)"),
    output_srid: Optional[str] = Form(None, description="Output SRID/EPSG code for reprojection"),
    output_bucket: Optional[str] = Form(None),
    output_path: Optional[str] = Form(None),
    options: Optional[str] = Form(None)
):
    """Upload and convert a geospatial file with optional SRID specification"""
    # Validate file size
    if file.size and file.size > MAX_FILE_SIZE:
        raise HTTPException(400, f"File size exceeds maximum of {MAX_FILE_SIZE} bytes")
    
    # Parse options if provided
    conversion_options = {}
    if options:
        try:
            conversion_options = json.loads(options)
        except json.JSONDecodeError:
            raise HTTPException(400, "Invalid JSON in options parameter")
    
    # Add SRID options if provided
    if input_srid:
        # Normalize SRID format (accept both "4326" and "EPSG:4326")
        if not input_srid.upper().startswith("EPSG:"):
            input_srid = f"EPSG:{input_srid}"
        conversion_options["s_srs"] = input_srid
        print(f"DEBUG: Input SRID set to {input_srid}")
    
    if output_srid:
        # Normalize output SRID format
        if not output_srid.upper().startswith("EPSG:"):
            output_srid = f"EPSG:{output_srid}"
        conversion_options["t_srs"] = output_srid
        print(f"DEBUG: Output SRID set to {output_srid}")
    
    # Save uploaded file
    upload_id = str(uuid.uuid4())
    upload_path = UPLOAD_DIR / f"{upload_id}_{file.filename}"
    
    async with aiofiles.open(upload_path, 'wb') as f:
        content = await file.read()
        await f.write(content)
    
    # Create job
    job_id = str(uuid.uuid4())
    job = Job(
        id=job_id,
        status=JobStatus.PENDING,
        output_format=output_format,
        input_source="upload",
        input_path=str(upload_path),
        created_at=datetime.utcnow(),
        options=conversion_options
    )
    
    if output_bucket and output_path:
        job.output_path = f"gs://{output_bucket}/{output_path}"
    
    jobs_store[job_id] = job
    
    # Process in background
    background_tasks.add_task(process_conversion, job_id)
    
    return job


@app.get("/api/v1/jobs/{job_id}", response_model=Job)
async def get_job(job_id: str):
    """Get job status"""
    if job_id not in jobs_store:
        raise HTTPException(404, "Job not found")
    
    return jobs_store[job_id]


@app.get("/api/v1/jobs", response_model=Dict[str, Any])
async def list_jobs():
    """List all jobs"""
    return {
        "jobs": list(jobs_store.values()),
        "count": len(jobs_store)
    }


@app.get("/api/v1/jobs/{job_id}/download")
async def download_result(job_id: str):
    """Download conversion result"""
    # Check if job exists
    if job_id not in jobs_store:
        print(f"DEBUG: Job {job_id} not found. Available jobs: {list(jobs_store.keys())}")
        raise HTTPException(404, "Job not found")
    
    job = jobs_store[job_id]
    
    if job.status != JobStatus.COMPLETED:
        raise HTTPException(400, f"Job status is {job.status}, not completed")
    
    # For GCS files, redirect
    if job.output_path and job.output_path.startswith("gs://"):
        raise HTTPException(
            303,
            "File stored in GCS, use the output_path",
            headers={"Location": job.output_path}
        )
    
    # For local files, construct the path
    if job.output_path:
        output_file = Path(job.output_path)
        
        # If file doesn't exist at stored path, try standard locations
        if not output_file.exists():
            # Try with different extensions based on format
            possible_paths = [
                OUTPUT_DIR / f"{job_id}_shapefile.zip",  # Zipped shapefile
                OUTPUT_DIR / f"{job_id}_output.parquet",
                OUTPUT_DIR / f"{job_id}_output.gpkg",
                OUTPUT_DIR / f"{job_id}_output.shp",
                OUTPUT_DIR / f"{job_id}_output.geojson",
                OUTPUT_DIR / f"{job_id}_output.fgb",
                OUTPUT_DIR / f"{job_id}_output",
                Path(job.output_path)
            ]
            
            for path in possible_paths:
                if path.exists():
                    output_file = path
                    break
        
        print(f"DEBUG: Trying to serve file: {output_file}")
        print(f"DEBUG: File exists: {output_file.exists()}")
        if output_file.exists():
            print(f"DEBUG: File size: {output_file.stat().st_size} bytes")
        
        if output_file.exists():
            # Determine appropriate filename for download
            if job.output_format.lower() in ['shapefile', 'shp'] and output_file.suffix == '.zip':
                download_filename = f"{job_id}_shapefile.zip"
                media_type = "application/zip"
            else:
                download_filename = output_file.name
                media_type = "application/octet-stream"
            
            return FileResponse(
                path=str(output_file),
                media_type=media_type,
                filename=download_filename,
                headers={
                    "Content-Disposition": f"attachment; filename={download_filename}"
                }
            )
    
    # If we get here, file not found
    raise HTTPException(404, f"Output file not found. Expected at: {job.output_path}")


@app.post("/api/v1/operations/reproject", status_code=202)
async def reproject(
    request: ReprojectionRequest,
    background_tasks: BackgroundTasks
):
    """Reproject a dataset to a different coordinate system"""
    job_id = str(uuid.uuid4())
    
    options = {
        "t_srs": request.target_srs
    }
    
    job = Job(
        id=job_id,
        status=JobStatus.PENDING,
        output_format=request.output_format,
        input_source="gcs",
        input_path=request.input_url,
        created_at=datetime.utcnow(),
        options=options
    )
    
    if request.output_bucket and request.output_path:
        job.output_path = f"gs://{request.output_bucket}/{request.output_path}"
    
    jobs_store[job_id] = job
    
    background_tasks.add_task(process_conversion, job_id)
    
    return job


@app.post("/api/v1/operations/clip", status_code=202)
async def clip(
    request: ClipRequest,
    background_tasks: BackgroundTasks
):
    """Clip a dataset using bounds or another geometry"""
    job_id = str(uuid.uuid4())
    
    options = {}
    if request.clip_bounds:
        options["clipsrc"] = " ".join(map(str, request.clip_bounds))
    elif request.clip_geometry_url:
        # Download clip geometry first
        options["clipsrc"] = request.clip_geometry_url
    else:
        raise HTTPException(400, "Either clip_bounds or clip_geometry_url must be provided")
    
    job = Job(
        id=job_id,
        status=JobStatus.PENDING,
        output_format=request.output_format,
        input_source="gcs",
        input_path=request.input_url,
        created_at=datetime.utcnow(),
        options=options
    )
    
    if request.output_bucket and request.output_path:
        job.output_path = f"gs://{request.output_bucket}/{request.output_path}"
    
    jobs_store[job_id] = job
    
    background_tasks.add_task(process_conversion, job_id)
    
    return job


@app.post("/api/v1/info")
async def get_file_info(file: UploadFile = File(...)):
    """Get information about a geospatial file"""
    # Save uploaded file temporarily
    temp_path = WORKSPACE_DIR / f"info_{uuid.uuid4()}_{file.filename}"
    
    async with aiofiles.open(temp_path, 'wb') as f:
        content = await file.read()
        await f.write(content)
    
    try:
        # Handle .gdb.zip files
        if str(temp_path).endswith('.gdb.zip'):
            extract_dir = WORKSPACE_DIR / f"extract_{uuid.uuid4()}"
            extract_dir.mkdir(exist_ok=True)
            
            with zipfile.ZipFile(temp_path, 'r') as zip_ref:
                zip_ref.extractall(extract_dir)
            
            # Find the .gdb directory
            for item in extract_dir.iterdir():
                if item.suffix == '.gdb' and item.is_dir():
                    temp_path = item
                    break
        
        # Open with GDAL
        dataset = ogr.Open(str(temp_path))
        if not dataset:
            raise HTTPException(400, "Could not open file with GDAL")
        
        info = {
            "driver": dataset.GetDriver().GetName(),
            "layer_count": dataset.GetLayerCount(),
            "layers": []
        }
        
        for i in range(dataset.GetLayerCount()):
            layer = dataset.GetLayerByIndex(i)
            layer_info = {
                "name": layer.GetName(),
                "feature_count": layer.GetFeatureCount(),
                "geometry_type": ogr.GeometryTypeToName(layer.GetGeomType()),
                "extent": layer.GetExtent() if layer.GetFeatureCount() > 0 else None,
                "srs": None,
                "fields": []
            }
            
            # Get SRS info
            srs = layer.GetSpatialRef()
            if srs:
                layer_info["srs"] = {
                    "proj4": srs.ExportToProj4(),
                    "epsg": srs.GetAttrValue("AUTHORITY", 1) if srs.GetAttrValue("AUTHORITY", 0) == "EPSG" else None,
                    "wkt": srs.ExportToWkt()
                }
            
            # Get field info
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
        
        dataset = None  # Close dataset
        return info
        
    finally:
        # Clean up temp file
        if temp_path.exists():
            if temp_path.is_file():
                temp_path.unlink()
            elif temp_path.is_dir():
                shutil.rmtree(temp_path.parent)  # Clean up extract directory


# Background processing functions

async def process_conversion(job_id: str):
    """Process a conversion job with multi-layer support"""
    job = jobs_store[job_id]
    
    try:
        job.status = JobStatus.PROCESSING
        print(f"DEBUG: Processing job {job_id}")
        
        # Download input file if from GCS
        input_file = job.input_path
        if job.input_source == "gcs":
            input_file = await download_from_gcs(job.input_path)
        
        # Handle .gdb files specially
        if '.gdb' in str(input_file):
            # Extract if zipped
            if str(input_file).endswith('.zip'):
                extract_dir = Path(input_file).parent / f"extract_{job_id}"
                extract_dir.mkdir(exist_ok=True)
                
                print(f"DEBUG: Extracting {input_file} to {extract_dir}")
                with zipfile.ZipFile(input_file, 'r') as zip_ref:
                    zip_ref.extractall(extract_dir)
                
                # Find the .gdb
                for item in extract_dir.iterdir():
                    if item.suffix == '.gdb':
                        input_file = str(item)
                        print(f"DEBUG: Found .gdb at {input_file}")
                        break
        
        # Handle CSV files specially - create VRT for better geometry handling
        if str(input_file).lower().endswith('.csv'):
            print(f"DEBUG: Detected CSV input, creating VRT for geometry handling")
            
            # Get SRID from options or use default
            input_srid = "EPSG:4326"  # Default to WGS84
            if job.options and "s_srs" in job.options:
                input_srid = job.options["s_srs"]
                print(f"DEBUG: Using specified input SRID: {input_srid}")
            
            # Read CSV to detect geometry columns
            import csv
            with open(input_file, 'r') as f:
                reader = csv.DictReader(f)
                headers = reader.fieldnames
                first_row = next(reader, None)
            
            # Detect geometry columns
            x_field = None
            y_field = None
            wkt_field = None
            
            # Check for WKT field
            for field in headers:
                if field.upper() in ['WKT', 'GEOM', 'GEOMETRY']:
                    wkt_field = field
                    break
            
            # Check for X/Y fields if no WKT
            if not wkt_field:
                for field in headers:
                    if field.lower() in ['longitude', 'lon', 'lng', 'x', 'long', 'easting']:
                        x_field = field
                    elif field.lower() in ['latitude', 'lat', 'y', 'northing']:
                        y_field = field
            
            # Create VRT file for CSV
            vrt_file = Path(str(input_file).replace('.csv', '.vrt'))
            
            if wkt_field:
                # VRT for WKT geometry
                vrt_content = f'''<OGRVRTDataSource>
    <OGRVRTLayer name="{Path(input_file).stem}">
        <SrcDataSource>{input_file}</SrcDataSource>
        <GeometryType>wkbUnknown</GeometryType>
        <LayerSRS>{input_srid}</LayerSRS>
        <GeometryField encoding="WKT" field="{wkt_field}"/>
    </OGRVRTLayer>
</OGRVRTDataSource>'''
            elif x_field and y_field:
                # VRT for X/Y columns
                vrt_content = f'''<OGRVRTDataSource>
    <OGRVRTLayer name="{Path(input_file).stem}">
        <SrcDataSource>{input_file}</SrcDataSource>
        <GeometryType>wkbPoint</GeometryType>
        <LayerSRS>{input_srid}</LayerSRS>
        <GeometryField encoding="PointFromColumns" x="{x_field}" y="{y_field}"/>
    </OGRVRTLayer>
</OGRVRTDataSource>'''
            else:
                # No geometry columns found, try default
                print(f"DEBUG: No obvious geometry columns found, trying defaults")
                vrt_content = f'''<OGRVRTDataSource>
    <OGRVRTLayer name="{Path(input_file).stem}">
        <SrcDataSource>{input_file}</SrcDataSource>
        <GeometryType>wkbPoint</GeometryType>
        <LayerSRS>{input_srid}</LayerSRS>
        <GeometryField encoding="PointFromColumns" x="longitude" y="latitude"/>
    </OGRVRTLayer>
</OGRVRTDataSource>'''
            
            # Write VRT file
            with open(vrt_file, 'w') as f:
                f.write(vrt_content)
            
            print(f"DEBUG: Created VRT file: {vrt_file} with SRID: {input_srid}")
            input_file = str(vrt_file)
        
        # Determine output file path
        output_file = OUTPUT_DIR / f"{job_id}_output"
        output_file = add_format_extension(output_file, job.output_format)
        
        # Build ogr2ogr command with skipfailures for multi-layer sources
        cmd = [
            "ogr2ogr",
            "-f", get_driver_name(job.output_format),
            str(output_file),
            str(input_file),
            "-skipfailures"  # Skip layer creation failures
        ]
        
        # For Parquet output from multi-layer source, process first layer only
        if job.output_format.lower() in ['parquet', 'geoparquet']:
            # Get first layer name
            info_cmd = ["ogrinfo", "-q", str(input_file)]
            info_result = await asyncio.create_subprocess_exec(
                *info_cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            stdout, _ = await info_result.communicate()
            
            # Parse first layer name
            for line in stdout.decode().split('\n'):
                if ': ' in line and '(' in line:
                    layer_name = line.split(':')[1].split('(')[0].strip()
                    if layer_name and not layer_name.startswith('INFO'):
                        cmd.append(layer_name)  # Add specific layer
                        print(f"DEBUG: Using layer: {layer_name}")
                        break
        
        # Add options
        if job.options:
            # Check if we have s_srs but no t_srs or a_srs
            if "s_srs" in job.options and "t_srs" not in job.options and "a_srs" not in job.options:
                # Use a_srs (assign SRS) instead of s_srs when not transforming
                job.options["a_srs"] = job.options["s_srs"]
                del job.options["s_srs"]
                print(f"DEBUG: Using -a_srs instead of -s_srs since no transformation requested")
            
            for key, value in job.options.items():
                if key == "clipsrc" and value.startswith("gs://"):
                    # Download clip file
                    clip_file = await download_from_gcs(value)
                    cmd.extend(["-clipsrc", str(clip_file)])
                elif key in ["s_srs", "t_srs", "a_srs"]:
                    # Handle SRID options
                    cmd.extend([f"-{key}", value])
                else:
                    cmd.extend([f"-{key}", value])
        
        print(f"DEBUG: Running command: {' '.join(cmd)}")
        
        # Execute conversion
        result = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        
        stdout, stderr = await result.communicate()
        
        print(f"DEBUG: Command stdout: {stdout.decode()}")
        if stderr:
            print(f"DEBUG: Command stderr: {stderr.decode()}")
        
        if result.returncode != 0:
            raise Exception(f"ogr2ogr failed: {stderr.decode()}")
        
        # Verify output file exists
        if not output_file.exists():
            raise Exception(f"Output file was not created: {output_file}")
        
        print(f"DEBUG: Output file created: {output_file}, size: {output_file.stat().st_size} bytes")
        
        # Handle shapefile output - create a zip with all components
        if job.output_format.lower() in ['shapefile', 'shp']:
            zip_path = OUTPUT_DIR / f"{job_id}_shapefile.zip"
            
            # Find all related shapefile files
            base_name = output_file.stem
            shapefile_extensions = ['.shp', '.dbf', '.shx', '.prj', '.cpg', '.qpj', '.shp.xml']
            
            with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                files_added = 0
                for ext in shapefile_extensions:
                    component_file = output_file.parent / f"{base_name}{ext}"
                    if component_file.exists():
                        zipf.write(component_file, component_file.name)
                        print(f"DEBUG: Added {component_file.name} to zip")
                        files_added += 1
                
                if files_added == 0:
                    raise Exception("No shapefile components found to zip")
            
            print(f"DEBUG: Created shapefile zip: {zip_path} with {files_added} files")
            
            # Upload to GCS if specified
            if job.output_path and job.output_path.startswith("gs://"):
                await upload_to_gcs(zip_path, job.output_path)
                # Remove local files after upload
                zip_path.unlink()
                for ext in shapefile_extensions:
                    component_file = output_file.parent / f"{base_name}{ext}"
                    if component_file.exists():
                        component_file.unlink()
            else:
                job.output_path = str(zip_path)
        else:
            # For non-shapefile formats, handle normally
            if job.output_path and job.output_path.startswith("gs://"):
                await upload_to_gcs(output_file, job.output_path)
                # Remove local file after upload
                output_file.unlink()
            else:
                job.output_path = str(output_file)
        
        # Update job status
        job.status = JobStatus.COMPLETED
        job.completed_at = datetime.utcnow()
        print(f"DEBUG: Job {job_id} completed successfully")
        
    except Exception as e:
        print(f"DEBUG: Job {job_id} failed with error: {str(e)}")
        job.status = JobStatus.FAILED
        job.error = str(e)
        job.completed_at = datetime.utcnow()
    
    finally:
        # Clean up input file if downloaded
        if job.input_source == "gcs" and Path(input_file).exists():
            Path(input_file).unlink()


async def download_from_gcs(gcs_url: str) -> Path:
    """Download file from GCS"""
    if not gcs_url.startswith("gs://"):
        raise ValueError("Invalid GCS URL")
    
    # Parse GCS URL
    parts = gcs_url[5:].split("/", 1)
    if len(parts) != 2:
        raise ValueError("Invalid GCS URL format")
    
    bucket_name, object_name = parts
    
    # Download file
    local_path = WORKSPACE_DIR / f"{uuid.uuid4()}_{Path(object_name).name}"
    
    bucket = gcs_client.bucket(bucket_name)
    blob = bucket.blob(object_name)
    blob.download_to_filename(str(local_path))
    
    return local_path


async def upload_to_gcs(local_path: Path, gcs_url: str):
    """Upload file to GCS"""
    # Parse GCS URL
    parts = gcs_url[5:].split("/", 1)
    if len(parts) != 2:
        raise ValueError("Invalid GCS URL format")
    
    bucket_name, object_name = parts
    
    # Upload file
    bucket = gcs_client.bucket(bucket_name)
    blob = bucket.blob(object_name)
    blob.upload_from_filename(str(local_path))


def get_driver_name(format_name: str) -> str:
    """Get GDAL driver name from format name"""
    driver_map = {
        "shapefile": "ESRI Shapefile",
        "shp": "ESRI Shapefile",
        "gpkg": "GPKG",
        "geopackage": "GPKG",
        "geojson": "GeoJSON",
        "json": "GeoJSON",
        "csv": "CSV",
        "kml": "KML",
        "gdb": "OpenFileGDB",
        "fgb": "FlatGeobuf",
        "flatgeobuf": "FlatGeobuf",
        "parquet": "Parquet",
        "geoparquet": "Parquet",
        "gml": "GML",
        "tab": "MapInfo File",
        "mif": "MapInfo File"
    }
    
    return driver_map.get(format_name.lower(), format_name)


def add_format_extension(path: Path, format_name: str) -> Path:
    """Add appropriate extension based on format"""
    ext_map = {
        "shapefile": ".shp",
        "shp": ".shp",
        "gpkg": ".gpkg",
        "geopackage": ".gpkg",
        "geojson": ".geojson",
        "json": ".json",
        "csv": ".csv",
        "kml": ".kml",
        "gdb": ".gdb",
        "fgb": ".fgb",
        "flatgeobuf": ".fgb",
        "parquet": ".parquet",
        "geoparquet": ".parquet",
        "gml": ".gml",
        "tab": ".tab",
        "mif": ".mif"
    }
    
    ext = ext_map.get(format_name.lower(), "")
    return path.with_suffix(ext)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)