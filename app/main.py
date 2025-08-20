# Enhanced main.py with GCS integration
"""
Enhanced GDAL Microservice with Google Cloud Storage integration
"""

import os
import asyncio
import shutil
import json
import uuid
import tempfile
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, List, Any
from enum import Enum

from fastapi import FastAPI, File, UploadFile, Form, HTTPException, BackgroundTasks, Body, Query
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator
from contextlib import asynccontextmanager
import aiofiles

from .gcs_handler import GCSHandler

# Configuration
WORKSPACE_DIR = Path("/tmp/gdal-workspace")
UPLOAD_DIR = Path("/tmp/gdal-uploads")
OUTPUT_DIR = Path("/tmp/gdal-outputs")
MAX_FILE_SIZE = int(os.getenv("MAX_FILE_SIZE", 524288000))  # 500MB default
COMMAND_TIMEOUT = int(os.getenv("COMMAND_TIMEOUT", 3600))  # 1 hour default
GCS_BUCKET = os.getenv("GCS_BUCKET")

# Ensure directories exist
for dir_path in [WORKSPACE_DIR, UPLOAD_DIR, OUTPUT_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# Initialize GCS handler
gcs_handler = GCSHandler(default_bucket=GCS_BUCKET)

# Allowed GDAL commands (same as before)
ALLOWED_COMMANDS = {
    "ogr2ogr": "Vector data conversion and processing",
    "ogrinfo": "Vector data information",
    "gdal_translate": "Raster data conversion",
    "gdalinfo": "Raster data information",
    "gdalwarp": "Raster reprojection and warping",
    "gdal_rasterize": "Vector to raster conversion",
    "gdal_polygonize": "Raster to vector conversion",
    "gdal_contour": "Contour generation from raster",
    "gdaldem": "DEM processing (hillshade, slope, etc.)",
    "gdalbuildvrt": "Build virtual raster",
    "ogrmerge": "Merge vector files",
    "gdal_merge": "Merge raster files"
}


class JobStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class GCSCommand(BaseModel):
    """Model for GDAL command execution with GCS support"""
    command: str = Field(..., description="GDAL command (e.g., ogr2ogr, gdalwarp)")
    args: List[str] = Field(..., description="Command arguments")
    input_gcs_urls: Optional[List[str]] = Field(None, description="List of GCS URLs for input files")
    output_gcs_url: Optional[str] = Field(None, description="GCS URL for output file")
    use_signed_urls: bool = Field(True, description="Use signed URLs for file access")
    
    @validator('input_gcs_urls')
    def validate_gcs_urls(cls, v):
        if v:
            for url in v:
                if not url.startswith('gs://'):
                    raise ValueError(f'Invalid GCS URL: {url}')
        return v
    
    @validator('output_gcs_url')
    def validate_output_gcs_url(cls, v):
        if v and not v.startswith('gs://'):
            raise ValueError(f'Invalid GCS URL: {v}')
        return v


class GCSUploadRequest(BaseModel):
    """Request for generating signed upload URLs"""
    filename: str = Field(..., description="Target filename")
    bucket: Optional[str] = Field(None, description="Target bucket (optional)")
    path: Optional[str] = Field("uploads", description="Path within bucket")
    content_type: Optional[str] = Field(None, description="File content type")


class CommandJob(BaseModel):
    """Enhanced job tracking with GCS support"""
    id: str
    status: JobStatus
    command: str
    args: List[str]
    input_gcs_urls: Optional[List[str]] = None
    output_gcs_url: Optional[str] = None
    output_signed_url: Optional[str] = None
    error: Optional[str] = None
    stdout: Optional[str] = None
    stderr: Optional[str] = None
    created_at: datetime
    completed_at: Optional[datetime] = None
    execution_time_seconds: Optional[float] = None


# In-memory job storage (use Redis in production)
jobs_store: Dict[str, CommandJob] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Cleanup old files on startup and shutdown"""
    print(f"🚀 Enhanced GDAL Microservice with GCS integration starting...")
    print(f"GCS Bucket: {GCS_BUCKET}")
    print(f"Available commands: {', '.join(ALLOWED_COMMANDS.keys())}")
    
    cleanup_old_files()
    yield
    
    print("Shutting down...")
    cleanup_old_files()


app = FastAPI(
    title="Enhanced GDAL Microservice",
    description="HTTP wrapper for GDAL commands with Google Cloud Storage integration",
    version="2.1.0",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    """Root endpoint with service info"""
    return {
        "service": "Enhanced GDAL Microservice",
        "version": "2.1.0",
        "description": "Execute GDAL commands via HTTP with GCS integration",
        "gcs_enabled": gcs_handler.client is not None,
        "default_bucket": GCS_BUCKET,
        "endpoints": {
            "/health": "Health check",
            "/commands": "List available commands",
            "/gcs/files": "List GCS files",
            "/gcs/upload-url": "Generate signed upload URL",
            "/gcs/execute": "Execute GDAL command on GCS files",
            "/gcs/execute-async": "Execute GDAL command on GCS files (async)",
            "/execute": "Execute GDAL command synchronously (local files)",
            "/execute-async": "Execute GDAL command asynchronously (local files)",
            "/upload-and-execute": "Upload file and execute command",
            "/jobs/{job_id}": "Get job status",
            "/jobs/{job_id}/download": "Download job result"
        }
    }


@app.get("/gcs/files")
async def list_gcs_files(
    bucket: Optional[str] = Query(None, description="Bucket name"),
    prefix: Optional[str] = Query("", description="File prefix filter")
):
    """List files in GCS bucket"""
    try:
        files = gcs_handler.list_files(bucket, prefix)
        return {
            "bucket": bucket or GCS_BUCKET,
            "prefix": prefix,
            "files": files,
            "count": len(files)
        }
    except Exception as e:
        raise HTTPException(500, f"Failed to list files: {str(e)}")


@app.post("/gcs/upload-url")
async def generate_upload_url(request: GCSUploadRequest):
    """Generate signed URL for file upload to GCS"""
    try:
        bucket_name = request.bucket or GCS_BUCKET
        if not bucket_name:
            raise HTTPException(400, "No bucket specified")
        
        # Create object path
        object_name = f"{request.path.strip('/')}/{request.filename}"
        
        # Generate signed URL for PUT
        signed_url = gcs_handler.generate_signed_url(
            bucket_name, 
            object_name, 
            expiration=3600, 
            method="PUT"
        )
        
        gcs_url = f"gs://{bucket_name}/{object_name}"
        
        return {
            "upload_url": signed_url,
            "gcs_url": gcs_url,
            "method": "PUT",
            "expires_in": 3600,
            "instructions": {
                "method": "PUT",
                "headers": {
                    "Content-Type": request.content_type or "application/octet-stream"
                },
                "body": "Raw file data"
            }
        }
        
    except Exception as e:
        raise HTTPException(500, f"Failed to generate upload URL: {str(e)}")


@app.post("/gcs/execute")
async def execute_gcs_command(request: GCSCommand):
    """
    Execute GDAL command on files in Google Cloud Storage
    
    Example request:
    {
        "command": "ogr2ogr",
        "args": ["-f", "GeoJSON", "{output}", "{input}"],
        "input_gcs_urls": ["gs://bucket/input.shp"],
        "output_gcs_url": "gs://bucket/output.geojson"
    }
    """
    if request.command not in ALLOWED_COMMANDS:
        raise HTTPException(400, f"Command '{request.command}' not allowed")
    
    work_id = str(uuid.uuid4())
    work_dir = WORKSPACE_DIR / work_id
    work_dir.mkdir(exist_ok=True)
    
    try:
        # Download input files from GCS
        local_inputs = []
        if request.input_gcs_urls:
            for gcs_url in request.input_gcs_urls:
                local_path = await gcs_handler.download_file(gcs_url, work_dir)
                local_inputs.append(str(local_path))
        
        # Prepare output path
        output_filename = f"output_{work_id}"
        if request.output_gcs_url:
            output_filename = Path(request.output_gcs_url).name
        
        local_output = work_dir / output_filename
        
        # Replace placeholders in args
        prepared_args = []
        input_index = 0
        
        for arg in request.args:
            if arg == "{input}":
                if input_index < len(local_inputs):
                    prepared_args.append(local_inputs[input_index])
                    input_index += 1
                else:
                    raise HTTPException(400, f"Not enough input files for {arg}")
            elif arg == "{output}":
                prepared_args.append(str(local_output))
            elif arg.startswith("{input"):
                # Handle {input0}, {input1}, etc.
                try:
                    idx = int(arg[6:-1])  # Extract number from {inputN}
                    if idx < len(local_inputs):
                        prepared_args.append(local_inputs[idx])
                    else:
                        raise HTTPException(400, f"Input file index {idx} not available")
                except:
                    prepared_args.append(arg)
            else:
                prepared_args.append(arg)
        
        # Execute GDAL command
        cmd = [request.command] + prepared_args
        result = await run_command(cmd, cwd=work_dir)
        
        # Upload output to GCS if specified
        output_gcs_url = None
        download_url = None
        
        if local_output.exists():
            if request.output_gcs_url:
                output_gcs_url = await gcs_handler.upload_file(local_output, request.output_gcs_url)
                
                if request.use_signed_urls:
                    bucket_name, object_name = gcs_handler.parse_gcs_url(request.output_gcs_url)
                    download_url = gcs_handler.generate_signed_url(bucket_name, object_name)
            else:
                # Move to local output directory for download
                final_path = OUTPUT_DIR / f"{work_id}_{output_filename}"
                shutil.move(str(local_output), str(final_path))
                download_url = f"/download/{work_id}/{output_filename}"
        
        return {
            "success": result["success"],
            "command": request.command,
            "args": prepared_args,
            "stdout": result.get("stdout"),
            "stderr": result.get("stderr"),
            "output_gcs_url": output_gcs_url,
            "download_url": download_url,
            "input_files": request.input_gcs_urls,
            "work_id": work_id
        }
        
    except Exception as e:
        raise HTTPException(500, f"Command execution failed: {str(e)}")
    
    finally:
        # Clean up workspace
        if work_dir.exists():
            shutil.rmtree(work_dir)


@app.post("/gcs/execute-async", status_code=202)
async def execute_gcs_command_async(
    request: GCSCommand,
    background_tasks: BackgroundTasks
):
    """Execute GDAL command on GCS files asynchronously"""
    if request.command not in ALLOWED_COMMANDS:
        raise HTTPException(400, f"Command '{request.command}' not allowed")
    
    # Create job
    job_id = str(uuid.uuid4())
    job = CommandJob(
        id=job_id,
        status=JobStatus.PENDING,
        command=request.command,
        args=request.args,
        input_gcs_urls=request.input_gcs_urls,
        output_gcs_url=request.output_gcs_url,
        created_at=datetime.utcnow()
    )
    
    jobs_store[job_id] = job
    
    # Execute in background
    background_tasks.add_task(process_gcs_command_async, job_id, request)
    
    return {
        "job_id": job_id,
        "status": "pending",
        "status_url": f"/jobs/{job_id}",
        "message": "GCS command execution started",
        "input_files": request.input_gcs_urls,
        "output_file": request.output_gcs_url
    }


@app.get("/gcs/download/{bucket_name}/{object_path:path}")
async def download_gcs_file(bucket_name: str, object_path: str):
    """Generate signed URL for downloading GCS file"""
    try:
        signed_url = gcs_handler.generate_signed_url(
            bucket_name, 
            object_path, 
            expiration=300,  # 5 minutes
            method="GET"
        )
        
        # Redirect to signed URL
        return RedirectResponse(url=signed_url)
        
    except Exception as e:
        raise HTTPException(500, f"Failed to generate download URL: {str(e)}")


# Keep existing endpoints for backward compatibility
# (All the original endpoints from main.py stay the same)

async def process_gcs_command_async(job_id: str, request: GCSCommand):
    """Process GCS command asynchronously"""
    job = jobs_store[job_id]
    start_time = datetime.utcnow()
    
    work_dir = WORKSPACE_DIR / job_id
    
    try:
        job.status = JobStatus.PROCESSING
        work_dir.mkdir(exist_ok=True)
        
        # Download input files
        local_inputs = []
        if request.input_gcs_urls:
            for gcs_url in request.input_gcs_urls:
                local_path = await gcs_handler.download_file(gcs_url, work_dir)
                local_inputs.append(str(local_path))
        
        # Prepare output path
        output_filename = f"output_{job_id}"
        if request.output_gcs_url:
            output_filename = Path(request.output_gcs_url).name
        
        local_output = work_dir / output_filename
        
        # Replace placeholders in args (same logic as sync version)
        prepared_args = []
        input_index = 0
        
        for arg in request.args:
            if arg == "{input}":
                if input_index < len(local_inputs):
                    prepared_args.append(local_inputs[input_index])
                    input_index += 1
            elif arg == "{output}":
                prepared_args.append(str(local_output))
            elif arg.startswith("{input") and arg.endswith("}"):
                try:
                    idx = int(arg[6:-1])
                    if idx < len(local_inputs):
                        prepared_args.append(local_inputs[idx])
                    else:
                        prepared_args.append(arg)
                except:
                    prepared_args.append(arg)
            else:
                prepared_args.append(arg)
        
        # Execute command
        cmd = [request.command] + prepared_args
        result = await run_command(cmd, cwd=work_dir)
        
        job.stdout = result.get("stdout")
        job.stderr = result.get("stderr")
        
        if result["success"]:
            # Upload output to GCS
            if local_output.exists() and request.output_gcs_url:
                await gcs_handler.upload_file(local_output, request.output_gcs_url)
                
                # Generate signed download URL
                bucket_name, object_name = gcs_handler.parse_gcs_url(request.output_gcs_url)
                job.output_signed_url = gcs_handler.generate_signed_url(
                    bucket_name, object_name, expiration=86400  # 24 hours
                )
            
            job.status = JobStatus.COMPLETED
        else:
            job.status = JobStatus.FAILED
            job.error = result.get("stderr") or "Command failed"
        
    except Exception as e:
        job.status = JobStatus.FAILED
        job.error = str(e)
    
    finally:
        job.completed_at = datetime.utcnow()
        job.execution_time_seconds = (job.completed_at - start_time).total_seconds()
        
        # Clean up workspace
        if work_dir.exists():
            shutil.rmtree(work_dir)


# Utility functions (keep existing ones and add new ones)

async def run_command(
    cmd: List[str],
    cwd: Optional[Path] = None,
    timeout: int = COMMAND_TIMEOUT
) -> Dict[str, Any]:
    """Execute a command and return results (same as before)"""
    try:
        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=cwd
        )
        
        try:
            stdout, stderr = await asyncio.wait_for(
                process.communicate(),
                timeout=timeout
            )
        except asyncio.TimeoutError:
            process.kill()
            return {
                "success": False,
                "error": f"Command timed out after {timeout} seconds",
                "stdout": "",
                "stderr": ""
            }
        
        return {
            "success": process.returncode == 0,
            "returncode": process.returncode,
            "stdout": stdout.decode('utf-8', errors='replace'),
            "stderr": stderr.decode('utf-8', errors='replace')
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "stdout": "",
            "stderr": ""
        }


def cleanup_old_files(hours: int = 24):
    """Clean up files older than specified hours (same as before)"""
    import time
    current_time = time.time()
    
    for directory in [WORKSPACE_DIR, OUTPUT_DIR, UPLOAD_DIR]:
        if not directory.exists():
            continue
            
        for filepath in directory.iterdir():
            if filepath.is_file():
                file_age_hours = (current_time - filepath.stat().st_mtime) / 3600
                if file_age_hours > hours:
                    try:
                        filepath.unlink()
                    except Exception:
                        pass


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)