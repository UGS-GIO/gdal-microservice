# app/main.py
"""
GDAL Microservice - A simple HTTP wrapper for GDAL command-line tools
Allows cloud services to execute GDAL commands via REST API
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

from fastapi import FastAPI, File, UploadFile, Form, HTTPException, BackgroundTasks, Body
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from contextlib import asynccontextmanager
import aiofiles

# Configuration
WORKSPACE_DIR = Path("/tmp/gdal-workspace")
UPLOAD_DIR = Path("/tmp/gdal-uploads")
OUTPUT_DIR = Path("/tmp/gdal-outputs")
MAX_FILE_SIZE = int(os.getenv("MAX_FILE_SIZE", 524288000))  # 500MB default
COMMAND_TIMEOUT = int(os.getenv("COMMAND_TIMEOUT", 3600))  # 1 hour default

# Ensure directories exist
for dir_path in [WORKSPACE_DIR, UPLOAD_DIR, OUTPUT_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# Allowed GDAL commands
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


class GDALCommand(BaseModel):
    """Model for GDAL command execution"""
    command: str = Field(..., description="GDAL command (e.g., ogr2ogr, gdalwarp)")
    args: List[str] = Field(..., description="Command arguments as you would use in CLI")
    input_files: Optional[Dict[str, str]] = Field(None, description="Map of filename to download URL (optional)")
    output_filename: Optional[str] = Field(None, description="Expected output filename")


class CommandJob(BaseModel):
    """Job tracking for async command execution"""
    id: str
    status: JobStatus
    command: str
    args: List[str]
    output_file: Optional[str] = None
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
    # Startup
    print(f"GDAL Microservice starting...")
    print(f"Available commands: {', '.join(ALLOWED_COMMANDS.keys())}")
    
    # Clean old files
    cleanup_old_files()
    
    yield
    
    # Shutdown
    print("Shutting down...")
    cleanup_old_files()


app = FastAPI(
    title="GDAL Microservice",
    description="HTTP wrapper for GDAL command-line tools",
    version="2.0.0",
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
        "service": "GDAL Microservice",
        "version": "2.0.0",
        "description": "Execute GDAL commands via HTTP",
        "endpoints": {
            "/health": "Health check",
            "/commands": "List available commands",
            "/execute": "Execute GDAL command synchronously",
            "/execute-async": "Execute GDAL command asynchronously",
            "/upload-and-execute": "Upload file and execute command",
            "/jobs/{job_id}": "Get job status",
            "/jobs/{job_id}/download": "Download job result"
        }
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    try:
        # Check if GDAL is available
        result = await run_command(["gdalinfo", "--version"])
        
        return {
            "status": "healthy",
            "gdal_available": result["success"],
            "gdal_version": result.get("stdout", "").strip(),
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


@app.get("/commands")
async def list_commands():
    """List available GDAL commands"""
    return {
        "commands": [
            {
                "command": cmd,
                "description": desc,
                "example": get_command_example(cmd)
            }
            for cmd, desc in ALLOWED_COMMANDS.items()
        ]
    }


@app.post("/execute")
async def execute_command(request: GDALCommand):
    """
    Execute a GDAL command synchronously and return the result.
    
    Example request:
    {
        "command": "ogr2ogr",
        "args": ["-f", "GeoJSON", "output.json", "input.shp"]
    }
    """
    # Validate command
    if request.command not in ALLOWED_COMMANDS:
        raise HTTPException(400, f"Command '{request.command}' not allowed. Use one of: {list(ALLOWED_COMMANDS.keys())}")
    
    # Create workspace for this execution
    work_id = str(uuid.uuid4())
    work_dir = WORKSPACE_DIR / work_id
    work_dir.mkdir(exist_ok=True)
    
    try:
        # Prepare file paths
        prepared_args = prepare_arguments(request.args, work_dir)
        
        # Build full command
        cmd = [request.command] + prepared_args
        
        # Execute command
        result = await run_command(cmd, cwd=work_dir)
        
        # Check for output file
        output_file = None
        if request.output_filename:
            output_path = work_dir / request.output_filename
            if output_path.exists():
                # Move to output directory
                final_path = OUTPUT_DIR / f"{work_id}_{request.output_filename}"
                shutil.move(str(output_path), str(final_path))
                output_file = str(final_path)
        
        return {
            "success": result["success"],
            "command": request.command,
            "args": request.args,
            "stdout": result.get("stdout"),
            "stderr": result.get("stderr"),
            "output_file": output_file,
            "download_url": f"/download/{work_id}/{request.output_filename}" if output_file else None
        }
        
    finally:
        # Clean up workspace
        if work_dir.exists():
            shutil.rmtree(work_dir)


@app.post("/execute-async", status_code=202)
async def execute_command_async(
    request: GDALCommand,
    background_tasks: BackgroundTasks
):
    """
    Execute a GDAL command asynchronously.
    Returns a job ID for tracking.
    """
    # Validate command
    if request.command not in ALLOWED_COMMANDS:
        raise HTTPException(400, f"Command '{request.command}' not allowed")
    
    # Create job
    job_id = str(uuid.uuid4())
    job = CommandJob(
        id=job_id,
        status=JobStatus.PENDING,
        command=request.command,
        args=request.args,
        created_at=datetime.utcnow()
    )
    
    jobs_store[job_id] = job
    
    # Execute in background
    background_tasks.add_task(
        process_command_async,
        job_id,
        request
    )
    
    return {
        "job_id": job_id,
        "status": "pending",
        "status_url": f"/jobs/{job_id}",
        "message": "Command execution started"
    }


@app.post("/upload-and-execute")
async def upload_and_execute(
    file: UploadFile = File(...),
    command: str = Form(...),
    args: str = Form(..., description="JSON array of arguments"),
    output_filename: Optional[str] = Form(None)
):
    """
    Upload a file and execute a GDAL command on it.
    
    Args should be a JSON array where {input} will be replaced with the uploaded filename.
    Example: ["-f", "GeoJSON", "output.json", "{input}"]
    """
    # Validate file size
    if file.size and file.size > MAX_FILE_SIZE:
        raise HTTPException(400, f"File size exceeds maximum of {MAX_FILE_SIZE} bytes")
    
    # Parse arguments
    try:
        args_list = json.loads(args)
    except json.JSONDecodeError:
        raise HTTPException(400, "Invalid JSON in args parameter")
    
    # Validate command
    if command not in ALLOWED_COMMANDS:
        raise HTTPException(400, f"Command '{command}' not allowed")
    
    # Create workspace
    work_id = str(uuid.uuid4())
    work_dir = WORKSPACE_DIR / work_id
    work_dir.mkdir(exist_ok=True)
    
    try:
        # Save uploaded file
        input_path = work_dir / file.filename
        async with aiofiles.open(input_path, 'wb') as f:
            content = await file.read()
            await f.write(content)
        
        # Replace {input} placeholder in args
        prepared_args = [
            str(input_path) if arg == "{input}" else arg
            for arg in args_list
        ]
        
        # Build and execute command
        cmd = [command] + prepared_args
        result = await run_command(cmd, cwd=work_dir)
        
        # Check for output file
        output_file = None
        download_url = None
        
        if output_filename:
            output_path = work_dir / output_filename
            if output_path.exists():
                # Move to output directory
                final_path = OUTPUT_DIR / f"{work_id}_{output_filename}"
                shutil.move(str(output_path), str(final_path))
                output_file = str(final_path)
                download_url = f"/download/{work_id}/{output_filename}"
        else:
            # Try to find any output file
            for item in work_dir.iterdir():
                if item.is_file() and item.name != file.filename:
                    final_path = OUTPUT_DIR / f"{work_id}_{item.name}"
                    shutil.move(str(item), str(final_path))
                    output_file = str(final_path)
                    download_url = f"/download/{work_id}/{item.name}"
                    break
        
        return {
            "success": result["success"],
            "command": command,
            "args": prepared_args,
            "stdout": result.get("stdout"),
            "stderr": result.get("stderr"),
            "output_file": output_file,
            "download_url": download_url
        }
        
    finally:
        # Clean up workspace
        if work_dir.exists():
            shutil.rmtree(work_dir)


@app.get("/jobs/{job_id}")
async def get_job_status(job_id: str):
    """Get job status and results"""
    if job_id not in jobs_store:
        raise HTTPException(404, "Job not found")
    
    job = jobs_store[job_id]
    return {
        "id": job.id,
        "status": job.status,
        "command": job.command,
        "args": job.args,
        "created_at": job.created_at.isoformat(),
        "completed_at": job.completed_at.isoformat() if job.completed_at else None,
        "execution_time_seconds": job.execution_time_seconds,
        "output_file": job.output_file,
        "download_url": f"/jobs/{job_id}/download" if job.output_file else None,
        "error": job.error,
        "stdout": job.stdout,
        "stderr": job.stderr
    }


@app.get("/jobs/{job_id}/download")
async def download_job_result(job_id: str):
    """Download the output file from a completed job"""
    if job_id not in jobs_store:
        raise HTTPException(404, "Job not found")
    
    job = jobs_store[job_id]
    
    if job.status != JobStatus.COMPLETED:
        raise HTTPException(400, f"Job status is {job.status}, not completed")
    
    if not job.output_file or not Path(job.output_file).exists():
        raise HTTPException(404, "Output file not found")
    
    return FileResponse(
        path=job.output_file,
        filename=Path(job.output_file).name,
        media_type="application/octet-stream"
    )


@app.get("/download/{work_id}/{filename}")
async def download_file(work_id: str, filename: str):
    """Direct download of output files"""
    file_path = OUTPUT_DIR / f"{work_id}_{filename}"
    
    if not file_path.exists():
        raise HTTPException(404, "File not found")
    
    return FileResponse(
        path=str(file_path),
        filename=filename,
        media_type="application/octet-stream"
    )


# Helper functions

async def run_command(
    cmd: List[str],
    cwd: Optional[Path] = None,
    timeout: int = COMMAND_TIMEOUT
) -> Dict[str, Any]:
    """Execute a command and return results"""
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


async def process_command_async(job_id: str, request: GDALCommand):
    """Process command asynchronously"""
    job = jobs_store[job_id]
    start_time = datetime.utcnow()
    
    try:
        job.status = JobStatus.PROCESSING
        
        # Create workspace
        work_dir = WORKSPACE_DIR / job_id
        work_dir.mkdir(exist_ok=True)
        
        # Prepare arguments
        prepared_args = prepare_arguments(request.args, work_dir)
        
        # Build command
        cmd = [request.command] + prepared_args
        
        # Execute
        result = await run_command(cmd, cwd=work_dir)
        
        # Store results
        job.stdout = result.get("stdout")
        job.stderr = result.get("stderr")
        
        if result["success"]:
            # Check for output file
            if request.output_filename:
                output_path = work_dir / request.output_filename
                if output_path.exists():
                    final_path = OUTPUT_DIR / f"{job_id}_{request.output_filename}"
                    shutil.move(str(output_path), str(final_path))
                    job.output_file = str(final_path)
            
            job.status = JobStatus.COMPLETED
        else:
            job.status = JobStatus.FAILED
            job.error = result.get("error") or "Command failed"
        
    except Exception as e:
        job.status = JobStatus.FAILED
        job.error = str(e)
    
    finally:
        job.completed_at = datetime.utcnow()
        job.execution_time_seconds = (job.completed_at - start_time).total_seconds()
        
        # Clean up workspace
        if work_dir.exists():
            shutil.rmtree(work_dir)


def prepare_arguments(args: List[str], work_dir: Path) -> List[str]:
    """Prepare command arguments, handling file paths"""
    prepared = []
    
    for arg in args:
        # Check if argument looks like a file path
        if "/" in arg or "\\" in arg or "." in arg:
            # If it's a relative path, make it absolute within work_dir
            path = Path(arg)
            if not path.is_absolute():
                arg = str(work_dir / arg)
        
        prepared.append(arg)
    
    return prepared


def get_command_example(command: str) -> str:
    """Get example usage for a command"""
    examples = {
        "ogr2ogr": '["-f", "GeoJSON", "output.json", "input.shp"]',
        "ogrinfo": '["-al", "-so", "input.shp"]',
        "gdal_translate": '["-of", "GTiff", "input.jp2", "output.tif"]',
        "gdalwarp": '["-t_srs", "EPSG:4326", "input.tif", "output.tif"]',
        "gdalinfo": '["input.tif"]',
        "gdal_rasterize": '["-a", "ATTRIBUTE", "-ts", "1024", "1024", "input.shp", "output.tif"]',
        "gdal_polygonize": '["input.tif", "output.shp"]',
        "gdal_contour": '["-a", "elev", "-i", "10", "input.tif", "output.shp"]',
        "gdaldem": '["hillshade", "input.tif", "output.tif"]',
        "ogrmerge": '["-o", "merged.shp", "file1.shp", "file2.shp"]'
    }
    return examples.get(command, "[]")


def cleanup_old_files(hours: int = 24):
    """Clean up files older than specified hours"""
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