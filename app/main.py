"""
GDAL Microservice with proper Google Cloud Storage VSI support
"""

import os
import asyncio
import shutil
import json
import uuid
import tempfile
import re
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, List, Any
from enum import Enum

from fastapi import FastAPI, File, UploadFile, Form, HTTPException, BackgroundTasks, Body
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator
from contextlib import asynccontextmanager
import aiofiles

# Try to import GCS handler, but don't fail if dependencies aren't available
try:
    from .gcs_handler import GCSHandler
    GCS_AVAILABLE = True
except ImportError:
    print("⚠️  GCS dependencies not available, running in local-only mode")
    GCSHandler = None
    GCS_AVAILABLE = False

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

# Initialize GCS handler if available
gcs_handler = None
if GCS_AVAILABLE:
    try:
        gcs_handler = GCSHandler(default_bucket=GCS_BUCKET)
        print(f"✅ GCS integration enabled with bucket: {GCS_BUCKET}")
    except Exception as e:
        print(f"⚠️  GCS initialization failed: {e}")
        GCS_AVAILABLE = False

# Configure GDAL for GCS access
def setup_gdal_gcs():
    """Configure GDAL to work with Google Cloud Storage"""
    # Enable GDAL's GCS virtual file system
    os.environ["CPL_GS_OAUTH_REFRESH_TOKEN"] = ""
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = os.getenv("GOOGLE_APPLICATION_CREDENTIALS", "")
    
    # Configure GDAL caching and performance
    os.environ["GDAL_CACHEMAX"] = str(os.getenv("GDAL_CACHEMAX", 1024))
    os.environ["GDAL_NUM_THREADS"] = os.getenv("GDAL_NUM_THREADS", "ALL_CPUS")
    os.environ["CPL_TMPDIR"] = str(WORKSPACE_DIR)
    
    # Enable verbose errors for debugging
    os.environ["CPL_DEBUG"] = os.getenv("CPL_DEBUG", "OFF")
    
    print("🔧 GDAL configured for Google Cloud Storage access")

# Setup GDAL on startup
setup_gdal_gcs()

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
    """Model for GDAL command execution (original format)"""
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
    print(f"🚀 GDAL Microservice starting...")
    print(f"GCS Integration: {'✅ Enabled' if GCS_AVAILABLE else '❌ Disabled'}")
    if GCS_AVAILABLE and GCS_BUCKET:
        print(f"GCS Bucket: {GCS_BUCKET}")
    print(f"Available commands: {', '.join(ALLOWED_COMMANDS.keys())}")
    
    # Test GDAL-GCS connectivity
    await test_gdal_gcs_connectivity()
    
    # Clean old files
    cleanup_old_files()
    
    yield
    
    # Shutdown
    print("Shutting down...")
    cleanup_old_files()


app = FastAPI(
    title="GDAL Microservice",
    description="HTTP wrapper for GDAL command-line tools with Google Cloud Storage support",
    version="2.0.1",
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
        "version": "2.0.1",
        "description": "Execute GDAL commands via HTTP with GCS VSI support",
        "gcs_enabled": GCS_AVAILABLE,
        "gdal_gcs_configured": check_gdal_gcs_config(),
        "default_bucket": GCS_BUCKET if GCS_AVAILABLE else None,
        "endpoints": {
            "/health": "Health check",
            "/commands": "List available commands",
            "/execute": "Execute GDAL command synchronously",
            "/execute-async": "Execute GDAL command asynchronously",
            "/upload-and-execute": "Upload file and execute command",
            "/jobs/{job_id}": "Get job status",
            "/jobs/{job_id}/download": "Download job result",
            "/test-gcs": "Test GCS connectivity"
        }
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    try:
        # Check if GDAL is available
        result = await run_command(["gdalinfo", "--version"])
        gcs_test = await test_simple_gcs_access()
        
        return {
            "status": "healthy",
            "gdal_available": result["success"],
            "gdal_version": result.get("stdout", "").strip(),
            "gcs_enabled": GCS_AVAILABLE,
            "gcs_vsi_working": gcs_test,
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


@app.get("/test-gcs")
async def test_gcs_endpoint():
    """Test GCS connectivity"""
    if not GCS_AVAILABLE:
        return {"gcs_available": False, "error": "GCS dependencies not installed"}
    
    test_results = {}
    
    # Test 1: Basic GCS access
    test_results["basic_gcs"] = await test_simple_gcs_access()
    
    # Test 2: GDAL VSI GCS access
    test_results["gdal_vsi_gcs"] = await test_gdal_vsi_gcs()
    
    # Test 3: Authentication status
    test_results["auth_configured"] = check_gdal_gcs_config()
    
    return {
        "gcs_tests": test_results,
        "recommendations": get_gcs_troubleshooting_tips(test_results)
    }


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
    Handles GCS VSI paths properly by downloading files when needed.
    """
    # Validate command
    if request.command not in ALLOWED_COMMANDS:
        raise HTTPException(400, f"Command '{request.command}' not allowed. Use one of: {list(ALLOWED_COMMANDS.keys())}")
    
    # Create workspace for this execution
    work_id = str(uuid.uuid4())
    work_dir = WORKSPACE_DIR / work_id
    work_dir.mkdir(exist_ok=True)
    
    try:
        print(f"🔍 Processing command: {request.command}")
        print(f"📁 Working directory: {work_dir}")
        print(f"📝 Arguments: {request.args}")
        
        # Process arguments and handle GCS VSI paths
        prepared_args = await prepare_arguments_with_gcs(request.args, work_dir)
        
        print(f"✅ Prepared arguments: {prepared_args}")
        
        # Build full command
        cmd = [request.command] + prepared_args
        
        # Execute command
        print(f"🚀 Executing: {' '.join(cmd)}")
        result = await run_command(cmd, cwd=work_dir)
        
        # Check for output file
        output_file = None
        download_url = None
        
        if request.output_filename:
            output_path = work_dir / request.output_filename
            if output_path.exists():
                # Move to output directory
                final_path = OUTPUT_DIR / f"{work_id}_{request.output_filename}"
                shutil.move(str(output_path), str(final_path))
                output_file = str(final_path)
                download_url = f"/download/{work_id}/{request.output_filename}"
        
        return {
            "success": result["success"],
            "command": request.command,
            "args": request.args,
            "prepared_args": prepared_args,
            "stdout": result.get("stdout"),
            "stderr": result.get("stderr"),
            "output_file": output_file,
            "download_url": download_url,
            "work_id": work_id
        }
        
    except Exception as e:
        print(f"❌ Command execution failed: {str(e)}")
        raise HTTPException(500, f"Command execution failed: {str(e)}")
        
    finally:
        # Clean up workspace
        if work_dir.exists():
            shutil.rmtree(work_dir)


@app.post("/execute-async", status_code=202)
async def execute_command_async(
    request: GDALCommand,
    background_tasks: BackgroundTasks
):
    """Execute a GDAL command asynchronously"""
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
    """Upload a file and execute a GDAL command on it"""
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
    response = {
        "id": job.id,
        "status": job.status,
        "command": job.command,
        "args": job.args,
        "created_at": job.created_at.isoformat(),
        "completed_at": job.completed_at.isoformat() if job.completed_at else None,
        "execution_time_seconds": job.execution_time_seconds,
        "output_file": job.output_file,
        "error": job.error,
        "stdout": job.stdout,
        "stderr": job.stderr
    }
    
    # Add download URL
    if job.output_file:
        response["download_url"] = f"/jobs/{job_id}/download"
    
    return response


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


# ===== HELPER FUNCTIONS =====

async def prepare_arguments_with_gcs(args: List[str], work_dir: Path) -> List[str]:
    """
    Prepare command arguments, handling GCS VSI paths by downloading files when needed
    """
    prepared_args = []
    
    for arg in args:
        if is_gcs_vsi_path(arg):
            print(f"🔄 Processing GCS VSI path: {arg}")
            # Download the file and replace with local path
            local_path = await download_gcs_vsi_file(arg, work_dir)
            prepared_args.append(str(local_path))
        elif arg.startswith('/vsi') and 'gs://' in arg:
            print(f"🔄 Processing complex VSI path: {arg}")
            # Handle complex VSI paths like /vsizip/gs://bucket/file.zip/folder
            local_path = await download_and_prepare_vsi_path(arg, work_dir)
            prepared_args.append(local_path)
        else:
            # Keep argument as-is
            prepared_args.append(arg)
    
    return prepared_args


def is_gcs_vsi_path(path: str) -> bool:
    """Check if path is a GCS VSI path that needs special handling"""
    return (path.startswith('/vsizip/gs://') or 
            path.startswith('/vsigzip/gs://') or 
            path.startswith('/vsitar/gs://') or
            path.startswith('gs://'))


async def download_gcs_vsi_file(vsi_path: str, work_dir: Path) -> Path:
    """
    Download a file referenced in a VSI path
    
    Examples:
    - /vsizip/gs://bucket/file.zip/folder -> downloads file.zip
    - gs://bucket/file.shp -> downloads file.shp
    """
    print(f"📥 Downloading GCS VSI file: {vsi_path}")
    
    # Extract the GCS URL from the VSI path
    gcs_url_match = re.search(r'gs://[^/]+/[^/\s]+', vsi_path)
    if not gcs_url_match:
        raise ValueError(f"Could not extract GCS URL from: {vsi_path}")
    
    gcs_url = gcs_url_match.group(0)
    print(f"📍 Extracted GCS URL: {gcs_url}")
    
    # Download using gsutil
    filename = Path(gcs_url).name
    local_path = work_dir / filename
    
    try:
        cmd = ["gsutil", "cp", gcs_url, str(local_path)]
        result = await run_command(cmd)
        
        if not result["success"]:
            raise Exception(f"gsutil failed: {result.get('stderr', 'Unknown error')}")
        
        print(f"✅ Downloaded to: {local_path}")
        return local_path
        
    except Exception as e:
        print(f"❌ Failed to download {gcs_url}: {e}")
        raise


async def download_and_prepare_vsi_path(vsi_path: str, work_dir: Path) -> str:
    """
    Download and prepare a complex VSI path
    
    Example: /vsizip/gs://bucket/file.zip/folder.gdb
    -> Downloads file.zip -> Returns /vsizip/local_path/file.zip/folder.gdb
    """
    print(f"🔧 Preparing complex VSI path: {vsi_path}")
    
    # Parse the VSI path components
    vsi_prefix = ""
    remaining_path = vsi_path
    
    # Extract VSI driver prefix
    vsi_drivers = ['/vsizip/', '/vsigzip/', '/vsitar/', '/vsi7z/', '/vsirar/']
    for driver in vsi_drivers:
        if vsi_path.startswith(driver):
            vsi_prefix = driver
            remaining_path = vsi_path[len(driver):]
            break
    
    # Extract GCS URL and internal path
    if 'gs://' in remaining_path:
        parts = remaining_path.split('/')
        gcs_parts = []
        internal_parts = []
        
        # Find where GCS URL ends (look for file extension)
        gcs_ended = False
        for i, part in enumerate(parts):
            if not gcs_ended and any(part.endswith(ext) for ext in ['.zip', '.tar', '.gz', '.7z', '.rar']):
                gcs_parts.append(part)
                gcs_ended = True
            elif not gcs_ended:
                gcs_parts.append(part)
            else:
                internal_parts.append(part)
        
        gcs_url = '/'.join(gcs_parts)
        internal_path = '/'.join(internal_parts) if internal_parts else ""
        
        print(f"📍 GCS URL: {gcs_url}")
        print(f"📁 Internal path: {internal_path}")
        
        # Download the file
        local_file = await download_gcs_vsi_file(gcs_url, work_dir)
        
        # Construct new VSI path with local file
        if internal_path:
            new_vsi_path = f"{vsi_prefix}{local_file}/{internal_path}"
        else:
            new_vsi_path = f"{vsi_prefix}{local_file}"
        
        print(f"✅ New VSI path: {new_vsi_path}")
        return new_vsi_path
    
    else:
        # No GCS URL found, return as-is
        return vsi_path


async def run_command(
    cmd: List[str],
    cwd: Optional[Path] = None,
    timeout: int = COMMAND_TIMEOUT
) -> Dict[str, Any]:
    """Execute a command and return results"""
    try:
        print(f"🔧 Running command: {' '.join(cmd)}")
        
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
        
        stdout_str = stdout.decode('utf-8', errors='replace')
        stderr_str = stderr.decode('utf-8', errors='replace')
        
        if process.returncode != 0:
            print(f"❌ Command failed with return code {process.returncode}")
            print(f"📤 STDOUT: {stdout_str}")
            print(f"📤 STDERR: {stderr_str}")
        else:
            print(f"✅ Command completed successfully")
        
        return {
            "success": process.returncode == 0,
            "returncode": process.returncode,
            "stdout": stdout_str,
            "stderr": stderr_str
        }
        
    except Exception as e:
        print(f"❌ Command execution error: {e}")
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
    
    work_dir = WORKSPACE_DIR / job_id
    
    try:
        job.status = JobStatus.PROCESSING
        work_dir.mkdir(exist_ok=True)
        
        # Prepare arguments with GCS handling
        prepared_args = await prepare_arguments_with_gcs(request.args, work_dir)
        
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
            job.error = result.get("error") or result.get("stderr") or "Command failed"
        
    except Exception as e:
        job.status = JobStatus.FAILED
        job.error = str(e)
    
    finally:
        job.completed_at = datetime.utcnow()
        job.execution_time_seconds = (job.completed_at - start_time).total_seconds()
        
        # Clean up workspace
        if work_dir.exists():
            shutil.rmtree(work_dir)


# ===== GCS TESTING AND CONFIGURATION =====

def check_gdal_gcs_config() -> bool:
    """Check if GDAL is properly configured for GCS access"""
    # Check for authentication
    auth_methods = [
        os.getenv("GOOGLE_APPLICATION_CREDENTIALS"),
        os.getenv("GCLOUD_PROJECT"),
        os.path.exists("/var/secrets/google/key.json"),  # Common Cloud Run path
    ]
    
    return any(auth_methods)


async def test_simple_gcs_access() -> bool:
    """Test basic GCS access using gsutil"""
    if not GCS_BUCKET:
        return False
    
    try:
        cmd = ["gsutil", "ls", f"gs://{GCS_BUCKET}/", "-l"]
        result = await run_command(cmd, timeout=30)
        return result["success"]
    except:
        return False


async def test_gdal_vsi_gcs() -> bool:
    """Test GDAL VSI GCS access"""
    if not GCS_BUCKET:
        return False
    
    try:
        # Try to list a GCS bucket using GDAL
        cmd = ["gdalinfo", f"/vsigs/{GCS_BUCKET}"]
        result = await run_command(cmd, timeout=30)
        return result["success"]
    except:
        return False


async def test_gdal_gcs_connectivity():
    """Test GDAL-GCS connectivity on startup"""
    print("🧪 Testing GDAL-GCS connectivity...")
    
    if not GCS_BUCKET:
        print("⚠️  No GCS bucket configured, skipping GCS tests")
        return
    
    # Test gsutil
    gsutil_works = await test_simple_gcs_access()
    print(f"📦 gsutil access: {'✅' if gsutil_works else '❌'}")
    
    # Test GDAL VSI
    gdal_vsi_works = await test_gdal_vsi_gcs()
    print(f"🔧 GDAL VSI GCS: {'✅' if gdal_vsi_works else '❌'}")
    
    if not gsutil_works:
        print("💡 Tip: Ensure Google Cloud credentials are properly configured")
        print("   - Service account key file")
        print("   - Workload Identity (recommended for Cloud Run)")
        print("   - Application Default Credentials")


def get_gcs_troubleshooting_tips(test_results: dict) -> List[str]:
    """Get troubleshooting tips based on test results"""
    tips = []
    
    if not test_results.get("basic_gcs", False):
        tips.append("Configure Google Cloud authentication (service account or Workload Identity)")
        tips.append("Ensure the service has Storage Object Viewer permissions")
    
    if not test_results.get("gdal_vsi_gcs", False):
        tips.append("GDAL VSI GCS access failed - this is expected, download files instead")
    
    if not test_results.get("auth_configured", False):
        tips.append("Set GOOGLE_APPLICATION_CREDENTIALS environment variable")
    
    return tips


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