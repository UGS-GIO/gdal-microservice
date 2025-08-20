"""Google Cloud Storage integration for GDAL operations"""

import os
import re
import uuid
import asyncio
import tempfile
from pathlib import Path
from typing import Optional, Dict, List, Tuple
from datetime import datetime, timedelta
import logging

from google.cloud import storage
from google.auth.exceptions import DefaultCredentialsError

logger = logging.getLogger(__name__)


class GCSHandler:
    """Handle Google Cloud Storage operations for GDAL processing"""
    
    def __init__(self, project_id: str = None, default_bucket: str = None):
        """
        Initialize GCS handler
        
        Args:
            project_id: GCP project ID
            default_bucket: Default bucket for operations
        """
        self.project_id = project_id or os.getenv("GCP_PROJECT")
        self.default_bucket = default_bucket or os.getenv("GCS_BUCKET")
        
        try:
            self.client = storage.Client(project=self.project_id)
            logger.info(f"GCS client initialized for project: {self.project_id}")
        except DefaultCredentialsError:
            logger.warning("No GCS credentials found, GCS operations will fail")
            self.client = None
    
    def parse_gcs_url(self, gcs_url: str) -> Tuple[str, str]:
        """Parse GCS URL into bucket and object path"""
        if not gcs_url.startswith("gs://"):
            raise ValueError("URL must start with gs://")
        
        match = re.match(r"gs://([^/]+)/(.+)", gcs_url)
        if not match:
            raise ValueError("Invalid GCS URL format: gs://bucket/path/file")
        
        return match.groups()
    
    def generate_signed_url(self, bucket_name: str, object_name: str, 
                          expiration: int = 3600, method: str = "GET") -> str:
        """
        Generate a signed URL for GCS object
        
        Args:
            bucket_name: GCS bucket name
            object_name: Object path in bucket
            expiration: URL expiration in seconds
            method: HTTP method (GET, PUT, POST)
        
        Returns:
            Signed URL string
        """
        if not self.client:
            raise RuntimeError("GCS client not initialized")
        
        bucket = self.client.bucket(bucket_name)
        blob = bucket.blob(object_name)
        
        url = blob.generate_signed_url(
            version="v4",
            expiration=datetime.utcnow() + timedelta(seconds=expiration),
            method=method
        )
        
        return url
    
    async def download_file(self, gcs_url: str, local_dir: Path) -> Path:
        """
        Download file from GCS to local directory
        
        Args:
            gcs_url: GCS URL (gs://bucket/path)
            local_dir: Local directory to save file
        
        Returns:
            Path to downloaded file
        """
        bucket_name, object_name = self.parse_gcs_url(gcs_url)
        
        # Generate unique local filename
        original_name = Path(object_name).name
        local_filename = f"{uuid.uuid4()}_{original_name}"
        local_path = local_dir / local_filename
        
        try:
            # Use gsutil for better performance with large files
            cmd = ["gsutil", "-m", "cp", gcs_url, str(local_path)]
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            stdout, stderr = await process.communicate()
            
            if process.returncode != 0:
                # Fallback to Python client
                await self._download_with_client(bucket_name, object_name, local_path)
            
            logger.info(f"Downloaded {gcs_url} to {local_path}")
            return local_path
            
        except Exception as e:
            logger.error(f"Failed to download {gcs_url}: {e}")
            raise
    
    async def _download_with_client(self, bucket_name: str, object_name: str, local_path: Path):
        """Download using Python client as fallback"""
        if not self.client:
            raise RuntimeError("GCS client not initialized")
        
        bucket = self.client.bucket(bucket_name)
        blob = bucket.blob(object_name)
        
        # Download in chunks for large files
        with open(local_path, "wb") as f:
            blob.download_to_file(f)
    
    async def upload_file(self, local_path: Path, gcs_url: str) -> str:
        """
        Upload file to GCS
        
        Args:
            local_path: Local file path
            gcs_url: Target GCS URL
        
        Returns:
            GCS URL of uploaded file
        """
        if not local_path.exists():
            raise FileNotFoundError(f"File not found: {local_path}")
        
        bucket_name, object_name = self.parse_gcs_url(gcs_url)
        
        try:
            # Use gsutil for better performance
            cmd = ["gsutil", "-m", "cp", str(local_path), gcs_url]
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            stdout, stderr = await process.communicate()
            
            if process.returncode != 0:
                # Fallback to Python client
                await self._upload_with_client(local_path, bucket_name, object_name)
            
            logger.info(f"Uploaded {local_path} to {gcs_url}")
            return gcs_url
            
        except Exception as e:
            logger.error(f"Failed to upload to {gcs_url}: {e}")
            raise
    
    async def _upload_with_client(self, local_path: Path, bucket_name: str, object_name: str):
        """Upload using Python client as fallback"""
        if not self.client:
            raise RuntimeError("GCS client not initialized")
        
        bucket = self.client.bucket(bucket_name)
        blob = bucket.blob(object_name)
        
        # Upload in chunks for large files
        blob.upload_from_filename(str(local_path))
    
    def get_file_info(self, gcs_url: str) -> Dict:
        """Get information about a GCS file"""
        if not self.client:
            raise RuntimeError("GCS client not initialized")
        
        bucket_name, object_name = self.parse_gcs_url(gcs_url)
        bucket = self.client.bucket(bucket_name)
        blob = bucket.blob(object_name)
        
        if not blob.exists():
            raise FileNotFoundError(f"File not found: {gcs_url}")
        
        blob.reload()
        
        return {
            "name": blob.name,
            "size": blob.size,
            "content_type": blob.content_type,
            "created": blob.time_created,
            "updated": blob.updated,
            "etag": blob.etag
        }
    
    def list_files(self, bucket_name: str = None, prefix: str = "") -> List[Dict]:
        """List files in bucket with optional prefix"""
        if not self.client:
            raise RuntimeError("GCS client not initialized")
        
        bucket_name = bucket_name or self.default_bucket
        if not bucket_name:
            raise ValueError("No bucket specified")
        
        bucket = self.client.bucket(bucket_name)
        blobs = bucket.list_blobs(prefix=prefix)
        
        files = []
        for blob in blobs:
            files.append({
                "name": blob.name,
                "size": blob.size,
                "content_type": blob.content_type,
                "updated": blob.updated,
                "gcs_url": f"gs://{bucket_name}/{blob.name}"
            })
        
        return files


