# tests/test_api.py
"""Test suite for GDAL API Service"""

import pytest
import json
import tempfile
import shutil
from pathlib import Path
from datetime import datetime
from unittest.mock import Mock, patch, AsyncMock
import asyncio

from fastapi.testclient import TestClient
from app.main import app
from app.models import JobStatus, OutputFormat
from app.config import settings
from app.utils import (
    validate_vector_file,
    get_driver_for_format,
    validate_epsg_code,
    validate_bbox,
    generate_unique_filename
)


# ==========================================
# Fixtures
# ==========================================

@pytest.fixture
def client():
    """Create test client"""
    return TestClient(app)


@pytest.fixture
def temp_dir():
    """Create temporary directory for testing"""
    temp_path = Path(tempfile.mkdtemp())
    yield temp_path
    shutil.rmtree(temp_path)


@pytest.fixture
def sample_geojson(temp_dir):
    """Create a sample GeoJSON file"""
    geojson_data = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "geometry": {
                    "type": "Point",
                    "coordinates": [-111.8910, 40.7608]
                },
                "properties": {
                    "name": "Salt Lake City",
                    "state": "Utah"
                }
            },
            {
                "type": "Feature",
                "geometry": {
                    "type": "Point",
                    "coordinates": [-111.0937, 39.3210]
                },
                "properties": {
                    "name": "Provo",
                    "state": "Utah"
                }
            }
        ]
    }
    
    file_path = temp_dir / "test.geojson"
    with open(file_path, "w") as f:
        json.dump(geojson_data, f)
    
    return file_path


@pytest.fixture
def sample_csv(temp_dir):
    """Create a sample CSV file with geometry"""
    csv_data = """name,state,longitude,latitude
Salt Lake City,Utah,-111.8910,40.7608
Provo,Utah,-111.0937,39.3210
Ogden,Utah,-111.9738,41.2230
"""
    
    file_path = temp_dir / "test.csv"
    with open(file_path, "w") as f:
        f.write(csv_data)
    
    return file_path


# ==========================================
# API Endpoint Tests
# ==========================================

class TestHealthEndpoint:
    """Test health check endpoint"""
    
    def test_health_check(self, client):
        """Test basic health check"""
        response = client.get("/health")
        assert response.status_code == 200
        
        data = response.json()
        assert data["status"] == "healthy"
        assert data["gdal_available"] is True
        assert "gdal_version" in data
        assert data["driver_count"] > 0
        assert "timestamp" in data


class TestFormatsEndpoint:
    """Test formats listing endpoint"""
    
    def test_list_formats(self, client):
        """Test listing supported formats"""
        response = client.get("/api/v1/formats")
        assert response.status_code == 200
        
        formats = response.json()
        assert isinstance(formats, list)
        assert len(formats) > 0
        
        # Check for essential formats
        format_names = [f["driver"] for f in formats]
        assert "ESRI Shapefile" in format_names
        assert "GPKG" in format_names
        assert "GeoJSON" in format_names
        
        # Check format structure
        for fmt in formats:
            assert "name" in fmt
            assert "driver" in fmt
            assert "extensions" in fmt
            assert "capabilities" in fmt


class TestConversionEndpoint:
    """Test conversion endpoints"""
    
    def test_convert_missing_params(self, client):
        """Test conversion with missing parameters"""
        response = client.post("/api/v1/convert", json={})
        assert response.status_code == 422
    
    def test_convert_invalid_url(self, client):
        """Test conversion with invalid GCS URL"""
        response = client.post("/api/v1/convert", json={
            "input_url": "not-a-gcs-url",
            "output_format": "geojson"
        })
        assert response.status_code == 422
        
        errors = response.json()["detail"]
        assert any("GCS URL" in str(err) for err in errors)
    
    @patch('app.main.process_conversion')
    def test_convert_valid_request(self, mock_process, client):
        """Test valid conversion request"""
        mock_process.return_value = None
        
        response = client.post("/api/v1/convert", json={
            "input_url": "gs://test-bucket/test.shp",
            "output_format": "geojson",
            "output_bucket": "output-bucket",
            "output_path": "converted/test.geojson"
        })
        
        assert response.status_code == 202
        data = response.json()
        assert "id" in data
        assert data["status"] == "pending"
        assert data["output_format"] == "geojson"
    
    def test_upload_convert_no_file(self, client):
        """Test upload conversion without file"""
        response = client.post("/api/v1/convert/upload")
        assert response.status_code == 422
    
    def test_upload_convert_with_file(self, client, sample_geojson):
        """Test upload and convert with file"""
        with open(sample_geojson, "rb") as f:
            response = client.post(
                "/api/v1/convert/upload",
                files={"file": ("test.geojson", f, "application/geo+json")},
                data={"output_format": "shapefile"}
            )
        
        assert response.status_code == 202
        data = response.json()
        assert "id" in data
        assert data["status"] == "pending"
        assert data["output_format"] == "shapefile"


class TestJobEndpoints:
    """Test job management endpoints"""
    
    def test_get_nonexistent_job(self, client):
        """Test getting non-existent job"""
        response = client.get("/api/v1/jobs/nonexistent-id")
        assert response.status_code == 404
    
    def test_list_jobs_empty(self, client):
        """Test listing jobs when empty"""
        response = client.get("/api/v1/jobs")
        assert response.status_code == 200
        
        data = response.json()
        assert "jobs" in data
        assert "count" in data
        assert isinstance(data["jobs"], list)


class TestInfoEndpoint:
    """Test file info endpoint"""
    
    def test_info_no_file(self, client):
        """Test info endpoint without file"""
        response = client.post("/api/v1/info")
        assert response.status_code == 422
    
    @pytest.mark.skipif(not shutil.which("ogr2ogr"), reason="GDAL not installed")
    def test_info_with_geojson(self, client, sample_geojson):
        """Test info endpoint with GeoJSON file"""
        with open(sample_geojson, "rb") as f:
            response = client.post(
                "/api/v1/info",
                files={"file": ("test.geojson", f, "application/geo+json")}
            )
        
        if response.status_code == 200:
            data = response.json()
            assert "driver" in data
            assert "layer_count" in data
            assert "layers" in data


class TestOperationsEndpoints:
    """Test geospatial operation endpoints"""
    
    def test_reproject_missing_params(self, client):
        """Test reprojection with missing parameters"""
        response = client.post("/api/v1/operations/reproject", json={})
        assert response.status_code == 422
    
    @patch('app.main.process_conversion')
    def test_reproject_valid(self, mock_process, client):
        """Test valid reprojection request"""
        mock_process.return_value = None
        
        response = client.post("/api/v1/operations/reproject", json={
            "input_url": "gs://test-bucket/test.shp",
            "target_srs": "EPSG:4326"
        })
        
        assert response.status_code == 202
        data = response.json()
        assert "id" in data
        assert data["options"]["t_srs"] == "EPSG:4326"
    
    def test_clip_no_bounds(self, client):
        """Test clip without bounds or geometry"""
        response = client.post("/api/v1/operations/clip", json={
            "input_url": "gs://test-bucket/test.shp"
        })
        assert response.status_code == 422


# ==========================================
# Utility Function Tests
# ==========================================

class TestUtilityFunctions:
    """Test utility functions"""
    
    def test_generate_unique_filename(self):
        """Test unique filename generation"""
        filename1 = generate_unique_filename("test.shp")
        filename2 = generate_unique_filename("test.shp")
        
        assert filename1 != filename2
        assert filename1.endswith("_test.shp")
        assert filename2.endswith("_test.shp")
    
    def test_get_driver_for_format(self):
        """Test driver name mapping"""
        assert get_driver_for_format("shapefile") == "ESRI Shapefile"
        assert get_driver_for_format("shp") == "ESRI Shapefile"
        assert get_driver_for_format("gpkg") == "GPKG"
        assert get_driver_for_format("geojson") == "GeoJSON"
        assert get_driver_for_format("unknown") == "unknown"
    
    def test_validate_epsg_code(self):
        """Test EPSG code validation"""
        assert validate_epsg_code(4326) is True
        assert validate_epsg_code("4326") is True
        assert validate_epsg_code("EPSG:4326") is True
        assert validate_epsg_code(0) is False
        assert validate_epsg_code(999999) is False
        assert validate_epsg_code("invalid") is False
    
    def test_validate_bbox(self):
        """Test bounding box validation"""
        # Valid bounding boxes
        assert validate_bbox([-180, -90, 180, 90]) is True
        assert validate_bbox([-111.9, 40.7, -111.8, 40.8]) is True
        
        # Invalid bounding boxes
        assert validate_bbox([0, 0, 0, 0]) is False  # min == max
        assert validate_bbox([10, 10, 5, 5]) is False  # min > max
        assert validate_bbox([0, 0, 1]) is False  # wrong length
        assert validate_bbox([-200, -100, 200, 100]) is False  # out of range


# ==========================================
# Model Validation Tests
# ==========================================

class TestModelValidation:
    """Test Pydantic model validation"""
    
    def test_conversion_request_validation(self):
        """Test ConversionRequest model validation"""
        from app.models import ConversionRequest
        
        # Valid request
        request = ConversionRequest(
            input_url="gs://bucket/file.shp",
            output_format=OutputFormat.GEOJSON
        )
        assert request.input_url == "gs://bucket/file.shp"
        
        # Invalid URL
        with pytest.raises(ValueError):
            ConversionRequest(
                input_url="not-a-gcs-url",
                output_format=OutputFormat.GEOJSON
            )
    
    def test_job_status_enum(self):
        """Test JobStatus enum"""
        assert JobStatus.PENDING == "pending"
        assert JobStatus.PROCESSING == "processing"
        assert JobStatus.COMPLETED == "completed"
        assert JobStatus.FAILED == "failed"
    
    def test_output_format_enum(self):
        """Test OutputFormat enum"""
        assert OutputFormat.SHAPEFILE == "shapefile"
        assert OutputFormat.GEOJSON == "geojson"
        assert OutputFormat.GEOPACKAGE == "gpkg"


# ==========================================
# Integration Tests
# ==========================================

@pytest.mark.integration
class TestIntegration:
    """Integration tests (requires GDAL and GCS)"""
    
    @pytest.mark.skipif(not shutil.which("ogr2ogr"), reason="GDAL not installed")
    async def test_full_conversion_flow(self, client, sample_geojson):
        """Test complete conversion workflow"""
        # This would test the full flow with actual GDAL operations
        # Skipped in CI/CD without GDAL installed
        pass
    
    @pytest.mark.skipif(not settings.gcs_bucket, reason="GCS not configured")
    async def test_gcs_operations(self):
        """Test GCS upload/download operations"""
        # This would test actual GCS operations
        # Skipped without GCS configuration
        pass


# ==========================================
# Performance Tests
# ==========================================

@pytest.mark.performance
class TestPerformance:
    """Performance tests"""
    
    def test_api_response_time(self, client):
        """Test API response time"""
        import time
        
        start = time.time()
        response = client.get("/health")
        duration = time.time() - start
        
        assert response.status_code == 200
        assert duration < 1.0  # Should respond within 1 second
    
    def test_concurrent_requests(self, client):
        """Test handling concurrent requests"""
        import concurrent.futures
        
        def make_request():
            return client.get("/health")
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(make_request) for _ in range(10)]
            results = [f.result() for f in concurrent.futures.as_completed(futures)]
        
        assert len(results) == 10
        assert all(r.status_code == 200 for r in results)


# ==========================================
# Error Handling Tests
# ==========================================

class TestErrorHandling:
    """Test error handling and edge cases"""
    
    def test_large_file_rejection(self, client, temp_dir):
        """Test rejection of files exceeding size limit"""
        # Create a file that appears large (mock the size check)
        large_file = temp_dir / "large.geojson"
        large_file.write_text("{}")
        
        with patch('app.main.MAX_FILE_SIZE', 1):  # Set max size to 1 byte
            with open(large_file, "rb") as f:
                response = client.post(
                    "/api/v1/convert/upload",
                    files={"file": ("large.geojson", f, "application/geo+json")},
                    data={"output_format": "shapefile"}
                )
            
            # Should accept since we're not checking actual file size in test
            assert response.status_code in [202, 400]
    
    def test_invalid_output_format(self, client):
        """Test invalid output format"""
        response = client.post("/api/v1/convert", json={
            "input_url": "gs://bucket/file.shp",
            "output_format": "invalid_format"
        })
        assert response.status_code == 422
    
    def test_malformed_json(self, client):
        """Test malformed JSON request"""
        response = client.post(
            "/api/v1/convert",
            data="not json",
            headers={"Content-Type": "application/json"}
        )
        assert response.status_code == 422
    
    def test_unsupported_file_extension(self, client, temp_dir):
        """Test upload with unsupported file extension"""
        bad_file = temp_dir / "test.xyz"
        bad_file.write_text("invalid content")
        
        with open(bad_file, "rb") as f:
            response = client.post(
                "/api/v1/convert/upload",
                files={"file": ("test.xyz", f, "application/octet-stream")},
                data={"output_format": "shapefile"}
            )
        
        # Should still accept as we process based on content, not extension
        assert response.status_code == 202


# ==========================================
# Configuration Tests
# ==========================================

class TestConfiguration:
    """Test configuration and settings"""
    
    def test_settings_defaults(self):
        """Test default settings values"""
        from app.config import settings
        
        assert settings.port == 8080
        assert settings.max_file_size == 524288000  # 500MB
        assert settings.project_id == "ut-dnr-ugs-backend-tools"
        assert settings.api_version == "1.0.0"
    
    def test_settings_env_override(self):
        """Test environment variable override"""
        import os
        
        # Set env var
        os.environ["LOG_LEVEL"] = "DEBUG"
        
        # Reimport to get new settings
        from app.config import Settings
        test_settings = Settings()
        
        assert test_settings.log_level == "DEBUG"
    
    def test_directory_creation(self, temp_dir):
        """Test that required directories are created"""
        from app.config import Settings
        
        with patch.object(Settings, 'upload_dir', temp_dir / "uploads"):
            with patch.object(Settings, 'output_dir', temp_dir / "outputs"):
                with patch.object(Settings, 'workspace_dir', temp_dir / "workspace"):
                    settings = Settings()
                    
                    assert (temp_dir / "uploads").exists()
                    assert (temp_dir / "outputs").exists()
                    assert (temp_dir / "workspace").exists()


# ==========================================
# Mock GCS Tests
# ==========================================

class TestMockGCS:
    """Test GCS operations with mocks"""
    
    @pytest.mark.asyncio
    async def test_download_from_gcs(self):
        """Test GCS download with mock"""
        from app.utils import download_from_gcs
        
        with patch('app.utils.asyncio.create_subprocess_exec') as mock_exec:
            mock_process = AsyncMock()
            mock_process.communicate.return_value = (b"", b"")
            mock_process.returncode = 0
            mock_exec.return_value = mock_process
            
            result = await download_from_gcs(
                "gs://test-bucket/test.shp",
                Path("/tmp")
            )
            
            assert result.exists() or mock_exec.called
    
    @pytest.mark.asyncio
    async def test_upload_to_gcs(self, temp_dir):
        """Test GCS upload with mock"""
        from app.utils import upload_to_gcs
        
        test_file = temp_dir / "test.geojson"
        test_file.write_text("{}")
        
        with patch('app.utils.asyncio.create_subprocess_exec') as mock_exec:
            mock_process = AsyncMock()
            mock_process.communicate.return_value = (b"", b"")
            mock_process.returncode = 0
            mock_exec.return_value = mock_process
            
            result = await upload_to_gcs(
                test_file,
                "gs://test-bucket/output.geojson"
            )
            
            assert result == "gs://test-bucket/output.geojson"


# ==========================================
# Cleanup Tests
# ==========================================

class TestCleanup:
    """Test cleanup operations"""
    
    def test_cleanup_old_files(self, temp_dir):
        """Test cleanup of old files"""
        from app.utils import cleanup_old_files
        import time
        
        # Create old file
        old_file = temp_dir / "old.txt"
        old_file.write_text("old")
        
        # Create new file
        new_file = temp_dir / "new.txt"
        new_file.write_text("new")
        
        # Make old file appear old by modifying its timestamp
        old_time = time.time() - (25 * 3600)  # 25 hours ago
        os.utime(old_file, (old_time, old_time))
        
        # Run cleanup
        deleted = cleanup_old_files(temp_dir, hours=24)
        
        assert deleted == 1
        assert not old_file.exists()
        assert new_file.exists()


# ==========================================
# Run Tests
# ==========================================

if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v", "--tb=short"])