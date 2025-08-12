"""Test that all dependencies are installed correctly"""

import sys
print(f"Python version: {sys.version}")

try:
    import pydantic
    print(f"✅ Pydantic {pydantic.VERSION} installed")
except ImportError as e:
    print(f"❌ Pydantic failed: {e}")

try:
    import fastapi
    print(f"✅ FastAPI {fastapi.__version__} installed")
except ImportError as e:
    print(f"❌ FastAPI failed: {e}")

try:
    from osgeo import gdal
    print(f"✅ GDAL {gdal.__version__} installed")
except ImportError as e:
    print(f"❌ GDAL failed: {e}")

try:
    from google.cloud import storage
    print("✅ Google Cloud Storage installed")
except ImportError as e:
    print(f"❌ Google Cloud Storage failed: {e}")

print("\n🎉 Setup complete! You can now run: uvicorn app.main:app --reload")