# Use the GDAL image directly as base
FROM ghcr.io/osgeo/gdal:ubuntu-full-3.11.3

# Install Python and required packages
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-venv \
    python3-dev \
    ca-certificates \
    curl \
    proj-data \
    proj-bin \
    libproj-dev \
    && rm -rf /var/lib/apt/lists/*

# Update PROJ data
RUN projsync --system-directory --all

# Create non-root user
RUN useradd -m -u 1001 gdal-user

# Set up Python virtual environment
ENV VIRTUAL_ENV=/opt/venv
RUN python3 -m venv $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

# Upgrade pip
RUN pip install --upgrade pip

# Install GDAL Python bindings first (matching the system GDAL version)
RUN pip install --no-cache-dir GDAL==$(gdal-config --version)

# Copy and install requirements
COPY requirements.txt /tmp/
# Remove gdal from requirements if it's there, since we installed it above
RUN grep -v "gdal" /tmp/requirements.txt > /tmp/requirements_filtered.txt || cp /tmp/requirements.txt /tmp/requirements_filtered.txt
RUN pip install --no-cache-dir -r /tmp/requirements_filtered.txt && rm /tmp/requirements*.txt

# Copy application code
WORKDIR /app
COPY . .

# Create necessary directories
RUN mkdir -p /tmp/gdal-workspace /tmp/gdal-uploads /tmp/gdal-outputs && \
    chown -R gdal-user:gdal-user /tmp/gdal-workspace /tmp/gdal-uploads /tmp/gdal-outputs /app

# Switch to non-root user
USER gdal-user

# Set environment variables
ENV PORT=8080
ENV GDAL_DATA=/usr/share/gdal
ENV PROJ_LIB=/usr/share/proj
ENV CPL_TMPDIR=/tmp/gdal-workspace
ENV GDAL_CACHEMAX=512
ENV GDAL_MAX_DATASET_POOL_SIZE=10
ENV PYTHONUNBUFFERED=1

EXPOSE 8080

# Run the application
CMD ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080"]