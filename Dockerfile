# Dockerfile - Enhanced with Google Cloud SDK
FROM ghcr.io/osgeo/gdal:ubuntu-full-3.11.3

# Install Python, Google Cloud SDK, and dependencies
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-venv \
    curl \
    gnupg \
    lsb-release \
    && rm -rf /var/lib/apt/lists/*

# Install Google Cloud SDK
RUN echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] https://packages.cloud.google.com/apt cloud-sdk main" | tee -a /etc/apt/sources.list.d/google-cloud-sdk.list && \
    curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | apt-key --keyring /usr/share/keyrings/cloud.google.gpg add - && \
    apt-get update -y && \
    apt-get install google-cloud-cli -y && \
    rm -rf /var/lib/apt/lists/*

# Set up Python environment
WORKDIR /app

# Create and activate virtual environment
ENV VIRTUAL_ENV=/opt/venv
RUN python3 -m venv $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

# Install Python packages in virtual environment
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY app/ ./app/

# Create necessary directories
RUN mkdir -p /tmp/gdal-workspace /tmp/gdal-uploads /tmp/gdal-outputs

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PORT=8080
ENV GDAL_CACHEMAX=1024
ENV GDAL_NUM_THREADS=ALL_CPUS
ENV CPL_TMPDIR=/tmp/gdal-workspace

# Configure gsutil for better performance
ENV CLOUDSDK_CORE_DISABLE_USAGE_REPORTING=true
ENV CLOUDSDK_METRICS_ENVIRONMENT=container

EXPOSE 8080

# Run the application using virtual environment's Python
CMD ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080"]