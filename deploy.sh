# deploy.sh - Deployment script for GDAL API on Cloud Run (Python version)

set -e

# Configuration
PROJECT_ID=${PROJECT_ID:-"ut-dnr-ugs-backend-tools"}
REGION=${REGION:-"us-central1"}
SERVICE_NAME="gdal-api"
IMAGE_NAME="gcr.io/${PROJECT_ID}/${SERVICE_NAME}"

echo "🚀 Deploying GDAL API (Python) to Cloud Run"
echo "Project: ${PROJECT_ID}"
echo "Region: ${REGION}"

# Step 1: Enable required APIs
echo "📦 Enabling required GCP APIs..."
gcloud services enable \
    run.googleapis.com \
    cloudbuild.googleapis.com \
    storage.googleapis.com \
    iap.googleapis.com \
    secretmanager.googleapis.com \
    --project=${PROJECT_ID}

# Step 2: Create service account
echo "👤 Creating service account..."
gcloud iam service-accounts create gdal-api-sa \
    --display-name="GDAL API Service Account" \
    --project=${PROJECT_ID} || echo "Service account already exists"

# Step 3: Grant necessary permissions
echo "🔐 Granting IAM permissions..."
gcloud projects add-iam-policy-binding ${PROJECT_ID} \
    --member="serviceAccount:gdal-api-sa@${PROJECT_ID}.iam.gserviceaccount.com" \
    --role="roles/storage.objectAdmin"

gcloud projects add-iam-policy-binding ${PROJECT_ID} \
    --member="serviceAccount:gdal-api-sa@${PROJECT_ID}.iam.gserviceaccount.com" \
    --role="roles/logging.logWriter"

# Step 4: Create GCS bucket for workspace
echo "🪣 Creating GCS bucket..."
gsutil mb -p ${PROJECT_ID} -l ${REGION} gs://${PROJECT_ID}-gdal-workspace || echo "Bucket already exists"

# Set lifecycle policy
cat > lifecycle.json <<EOF
{
  "lifecycle": {
    "rule": [
      {
        "action": {"type": "Delete"},
        "condition": {"age": 1}
      }
    ]
  }
}
EOF
gsutil lifecycle set lifecycle.json gs://${PROJECT_ID}-gdal-workspace
rm lifecycle.json

# Step 5: Build and push Docker image
echo "🐳 Building Docker image..."
docker build -t ${IMAGE_NAME}:latest .

echo "📤 Pushing Docker image to GCR..."
docker push ${IMAGE_NAME}:latest

# Step 6: Deploy to Cloud Run
echo "☁️ Deploying to Cloud Run..."
gcloud run deploy ${SERVICE_NAME} \
    --image=${IMAGE_NAME}:latest \
    --platform=managed \
    --region=${REGION} \
    --memory=4Gi \
    --cpu=2 \
    --timeout=3600 \
    --max-instances=10 \
    --min-instances=0 \
    --concurrency=10 \
    --service-account=gdal-api-sa@${PROJECT_ID}.iam.gserviceaccount.com \
    --set-env-vars="GCS_BUCKET=${PROJECT_ID}-gdal-workspace" \
    --set-env-vars="MAX_FILE_SIZE=524288000" \
    --set-env-vars="GDAL_CACHEMAX=1024" \
    --no-allow-unauthenticated \
    --project=${PROJECT_ID}

# Step 7: Configure IAP (if needed)
echo "🔒 Configuring Identity-Aware Proxy..."
SERVICE_URL=$(gcloud run services describe ${SERVICE_NAME} --region=${REGION} --format='value(status.url)' --project=${PROJECT_ID})

echo "✅ Deployment complete!"
echo "Service URL: ${SERVICE_URL}"
echo ""
echo "Next steps:"
echo "1. Configure IAP through the GCP Console"
echo "2. Add authorized users/groups to IAP"
echo "3. Test the API endpoints"

# Testing commands
echo ""
echo "📝 Test commands:"
echo "# Health check"
echo "curl ${SERVICE_URL}/health"
echo ""
echo "# List formats"
echo "curl ${SERVICE_URL}/api/v1/formats"
echo ""
echo "# Convert file from GCS"
echo "curl -X POST ${SERVICE_URL}/api/v1/convert \\"
echo "  -H 'Content-Type: application/json' \\"
echo "  -d '{"
echo '    "input_url": "gs://your-bucket/input.shp",'
echo '    "output_format": "geojson",'
echo '    "output_bucket": "your-bucket",'
echo '    "output_path": "output/result.geojson"'
echo "  }'"