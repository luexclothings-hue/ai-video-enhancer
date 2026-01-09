# AI Video Enhancer - GPU Worker

This worker receives video jobs from Pub/Sub, processes them using **Stream-DiffVSR** on a GPU, and uploads the results.

## 📋 Prerequisites
1.  **Google Cloud Project** with billing enabled.
2.  Enable APIs:
    *   Cloud Run Admin API
    *   Artifact Registry API
    *   Compute Engine API (for GPU quotas)
3.  **GPU Quota**: Ensure you have quota for `NVIDIA L4 GPUs` in `us-central1` (or your chosen region).

## 🚀 Deployment Guide

### 1. Build & Push Docker Image
We need to build the image (which includes CUDA drivers) and store it in Google Artifact Registry.

```bash
# 1. Create a repository (if you haven't already for the backend)
gcloud artifacts repositories create video-enhancer-repo \
    --repository-format=docker \
    --location=us-central1 \
    --description="Docker repository for Video Enhancer"

# 2. Build and Submit the build to Cloud Build
# (This builds the image in the cloud so you don't need to download 5GB nvidia images locally)
gcloud builds submit --tag us-central1-docker.pkg.dev/[PROJECT_ID]/video-enhancer-repo/worker:latest .
```

### 2. Deploy to Cloud Run (with GPU)
Replace the placeholders `[PROJECT_ID]`, `[DB_PASS]`, `[INSTANCE_CONNECTION_NAME]`, etc.

```bash
gcloud run deploy video-enhancer-worker \
  --image us-central1-docker.pkg.dev/[PROJECT_ID]/video-enhancer-repo/worker:latest \
  --region us-central1 \
  --no-allow-unauthenticated \
  --service-account [YOUR_SERVICE_ACCOUNT_EMAIL] \
  --set-env-vars GCP_PROJECT_ID=[PROJECT_ID] \
  --set-env-vars GCS_BUCKET_ENHANCED=[ENHANCED_BUCKET_NAME] \
  --set-env-vars DATABASE_URL="postgresql://postgres:[DB_PASS]@/video_enhancer?host=/cloudsql/[INSTANCE_CONNECTION_NAME]" \
  --add-cloudsql-instances [INSTANCE_CONNECTION_NAME] \
  --cpu 4 \
  --memory 16Gi \
  --no-cpu-throttling \
  --gpu 1 \
  --gpu-type nvidia-l4 \
  --timeout 3600
```
*   **Timeout**: Set to 3600s (60 mins) to allow long video processing.
*   **GPU**: We request 1 NVIDIA L4.
*   **Memory**: 16GB is recommended for AI models.

### 3. Configure Pub/Sub Trigger
Now we need to tell Pub/Sub to send messages to this worker.

1.  **Get Worker URL**: Copy the URL from the deployment output (e.g., `https://video-enhancer-worker-xyz.a.run.app`).
2.  **Create Subscription**:
    ```bash
    gcloud pubsub subscriptions create video-enhancement-sub \
      --topic video-enhancement-jobs \
      --push-endpoint=[WORKER_URL]/process \
      --push-auth-service-account=[YOUR_SERVICE_ACCOUNT_EMAIL] \
      --ack-deadline=600
    ```
    *   `--ack-deadline`: 600s (10 mins) or more. This tells Pub/Sub "Wait this long for a response before assuming failure". ideally set closer to 60 mins.

## 🛠️ Testing
1.  Upload a video via the Backend API.
2.  Check the **Cloud Run Logs** for the worker.
3.  You should see:
    *   `Received Job ID: ...`
    *   `Video downloaded...`
    *   `Processing frame 1/XXX...`
    *   `Job Completed Successfully.`