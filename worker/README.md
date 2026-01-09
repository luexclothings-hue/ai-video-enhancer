# AI Video Enhancer - Worker

This is the AI processing worker. It is designed to run on **Google Cloud Run** with **GPU support** (NVIDIA L4).

## Architecture
- **Trigger**: Pub/Sub Push Subscription (POST `/process`)
- **Inputs**: JSON payload with `gcsRawPath`.
- **Logic**:
    1.  Downloads raw video from GCS.
    2.  Splits to frames (FFmpeg).
    3.  Runs Inference (Stream-DiffVSR).
    4.  Stitches frames.
    5.  Uploads result to GCS.

## Local Development (CPU Only)
1.  Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```
2.  Run server:
    ```bash
    python main.py
    ```
3.  Test with `curl` (Mimic Pub/Sub):
    ```bash
    curl -X POST http://localhost:8080/process \
         -H "Content-Type: application/json" \
         -d '{"message": {"data": "eyJqb2JJZCI6IDEyMywgImdjc1Jhd1BhdGgiOiAiZ3M6Ly9teS1idWNrZXQvdmlkZW8ubXA0In0="}}'
    ```
    *(Base64 payload = `{"jobId": 123, "gcsRawPath": "gs://my-bucket/video.mp4"}`)*

## Cloud Run Deployment (GPU)

### 1. Dockerfile
*Note: A `Dockerfile` is needed at the root of `worker/` to install CUDA/Drivers.*
(To be added in next phase when integrating the model).

### 2. Deploy Command
```bash
gcloud run deploy video-enhancer-worker \
  --source . \
  --region us-central1 \
  --no-allow-unauthenticated \
  --set-env-vars GCP_PROJECT_ID=[YOUR_PROJECT_ID] \
  --no-cpu-throttling \
  --gpu 1 \
  --gpu-type nvidia-l4
```
*Note: You must setup the Pub/Sub Push subscription to point to this service URL.*