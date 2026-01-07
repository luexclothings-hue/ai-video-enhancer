# GCP Deployment Guide - Cost Optimized

**Monthly Cost: ~$50-80** (vs $400+ traditional deployment)

## Prerequisites

- GCP account with billing enabled
- $300 credits will last 5+ months

## Step 1: Create GCP Project

1. Go to https://console.cloud.google.com/
2. Click project dropdown → "New Project"
3. Name: `AI Video Enhancer`, ID: `video-enhancer-prod-2024`
4. Click "Create"

## Step 2: Enable APIs

Go to APIs & Services > Library and enable:

- Compute Engine API
- Cloud SQL Admin API
- Cloud Storage API
- Cloud Pub/Sub API
- Cloud Run API
- Container Registry API

## Step 3: Create Cloud SQL Database

1. Go to SQL > Create Instance > PostgreSQL
2. Configure:
   - Instance ID: `video-enhancer-db`
   - Password: (generate strong password)
   - Region: `us-central1`
   - Machine: `db-f1-micro` ($7/month)
   - Storage: `10 GB SSD`
3. Create database named `video_enhancer`

## Step 4: Create Storage Buckets

Create two buckets in Cloud Storage:

- `video-enhancer-raw-prod` (raw videos)
- `video-enhancer-enhanced-prod` (processed videos)
- Region: `us-central1`
- Storage class: `Standard`

## Step 5: Create Pub/Sub Topic

1. Go to Pub/Sub > Topics
2. Create topic: `video-jobs`
3. Create subscription: `video-jobs-subscription`

## Step 6: Create Service Account

1. Go to IAM & Admin > Service Accounts
2. Create account: `video-enhancer`
3. Add roles:
   - Cloud SQL Client
   - Storage Admin
   - Pub/Sub Editor
4. Create JSON key and download

## Step 7: Deploy API to Cloud Run

### Build Container:

Open Cloud Shell and run:

```bash
git clone https://github.com/luexclothings-hue/ai-video-enhancer.git
cd ai-video-enhancer

docker build -f backend/apps/api/Dockerfile.cloudrun -t gcr.io/$GOOGLE_CLOUD_PROJECT/video-enhancer-api:latest backend/apps/api/
gcloud auth configure-docker
docker push gcr.io/$GOOGLE_CLOUD_PROJECT/video-enhancer-api:latest
```

### Create Cloud Run Service:

1. Go to Cloud Run > Create Service
2. Container image: `gcr.io/YOUR_PROJECT_ID/video-enhancer-api:latest`
3. Service name: `video-enhancer-api`
4. Region: `us-central1`
5. Authentication: `Allow unauthenticated`
6. CPU: 1, Memory: 1 GiB
7. Min instances: 0, Max: 10

### Environment Variables:

Copy from `.env.production` file and update:

```env
NODE_ENV=production
DATABASE_URL=postgresql://postgres:YOUR_DB_PASSWORD@/video_enhancer?host=/cloudsql/YOUR_PROJECT_ID:us-central1:video-enhancer-db
JWT_SECRET=your-super-secure-jwt-secret
GCP_PROJECT_ID=your-project-id
GCS_BUCKET_VIDEOS_RAW=video-enhancer-raw-prod
GCS_BUCKET_VIDEOS_ENHANCED=video-enhancer-enhanced-prod
PUBSUB_TOPIC_VIDEO_JOBS=video-jobs
```

### Connect to Cloud SQL:

In Connections tab, add your Cloud SQL instance.

## Step 8: Create GPU Worker VM

1. Go to Compute Engine > Create Instance
2. Name: `video-enhancer-worker`
3. Region: `us-central1-a`
4. Machine: `n1-standard-4`
5. GPU: `NVIDIA Tesla T4` (1 GPU)
6. **VM provisioning**: `Spot` (70% cheaper!)
7. Boot disk: Ubuntu 20.04, 50GB
8. Add startup script from `backend/apps/worker/startup-script.sh`

## Step 9: Set Up Auto-Start Function

1. Go to Cloud Functions > Create
2. Name: `start-worker-vm`
3. Trigger: Pub/Sub topic `video-jobs`
4. Runtime: Python 3.9
5. Copy code from `cloud-functions/start-worker/main.py`
6. Update PROJECT_ID in the code

## Step 10: Test Deployment

1. Get Cloud Run URL from Cloud Run console
2. Test: `curl https://YOUR_CLOUD_RUN_URL/health`
3. Should return: `{"status":"ok"}`

## Cost Breakdown

- Cloud SQL f1-micro: $7/month
- Cloud Storage: $5/month
- Cloud Run: $10/month
- Spot GPU VM: $30/month (actual usage)
- Total: ~$52/month

## Your API URL

`https://video-enhancer-api-xxx-uc.a.run.app`

Done! Your cost-optimized AI video enhancer is deployed.
