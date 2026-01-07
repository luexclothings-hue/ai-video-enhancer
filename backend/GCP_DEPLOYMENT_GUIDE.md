# Google Cloud Platform Deployment Guide

## AI Video Enhancer - Production Deployment

**Monthly Cost: ~$50-80** (optimized for small scale)

---

## 🚀 Quick Setup

### Prerequisites

- GCP account with billing enabled
- $300 credits will last 5+ months

### Step 1: Create GCP Project

1. Go to https://console.cloud.google.com/
2. Click project dropdown → "New Project"
3. Name: `AI Video Enhancer`
4. Click "Create"

### Step 2: Enable APIs

Go to APIs & Services > Library and enable:

- Cloud Run API
- Cloud SQL Admin API
- Cloud Storage API
- Cloud Pub/Sub API
- Container Registry API

### Step 3: Create Cloud SQL Database

1. Go to SQL > Create Instance > PostgreSQL
2. Configure:
   - Instance ID: `video-enhancer-db`
   - Password: (save this password!)
   - Region: `us-central1`
   - Machine: `db-f1-micro` ($7/month)
   - Storage: `10 GB SSD`
3. Create database named `video_enhancer`

### Step 4: Create Storage Buckets

Create two buckets in Cloud Storage:

1. **Raw Videos Bucket**:
   - Name: `video-enhancer-raw-prod`
   - Region: `us-central1`
   - Storage class: `Standard`
   - **Lifecycle**: Delete objects after 30 days

2. **Enhanced Videos Bucket**:
   - Name: `video-enhancer-enhanced-prod`
   - Region: `us-central1`
   - Storage class: `Standard`
   - **Lifecycle**: Delete objects after 90 days

**Set Lifecycle Policies**:

1. Go to each bucket → Lifecycle tab
2. Add rule: Delete objects older than 30 days (raw) / 90 days (enhanced)
3. This saves storage costs automatically

### Step 5: Create Pub/Sub Topic

1. Go to Pub/Sub > Topics
2. Create topic: `video-jobs`
3. Create subscription: `video-jobs-subscription`

### Step 6: Create Service Account

1. Go to IAM & Admin > Service Accounts
2. Create account: `video-enhancer`
3. Add roles:
   - Cloud SQL Client
   - Storage Admin
   - Pub/Sub Editor
4. Create JSON key and download it

---

## 🐳 Deploy API to Cloud Run

### Build and Deploy:

1. **Open Cloud Shell** in GCP Console

2. **Run these commands**:

```bash
# Clone repository
git clone https://github.com/luexclothings-hue/ai-video-enhancer.git
cd ai-video-enhancer

# Build using Cloud Build (simpler and more reliable)
gcloud builds submit --tag gcr.io/$GOOGLE_CLOUD_PROJECT/video-enhancer-api backend/apps/api
```

### Create Cloud Run Service:

1. Go to **Cloud Run > Create Service**
2. Fill in:
   - **Container image URL**: `gcr.io/YOUR_PROJECT_ID/video-enhancer-api:latest`
   - **Service name**: `video-enhancer-api`
   - **Region**: `us-central1`
   - **Authentication**: `Allow unauthenticated invocations`
   - **CPU**: 1, **Memory**: 1 GiB
   - **Min instances**: 0, **Max instances**: 10

### Environment Variables:

In **Variables & Secrets** tab, add these (update YOUR\_\* values):

```
NODE_ENV=production
DATABASE_URL=postgresql://postgres:YOUR_DB_PASSWORD@/video_enhancer?host=/cloudsql/YOUR_PROJECT_ID:us-central1:video-enhancer-db
JWT_SECRET=your-super-secure-jwt-secret-key-change-this
GCP_PROJECT_ID=YOUR_PROJECT_ID
GCS_BUCKET_VIDEOS_RAW=video-enhancer-raw-prod
GCS_BUCKET_VIDEOS_ENHANCED=video-enhancer-enhanced-prod
PUBSUB_TOPIC_VIDEO_JOBS=video-jobs
```

### Connect to Database:

1. In **Connections** tab
2. Click **Add Connection**
3. Select your Cloud SQL instance
4. Click **Create**

---

## 🎮 Deploy GPU Worker (Optional - for processing)

### Create GPU VM:

1. Go to **Compute Engine > Create Instance**
2. Configure:
   - Name: `video-enhancer-worker`
   - Region: `us-central1-a`
   - Machine: `n1-standard-4`
   - GPU: `NVIDIA Tesla T4` (1 GPU)
   - **VM provisioning**: `Spot` (70% cheaper!)
   - Boot disk: Ubuntu 20.04, 50GB

### Setup Worker:

SSH into the VM and run:

```bash
# Install Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh

# Install NVIDIA Docker
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list
sudo apt-get update && sudo apt-get install -y nvidia-docker2
sudo systemctl restart docker

# Clone and build worker
git clone https://github.com/luexclothings-hue/ai-video-enhancer.git
cd ai-video-enhancer/backend/apps/worker
cp .env.example .env
# Edit .env with your settings
sudo docker build -t video-enhancer-worker .
sudo docker run -d --name worker --restart unless-stopped --gpus all video-enhancer-worker
```

---

## ✅ Test Your Deployment

1. **Get your API URL** from Cloud Run console
2. **Test health endpoint**:
   ```bash
   curl https://YOUR_CLOUD_RUN_URL/health
   ```
3. **Expected response**:
   ```json
   { "status": "ok", "database": "connected" }
   ```

---

## 💰 Cost Breakdown

| Service       | Configuration  | Monthly Cost   |
| ------------- | -------------- | -------------- |
| Cloud SQL     | db-f1-micro    | $7             |
| Cloud Storage | 100GB          | $3             |
| Cloud Run     | 50K requests   | $5             |
| Pub/Sub       | 10K messages   | $0.40          |
| GPU VM (Spot) | 20 hours/month | $30            |
| **Total**     |                | **~$45/month** |

---

## 🔄 How It Works

### Request Flow:

1. **User uploads video** → Cloud Run API
2. **API stores video** → Cloud Storage
3. **API creates job** → Pub/Sub queue
4. **GPU worker processes** → Enhanced video
5. **User downloads result** → Cloud Storage

### Architecture:

```
Frontend → Cloud Run API → Cloud SQL (metadata)
                        → Cloud Storage (videos)
                        → Pub/Sub → GPU Worker
```

---

## 🎯 Your Deployed API

**API URL**: `https://video-enhancer-api-xxx-uc.a.run.app`  
**Documentation**: `https://YOUR_API_URL/documentation`  
**Health Check**: `https://YOUR_API_URL/health`

**Done!** Your AI video enhancer is deployed and ready to use.
