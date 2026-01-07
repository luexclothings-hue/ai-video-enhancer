# Google Cloud Platform Deployment Guide

## AI Video Enhancer - Cost-Optimized Production Deployment

This guide will walk you through deploying your AI Video Enhancer to Google Cloud Platform using cost-optimized services perfect for small to medium scale (50-500 users/month).

## 📋 Prerequisites

- Google Cloud Platform account with billing enabled
- Domain name (optional, for custom domain)
- **Estimated monthly cost: $50-80** (vs $400-600 with traditional VMs)

## 💰 Cost Optimization Strategy

**Traditional Approach**: $400-600/month

- Always-on VMs
- Oversized database
- 24/7 GPU usage

**Our Optimized Approach**: $50-80/month

- Cloud Run for API (serverless, scales to zero)
- Micro database instance
- Preemptible GPU with auto-shutdown
- Pay-per-use model

---

## 🚀 Step 1: Create GCP Project

### Via Google Cloud Console:

1. **Go to Google Cloud Console**: https://console.cloud.google.com/
2. **Click the project dropdown** (top left, next to "Google Cloud")
3. **Click "New Project"**
4. **Fill in project details**:
   - **Project name**: `AI Video Enhancer`
   - **Project ID**: `video-enhancer-prod-2024` (must be globally unique)
   - **Organization**: Select your organization (if applicable)
5. **Click "Create"**
6. **Wait for project creation** (30-60 seconds)
7. **Select your new project** from the dropdown

---

## 🔌 Step 2: Enable Required APIs

### Via Google Cloud Console:

1. **Go to APIs & Services > Library**
2. **Search and enable these APIs** (click each, then click "Enable"):
   - **Compute Engine API**
   - **Cloud SQL Admin API**
   - **Cloud Storage API**
   - **Cloud Pub/Sub API**
   - **Cloud Resource Manager API**
   - **Cloud Build API** (for Docker builds)

**Note**: Each API takes 1-2 minutes to enable.

---

## 🗄️ Step 3: Set Up Cloud SQL Database (Cost-Optimized)

### Create Database Instance:

1. **Go to SQL > Overview**
2. **Click "Create Instance"**
3. **Choose "PostgreSQL"**
4. **Configure instance**:
   - **Instance ID**: `video-enhancer-db`
   - **Password**: Generate a strong password (save it!)
   - **Database version**: `PostgreSQL 15`
   - **Region**: `us-central1` (Iowa) - cost-effective
   - **Zone**: `us-central1-a`
   - **Machine type**: `db-f1-micro` (1 shared vCPU, 0.6 GB RAM) - **$7/month**
   - **Storage**: `10 GB SSD` with auto-increase enabled
   - **High availability**: **Disabled** (saves ~$100/month)
   - **Automated backups**: **Enabled** (7 days retention)
5. **Click "Create Instance"** (takes 5-10 minutes)

### Create Database:

1. **Go to your SQL instance** > **Databases**
2. **Click "Create Database"**
3. **Database name**: `video_enhancer`
4. **Click "Create"**

### Get Connection Details:

1. **Go to your SQL instance** > **Overview**
2. **Note the "Connection name"** (format: `project-id:region:instance-name`)
3. **Save this for later configuration**

---

## 🪣 Step 4: Create Cloud Storage Buckets

### Create Raw Videos Bucket:

1. **Go to Cloud Storage > Buckets**
2. **Click "Create Bucket"**
3. **Configure bucket**:
   - **Name**: `video-enhancer-raw-prod` (must be globally unique)
   - **Location type**: `Region`
   - **Location**: `us-central1`
   - **Storage class**: `Standard`
   - **Access control**: `Uniform`
   - **Protection tools**: Enable soft delete (7 days)
4. **Click "Create"**

### Create Enhanced Videos Bucket:

1. **Click "Create Bucket"** again
2. **Configure bucket**:
   - **Name**: `video-enhancer-enhanced-prod`
   - **Location type**: `Region`
   - **Location**: `us-central1`
   - **Storage class**: `Standard`
   - **Access control**: `Uniform`
   - **Protection tools**: Enable soft delete (7 days)
3. **Click "Create"**

### Set Lifecycle Policy (Optional):

1. **Go to each bucket** > **Lifecycle**
2. **Click "Add Rule"**
3. **Configure rule**:
   - **Action**: `Delete object`
   - **Condition**: `Age` = `30 days`
4. **Click "Create"**

---

## 📨 Step 5: Set Up Pub/Sub

### Create Topic:

1. **Go to Pub/Sub > Topics**
2. **Click "Create Topic"**
3. **Topic ID**: `video-jobs`
4. **Leave other settings as default**
5. **Click "Create"**

### Create Subscription:

1. **Go to Pub/Sub > Subscriptions**
2. **Click "Create Subscription"**
3. **Configure subscription**:
   - **Subscription ID**: `video-jobs-subscription`
   - **Topic**: Select `video-jobs`
   - **Delivery type**: `Pull`
   - **Acknowledgment deadline**: `600 seconds` (10 minutes)
   - **Message retention duration**: `7 days`
4. **Click "Create"**

---

## 🔐 Step 6: Create Service Account

### Create Service Account:

1. **Go to IAM & Admin > Service Accounts**
2. **Click "Create Service Account"**
3. **Configure account**:
   - **Service account name**: `video-enhancer`
   - **Service account ID**: `video-enhancer` (auto-filled)
   - **Description**: `Service account for AI Video Enhancer application`
4. **Click "Create and Continue"**

### Grant Roles:

1. **Add these roles** (click "Add Another Role" for each):
   - `Cloud SQL Client`
   - `Storage Admin`
   - `Pub/Sub Editor`
   - `Compute Instance Admin (v1)` (if using auto-scaling)
2. **Click "Continue"**
3. **Skip "Grant users access"**
4. **Click "Done"**

### Create Key:

1. **Click on your service account**
2. **Go to "Keys" tab**
3. **Click "Add Key" > "Create New Key"**
4. **Choose "JSON"**
5. **Click "Create"**
6. **Save the downloaded JSON file** as `gcp-service-account.json`

---

## 💻 Step 7: Deploy API Server to Cloud Run (Serverless)

### Prepare API for Cloud Run:

1. **Go to Cloud Run > Services**
2. **Click "Create Service"**
3. **Choose "Deploy one revision from an existing container image"**
4. **We'll build and push the image first**

### Build and Push Container Image:

1. **Go to Cloud Build > History**
2. **We'll use Cloud Build to build our image**

**First, let's prepare the API code for Cloud Run:**

### Create Cloud Run Dockerfile:

Create `backend/apps/api/Dockerfile.cloudrun`:

```dockerfile
FROM node:20-alpine

WORKDIR /app

# Copy package files
COPY package*.json ./
RUN npm ci --only=production

# Copy source code
COPY . .

# Generate Prisma client
RUN npx prisma generate

# Build the application
RUN npm run build

# Expose port (Cloud Run uses PORT env var)
EXPOSE $PORT

# Start the application
CMD ["npm", "start"]
```

### Deploy to Cloud Run:

1. **Go to Cloud Run > Services**
2. **Click "Create Service"**
3. **Configure service**:
   - **Service name**: `video-enhancer-api`
   - **Region**: `us-central1`
   - **CPU allocation**: `CPU is only allocated during request processing`
   - **Minimum instances**: `0` (scales to zero)
   - **Maximum instances**: `10`
   - **CPU**: `1`
   - **Memory**: `1 GiB`
   - **Request timeout**: `300 seconds`
   - **Maximum concurrent requests**: `80`

### Set Environment Variables:

In the **Variables & Secrets** tab, add:

```env
NODE_ENV=production
DATABASE_URL=postgresql://postgres:YOUR_PASSWORD@/video_enhancer?host=/cloudsql/YOUR_PROJECT:us-central1:video-enhancer-db
JWT_SECRET=your-super-secure-jwt-secret-key
GCP_PROJECT_ID=your-project-id
GCS_BUCKET_VIDEOS_RAW=video-enhancer-raw-prod
GCS_BUCKET_VIDEOS_ENHANCED=video-enhancer-enhanced-prod
PUBSUB_TOPIC_VIDEO_JOBS=video-jobs
```

### Connect to Cloud SQL:

1. **In Cloud Run service configuration**
2. **Go to "Connections" tab**
3. **Click "Add Connection"**
4. **Select your Cloud SQL instance**
5. **This automatically handles the connection**

6. **Click "Create"**

---

## 🎮 Step 8: Create Cost-Optimized GPU Worker VM

### Create Preemptible GPU VM Instance:

1. **Go to Compute Engine > VM Instances**
2. **Click "Create Instance"**
3. **Configure VM**:
   - **Name**: `video-enhancer-worker`
   - **Region**: `us-central1`
   - **Zone**: `us-central1-a`
   - **Machine configuration**:
     - **Series**: `N1`
     - **Machine type**: `n1-standard-4` (4 vCPUs, 15 GB RAM)
   - **GPUs**: **Click "Add GPU"**
     - **GPU type**: `NVIDIA Tesla T4` (best cost/performance)
     - **Number of GPUs**: `1`
   - **Availability policies**:
     - **VM provisioning model**: `Spot` (70% cheaper!)
     - **On VM preemption**: `Stop` (don't delete)
   - **Boot disk**:
     - **Operating system**: `Ubuntu`
     - **Version**: `Ubuntu 20.04 LTS`
     - **Boot disk type**: `Standard persistent disk`
     - **Size**: `50 GB` (reduced from 100GB)
   - **Advanced options > Management**:
     - **Metadata**: Add key-value pairs:
       - **Key**: `install-nvidia-driver`, **Value**: `True`
       - **Key**: `enable-oslogin`, **Value**: `TRUE`
   - **Advanced options > Management > Automation**:
     - **Startup script**: Add auto-shutdown script:

```bash
#!/bin/bash
# Auto-shutdown after 2 hours of inactivity
echo "*/30 * * * * /opt/check-activity.sh" | crontab -

# Create activity check script
cat > /opt/check-activity.sh << 'EOF'
#!/bin/bash
# Check if worker is processing jobs
if ! docker logs worker --since=30m 2>/dev/null | grep -q "Processing video"; then
    # No activity in 30 minutes, check if any jobs in queue
    JOBS=$(gcloud pubsub subscriptions pull video-jobs-subscription --limit=1 --format="value(message.data)" 2>/dev/null | wc -l)
    if [ "$JOBS" -eq 0 ]; then
        echo "No jobs in queue, shutting down to save costs"
        sudo shutdown -h now
    fi
fi
EOF
chmod +x /opt/check-activity.sh
```

### Cost Comparison:

| Configuration    | Monthly Cost | When to Use                   |
| ---------------- | ------------ | ----------------------------- |
| **Spot VM**      | ~$105        | **Recommended** - 70% savings |
| **Regular VM**   | ~$350        | High availability needed      |
| **Actual Usage** | ~$20-40      | With auto-shutdown (our goal) |

4. **Click "Create"** (takes 3-5 minutes)

### Set Up Auto-Start Trigger:

Create a Cloud Function to start the worker when jobs arrive:

1. **Go to Cloud Functions > Create Function**
2. **Configure function**:
   - **Function name**: `start-worker-vm`
   - **Region**: `us-central1`
   - **Trigger type**: `Pub/Sub`
   - **Topic**: `video-jobs`
3. **Runtime**: `Python 3.9`
4. **Source code**:

```python
import functions_framework
from google.cloud import compute_v1

@functions_framework.cloud_event
def start_worker(cloud_event):
    compute_client = compute_v1.InstancesClient()
    project = 'your-project-id'
    zone = 'us-central1-a'
    instance = 'video-enhancer-worker'

    try:
        # Start the instance if it's stopped
        operation = compute_client.start(
            project=project,
            zone=zone,
            instance=instance
        )
        print(f"Starting worker VM: {operation.name}")
    except Exception as e:
        print(f"Worker already running or error: {e}")
```

5. **Deploy the function**

---

## 🔥 Step 9: Configure Firewall Rules (Not Needed for Cloud Run)

**Note**: Cloud Run handles HTTPS automatically, so we don't need custom firewall rules for the API. We only need rules for the GPU worker if you want to access it directly.

### Optional: Create Firewall Rule for Worker SSH:

1. **Go to VPC Network > Firewall**
2. **Click "Create Firewall Rule"**
3. **Configure rule**:
   - **Name**: `allow-worker-ssh`
   - **Direction**: `Ingress`
   - **Action**: `Allow`
   - **Targets**: `Specified target tags`
   - **Target tags**: `video-enhancer-worker`
   - **Source IP ranges**: `0.0.0.0/0`
   - **Protocols and ports**:
     - ✅ **TCP** - Ports: `22`
4. **Click "Create"**

---

## 🚀 Step 10: Deploy Application

### Deploy API to Cloud Run:

**Option A: Using Cloud Build (Recommended)**

1. **Connect your GitHub repository**:
   - Go to **Cloud Build > Triggers**
   - Click **"Create Trigger"**
   - Connect your GitHub repository
   - Configure trigger for `main` branch

2. **Create cloudbuild.yaml** in your repository root:

```yaml
steps:
  # Build API container
  - name: 'gcr.io/cloud-builders/docker'
    args: ['build', '-t', 'gcr.io/$PROJECT_ID/video-enhancer-api', './backend/apps/api']

  # Push to Container Registry
  - name: 'gcr.io/cloud-builders/docker'
    args: ['push', 'gcr.io/$PROJECT_ID/video-enhancer-api']

  # Deploy to Cloud Run
  - name: 'gcr.io/cloud-builders/gcloud'
    args:
      - 'run'
      - 'deploy'
      - 'video-enhancer-api'
      - '--image'
      - 'gcr.io/$PROJECT_ID/video-enhancer-api'
      - '--region'
      - 'us-central1'
      - '--platform'
      - 'managed'
      - '--allow-unauthenticated'
```

**Option B: Manual Deployment**

1. **Build and push locally**:

```bash
# Build the image
cd backend/apps/api
docker build -t gcr.io/YOUR_PROJECT_ID/video-enhancer-api .

# Push to Google Container Registry
docker push gcr.io/YOUR_PROJECT_ID/video-enhancer-api

# Deploy to Cloud Run
gcloud run deploy video-enhancer-api \
  --image gcr.io/YOUR_PROJECT_ID/video-enhancer-api \
  --region us-central1 \
  --platform managed \
  --allow-unauthenticated
```

### Deploy GPU Worker:

1. **SSH into Worker VM**:
   - Go to **Compute Engine > VM Instances**
   - Click **SSH** next to `video-enhancer-worker`

2. **Install NVIDIA Docker**:

```bash
# Install Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh

# Install NVIDIA Container Toolkit
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list

sudo apt-get update && sudo apt-get install -y nvidia-docker2
sudo systemctl restart docker

# Test GPU access
sudo docker run --rm --gpus all nvidia/cuda:11.8-base-ubuntu22.04 nvidia-smi
```

3. **Deploy Worker**:

```bash
# Clone repository
cd /opt/video-enhancer
git clone YOUR_REPOSITORY_URL .

# Set up environment
cd backend/apps/worker
cp .env.example .env
nano .env
```

**Worker Environment** (`.env`):

```env
DATABASE_URL=postgresql://postgres:YOUR_PASSWORD@/video_enhancer?host=/cloudsql/YOUR_PROJECT:us-central1:video-enhancer-db
GCP_PROJECT_ID=your-project-id
GCS_BUCKET_VIDEOS_RAW=video-enhancer-raw-prod
GCS_BUCKET_VIDEOS_ENHANCED=video-enhancer-enhanced-prod
PUBSUB_SUBSCRIPTION_ID=video-jobs-subscription
GOOGLE_APPLICATION_CREDENTIALS=/opt/video-enhancer/gcp-service-account.json
DEVICE=cuda
BATCH_SIZE=8
ENABLE_TENSORRT=false
TARGET_HEIGHT=720
TARGET_WIDTH=1280
```

```bash
# Build and run
docker build -t video-enhancer-worker .
docker run -d --name worker --restart unless-stopped --gpus all \
  -v /opt/video-enhancer/gcp-service-account.json:/app/gcp-service-account.json:ro \
  video-enhancer-worker
```

---

## 🌐 Step 11: Get Your API URL

### Find Your Cloud Run Service URL:

1. **Go to Cloud Run > Services**
2. **Click on `video-enhancer-api`**
3. **Copy the URL** (e.g., `https://video-enhancer-api-xxx-uc.a.run.app`)
4. **Test API**: Visit `https://YOUR_CLOUD_RUN_URL/health`

### Configure Custom Domain (Optional):

1. **Go to Cloud Run > Manage Custom Domains**
2. **Click "Add Mapping"**
3. **Select your service**
4. **Enter your domain** (e.g., `api.yourdomain.com`)
5. **Follow DNS configuration instructions**

---

## 🔍 Step 12: Verify Deployment

### Check API Health:

```bash
curl https://YOUR_CLOUD_RUN_URL/health
```

**Expected Response**:

```json
{
  "status": "ok",
  "timestamp": "2024-01-07T...",
  "database": "connected",
  "version": "1.0.0"
}
```

### Check Worker Status:

```bash
# SSH into worker VM (when it's running)
gcloud compute ssh video-enhancer-worker --zone=us-central1-a

# Check if worker is running
sudo docker logs worker -f
```

**Expected Output**:

```
Loading Stream-DiffVSR model from Jamichsu/Stream-DiffVSR...
Model loaded successfully on cuda
Worker listening for jobs on video-jobs-subscription...
```

### Test Auto-Start Functionality:

1. **Upload a test video** via your API
2. **Check Cloud Function logs** to see if worker started
3. **Monitor worker processing** in logs
4. **Verify auto-shutdown** after processing completes

---

## 💰 Cost Breakdown (Optimized)

| Service            | Configuration      | Monthly Cost   | Savings        |
| ------------------ | ------------------ | -------------- | -------------- |
| **Cloud SQL**      | db-f1-micro        | $7             | $18 saved      |
| **Cloud Storage**  | 200GB Standard     | $5             | $5 saved       |
| **Pub/Sub**        | 100K messages      | $0.40          | $0.60 saved    |
| **Cloud Run API**  | 100K requests      | $10            | $40 saved      |
| **Spot GPU VM**    | n1-standard-4 + T4 | $30\*          | $320 saved     |
| **Cloud Function** | VM auto-start      | $1             | New            |
| **Network**        | Egress traffic     | $5             | $10 saved      |
| **Total**          |                    | **~$58/month** | **$393 saved** |

\*Actual usage with auto-shutdown. Spot pricing is $105/month if running 24/7.

### Cost Optimization Features:

✅ **Cloud Run scales to zero** - No cost when not used  
✅ **Spot GPU VM** - 70% cheaper than regular VMs  
✅ **Auto-shutdown** - GPU only runs when processing  
✅ **Smaller database** - Right-sized for your scale  
✅ **Lifecycle policies** - Auto-delete old files

### Usage Estimates for 50 Users/Month:

- **API requests**: ~10,000/month
- **GPU processing**: ~20 hours/month
- **Storage**: ~100GB videos
- **Actual cost**: $35-50/month

---

## 🔧 Step 13: Set Up Budget Alerts & Monitoring

### Create Budget Alert:

1. **Go to Billing > Budgets & Alerts**
2. **Click "Create Budget"**
3. **Configure budget**:
   - **Name**: `Video Enhancer Monthly Budget`
   - **Budget amount**: `$80` (20% buffer over expected $60)
   - **Alert thresholds**: `50%, 90%, 100%`
   - **Email notifications**: Your email
4. **Click "Finish"**

### Set Up Monitoring:

1. **Go to Monitoring > Overview**
2. **Create a workspace** (if prompted)
3. **Set up alerts for**:
   - Cloud Run error rate > 5%
   - Cloud SQL CPU > 80%
   - GPU VM memory > 90%
   - Storage usage > 80%

### Enable Cost Optimization:

1. **Go to Compute Engine > VM Instances**
2. **Click on worker VM**
3. **Click "Edit"**
4. **Under "Management"**:
   - **Preemptibility**: `On` (if not already set)
   - **Automatic restart**: `Off` (for spot instances)
   - **On host maintenance**: `Terminate VM instance`

---

## 🎯 Next Steps

### For Production:

1. **Set up a domain name** and SSL certificate
2. **Configure load balancer** for high availability
3. **Set up automated backups**
4. **Implement monitoring and alerting**
5. **Set up CI/CD pipeline** for deployments

### For Testing:

1. **Import Postman collection** from `docs/postman_collection.json`
2. **Import Postman environment** from `docs/postman_environments.json`
3. **Update environment variables** with your API IP
4. **Test all endpoints**

---

## 🆘 Troubleshooting

### Common Issues:

**API won't start:**

- Check database connection string
- Verify service account permissions
- Check firewall rules

**Worker not processing jobs:**

- Verify GPU drivers: `nvidia-smi`
- Check Pub/Sub subscription
- Monitor worker logs: `docker logs worker -f`

**High costs:**

- Check for stuck VMs
- Review storage lifecycle policies
- Monitor Pub/Sub message retention

### Support Resources:

- **Google Cloud Documentation**: https://cloud.google.com/docs
- **GPU Troubleshooting**: https://cloud.google.com/compute/docs/gpus
- **Cloud SQL Connection**: https://cloud.google.com/sql/docs/postgres/connect-compute-engine

---

**🎉 Congratulations! Your Cost-Optimized AI Video Enhancer is now deployed on Google Cloud Platform!**

Your API is available at: `https://YOUR_CLOUD_RUN_URL`  
API Documentation: `https://YOUR_CLOUD_RUN_URL/documentation`  
**Monthly Cost**: ~$50-80 (vs $400-600 traditional deployment)  
**Credits Usage**: 5+ months from your $300 credits!
