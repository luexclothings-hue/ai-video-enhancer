# Google Cloud Platform Deployment Guide
## AI Video Enhancer - Production Deployment

This guide will walk you through deploying your AI Video Enhancer to Google Cloud Platform using the web console (UI) with specific GPU recommendations and settings.

## 📋 Prerequisites

- Google Cloud Platform account with billing enabled
- Domain name (optional, for custom domain)
- Estimated monthly cost: $400-600 (with GPU VM running 24/7)

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

## 🗄️ Step 3: Set Up Cloud SQL Database

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
   - **Machine type**: `db-g1-small` (1 vCPU, 1.7 GB RAM) - $25/month
   - **Storage**: `20 GB SSD` with auto-increase enabled
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

## 💻 Step 7: Create API Server VM

### Create VM Instance:

1. **Go to Compute Engine > VM Instances**
2. **Click "Create Instance"**
3. **Configure VM**:
   - **Name**: `video-enhancer-api`
   - **Region**: `us-central1`
   - **Zone**: `us-central1-a`
   - **Machine configuration**: 
     - **Series**: `N1`
     - **Machine type**: `n1-standard-2` (2 vCPUs, 7.5 GB RAM) - $50/month
   - **Boot disk**:
     - **Operating system**: `Ubuntu`
     - **Version**: `Ubuntu 20.04 LTS`
     - **Boot disk type**: `Standard persistent disk`
     - **Size**: `50 GB`
   - **Firewall**: 
     - ✅ **Allow HTTP traffic**
     - ✅ **Allow HTTPS traffic**
4. **Click "Create"** (takes 2-3 minutes)

---

## 🎮 Step 8: Create GPU Worker VM

### Create GPU VM Instance:

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
     - **GPU type**: `NVIDIA Tesla T4` (recommended for cost/performance)
     - **Number of GPUs**: `1`
     - **GPU platform**: Leave as default
   - **Boot disk**:
     - **Operating system**: `Ubuntu`
     - **Version**: `Ubuntu 20.04 LTS`
     - **Boot disk type**: `Standard persistent disk`
     - **Size**: `100 GB` (need space for models and temp files)
   - **Advanced options > Management**:
     - **Metadata**: Add key-value pair:
       - **Key**: `install-nvidia-driver`
       - **Value**: `True`

### GPU Recommendations by Budget:

| GPU Type | Monthly Cost | Performance | Recommended For |
|----------|--------------|-------------|-----------------|
| **Tesla T4** | ~$350 | Good | **Recommended** - Best cost/performance |
| **Tesla V100** | ~$800 | Excellent | High-volume processing |
| **Tesla A100** | ~$1200 | Outstanding | Maximum performance |

4. **Click "Create"** (takes 3-5 minutes)

---

## 🔥 Step 9: Configure Firewall Rules

### Create Firewall Rule:

1. **Go to VPC Network > Firewall**
2. **Click "Create Firewall Rule"**
3. **Configure rule**:
   - **Name**: `allow-video-enhancer-http`
   - **Direction**: `Ingress`
   - **Action**: `Allow`
   - **Targets**: `Specified target tags`
   - **Target tags**: `http-server,https-server`
   - **Source IP ranges**: `0.0.0.0/0`
   - **Protocols and ports**: 
     - ✅ **TCP** - Ports: `80,443`
4. **Click "Create"**

---

## 🚀 Step 10: Deploy Application

### Upload Service Account Key:

1. **SSH into API VM**:
   - Go to **Compute Engine > VM Instances**
   - Click **SSH** next to `video-enhancer-api`
2. **Upload service account key**:
   ```bash
   # Create directory
   mkdir -p /opt/video-enhancer
   
   # Upload your gcp-service-account.json file
   # (Use the upload button in SSH terminal or scp)
   ```

### Deploy API Server:

```bash
# Install Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh
sudo usermod -aG docker $USER

# Clone your repository
cd /opt/video-enhancer
git clone YOUR_REPOSITORY_URL .

# Set up environment
cd backend/apps/api
cp .env.example .env

# Edit environment file with your values
nano .env
```

**Environment Configuration** (`.env`):
```env
NODE_ENV=production
PORT=3000
DATABASE_URL=postgresql://postgres:YOUR_PASSWORD@/video_enhancer?host=/cloudsql/YOUR_PROJECT:us-central1:video-enhancer-db
JWT_SECRET=your-super-secure-jwt-secret-key
GCP_PROJECT_ID=your-project-id
GCS_BUCKET_VIDEOS_RAW=video-enhancer-raw-prod
GCS_BUCKET_VIDEOS_ENHANCED=video-enhancer-enhanced-prod
PUBSUB_TOPIC_VIDEO_JOBS=video-jobs
GOOGLE_APPLICATION_CREDENTIALS=/opt/video-enhancer/gcp-service-account.json
```

```bash
# Build and run
docker build -t video-enhancer-api .
docker run -d --name api --restart unless-stopped -p 80:3000 \
  -v /opt/video-enhancer/gcp-service-account.json:/app/gcp-service-account.json:ro \
  video-enhancer-api
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

## 🌐 Step 11: Get External IP Addresses

### Find Your API Server IP:

1. **Go to Compute Engine > VM Instances**
2. **Find `video-enhancer-api`**
3. **Note the "External IP"** (e.g., `34.123.45.67`)
4. **Test API**: Visit `http://YOUR_API_IP/health`

---

## 🔍 Step 12: Verify Deployment

### Check API Health:
```bash
curl http://YOUR_API_IP/health
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

### Check Worker Logs:
```bash
# SSH into worker VM
sudo docker logs worker -f
```

**Expected Output**:
```
Loading Stream-DiffVSR model from Jamichsu/Stream-DiffVSR...
Model loaded successfully on cuda
Worker listening for jobs on video-jobs-subscription...
```

---

## 💰 Cost Breakdown

| Service | Configuration | Monthly Cost |
|---------|---------------|--------------|
| **Cloud SQL** | db-g1-small | $25 |
| **Cloud Storage** | 500GB Standard | $10 |
| **Pub/Sub** | 1M messages | $1 |
| **API VM** | n1-standard-2 | $50 |
| **GPU VM** | n1-standard-4 + T4 | $350 |
| **Network** | Egress traffic | $10-20 |
| **Total** | | **~$446-456/month** |

### Cost Optimization Tips:
- Use **Preemptible GPU VMs** (70% cheaper, but can be terminated)
- Set up **auto-shutdown** during low usage hours
- Use **committed use discounts** for long-term deployments
- Monitor usage with **budget alerts**

---

## 🔧 Step 13: Set Up Monitoring (Optional)

### Enable Monitoring:

1. **Go to Monitoring > Overview**
2. **Create a workspace** (if prompted)
3. **Set up alerts for**:
   - CPU usage > 80%
   - Memory usage > 80%
   - Disk usage > 80%
   - API error rate > 5%

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

**🎉 Congratulations! Your AI Video Enhancer is now deployed on Google Cloud Platform!**

Your API is available at: `http://YOUR_API_IP`
API Documentation: `http://YOUR_API_IP/documentation`