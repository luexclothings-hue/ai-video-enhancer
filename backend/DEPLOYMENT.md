# Deployment Guide - Google Cloud Platform

Complete guide for deploying the AI Video Enhancement backend to GCP with GPU support.

## 📋 Prerequisites

- GCP Account with billing enabled
- `gcloud` CLI installed and configured
- Domain name (optional, for production)
- Estimated cost: ~$300-500/month (with GPU VM running 24/7)

## 🗺️ Deployment Checklist

- [ ] Create GCP project
- [ ] Enable required APIs
- [ ] Set up Cloud SQL (PostgreSQL)
- [ ] Create Cloud Storage buckets
- [ ] Set up Pub/Sub topic and subscription
- [ ] Create GPU VM for worker
- [ ] Install NVIDIA drivers and CUDA
- [ ] Deploy API server
- [ ] Deploy Python worker
- [ ] Configure domain and SSL
- [ ] Set up monitoring and logging

---

## 1️⃣ GCP Project Setup

### Create Project

```bash
# Set project ID
export PROJECT_ID="video-enhancer-prod"

# Create project
gcloud projects create $PROJECT_ID --name="AI Video Enhancer"

# Set as active project
gcloud config set project $PROJECT_ID

# Link billing account (replace with your billing account ID)
gcloud billing projects link $PROJECT_ID --billing-account=XXXXXX-XXXXXX-XXXXXX
```

### Enable Required APIs

```bash
gcloud services enable \
  compute.googleapis.com \
  sqladmin.googleapis.com \
  storage.googleapis.com \
  pubsub.googleapis.com \
  cloudresourcemanager.googleapis.com
```

---

## 2️⃣ Cloud SQL (PostgreSQL)

### Create Instance

```bash
gcloud sql instances create video-enhancer-db \
  --database-version=POSTGRES_15 \
  --tier=db-g1-small \
  --region=us-central1 \
  --root-password=YOUR_SECURE_PASSWORD \
  --storage-size=20GB \
  --storage-auto-increase
```

### Create Database

```bash
gcloud sql databases create video_enhancer \
  --instance=video-enhancer-db
```

### Get Connection String

```bash
# Get instance connection name
gcloud sql instances describe video-enhancer-db --format="value(connectionName)"

# Output: PROJECT_ID:REGION:INSTANCE_NAME
# Use this to build DATABASE_URL:
# postgresql://postgres:PASSWORD@/video_enhancer?host=/cloudsql/PROJECT_ID:REGION:INSTANCE_NAME
```

---

## 3️⃣ Cloud Storage Buckets

### Create Buckets

```bash
# Raw videos bucket
gsutil mb -l us-central1 gs://video-enhancer-raw-prod

# Enhanced videos bucket
gsutil mb -l us-central1 gs://video-enhancer-enhanced-prod

# Set lifecycle policy (optional: delete after 30 days)
echo '{"lifecycle": {"rule": [{"action": {"type": "Delete"}, "condition": {"age": 30}}]}}' > lifecycle.json
gsutil lifecycle set lifecycle.json gs://video-enhancer-raw-prod
```

### Set CORS (if serving videos directly to web)

```bash
echo '[{"origin": ["*"], "method": ["GET"], "maxAgeSeconds": 3600}]' > cors.json
gsutil cors set cors.json gs://video-enhancer-enhanced-prod
```

---

## 4️⃣ Cloud Pub/Sub

### Create Topic and Subscription

```bash
# Create topic
gcloud pubsub topics create video-jobs

# Create subscription for worker
gcloud pubsub subscriptions create video-jobs-subscription \
  --topic=video-jobs \
  --ack-deadline=600 \
  --message-retention-duration=7d
```

---

## 5️⃣ Deploy API Server

### Option A: Compute Engine VM

```bash
# Create VM for API
gcloud compute instances create video-enhancer-api \
  --machine-type=n1-standard-2 \
  --zone=us-central1-a \
  --image-family=ubuntu-2204-lts \
  --image-project=ubuntu-os-cloud \
  --boot-disk-size=50GB \
  --tags=http-server,https-server

# SSH into VM
gcloud compute ssh video-enhancer-api --zone=us-central1-a
```

Inside the VM:

```bash
# Install Node.js 20
curl -fsSL https://deb.nodesource.com/setup_20.x | sudo -E bash -
sudo apt-get install -y nodejs

# Install Git
sudo apt-get install -y git

# Clone your repo
git clone YOUR_REPO_URL /opt/video-enhancer
cd /opt/video-enhancer/backend/apps/api

# Install dependencies
npm install

# Set up environment
cp .env.example .env
nano .env  # Edit with production values

# Generate Prisma client
npm run generate

# Run migrations
npm run migrate

# Build
npm run build

# Install PM2 for process management
sudo npm install -g pm2

# Start API
pm2 start dist/server.js --name video-enhancer-api

# Save PM2 config
pm2 save
pm2 startup
```

### Option B: Cloud Run (for API only, not GPU tasks)

```bash
# Build container
cd apps/api
gcloud builds submit --tag gcr.io/$PROJECT_ID/video-enhancer-api

# Deploy to Cloud Run
gcloud run deploy video-enhancer-api \
  --image gcr.io/$PROJECT_ID/video-enhancer-api \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated \
  --set-env-vars="NODE_ENV=production" \
  --set-cloudsql-instances=$PROJECT_ID:us-central1:video-enhancer-db
```

---

## 6️⃣ GPU Worker Setup

### Create GPU VM

```bash
gcloud compute instances create video-enhancer-worker \
  --zone=us-central1-a \
  --machine-type=n1-standard-4 \
  --accelerator=type=nvidia-tesla-t4,count=1 \
  --image-family=ubuntu-2204-lts \
  --image-project=ubuntu-os-cloud \
  --boot-disk-size=100GB \
  --maintenance-policy=TERMINATE \
  --metadata=install-nvidia-driver=True
```

### Install Dependencies

SSH into worker:

```bash
gcloud compute ssh video-enhancer-worker --zone=us-central1-a
```

Inside the worker VM:

```bash
# Install NVIDIA drivers (if not auto-installed)
sudo apt-get update
sudo apt-get install -y nvidia-driver-535

# Verify GPU
nvidia-smi

# Install CUDA 11.8
wget https://developer.download.nvidia.com/compute/cuda/11.8.0/local_installers/cuda_11.8.0_520.61.05_linux.run
sudo sh cuda_11.8.0_520.61.05_linux.run --silent --toolkit

# Add to PATH
echo 'export PATH=/usr/local/cuda-11.8/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda-11.8/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc

# Install Python 3.10
sudo apt-get install -y python3.10 python3.10-venv python3-pip

# Install FFmpeg
sudo apt-get install -y ffmpeg

# Clone repo
sudo mkdir -p /opt/video-enhancer
sudo chown $USER:$USER /opt/video-enhancer
git clone YOUR_REPO_URL /opt/video-enhancer
cd /opt/video-enhancer/backend/apps/worker

# Create virtual environment
python3.10 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Clone Stream-DiffVSR
git clone https://github.com/jamichss/Stream-DiffVSR.git stream_diffvsr
cd stream_diffvsr

# Download pretrained weights
# Follow instructions from: https://jamichss.github.io/stream-diffvsr-project-page/
# Place weights in checkpoints/

# Set up environment
cd ..
cp .env.example .env
nano .env  # Edit with production values

# Test worker
python main.py
```

### Set up as systemd service

```bash
sudo nano /etc/systemd/system/video-worker.service
```

Add:

```ini
[Unit]
Description=Video Enhancement Worker
After=network.target

[Service]
Type=simple
User=YOUR_USERNAME
WorkingDirectory=/opt/video-enhancer/backend/apps/worker
Environment="PATH=/opt/video-enhancer/backend/apps/worker/.venv/bin:/usr/local/cuda-11.8/bin"
Environment="LD_LIBRARY_PATH=/usr/local/cuda-11.8/lib64"
ExecStart=/opt/video-enhancer/backend/apps/worker/.venv/bin/python main.py
Restart=always

[Install]
WantedBy=multi-user.target
```

Enable and start:

```bash
sudo systemctl daemon-reload
sudo systemctl enable video-worker
sudo systemctl start video-worker
sudo systemctl status video-worker
```

---

## 7️⃣ Firewall Rules

```bash
# Allow HTTP/HTTPS for API
gcloud compute firewall-rules create allow-http \
  --allow tcp:80,tcp:443 \
  --target-tags http-server,https-server
```

---

## 8️⃣ Domain and SSL (Optional)

### Using Cloud Load Balancer

1. Reserve static IP
2. Configure backend service
3. Configure URL map
4. Set up SSL certificate
5. Update DNS

Refer to: https://cloud.google.com/load-balancing/docs/https

---

## 9️⃣ Monitoring & Logging

### Cloud Logging

```bash
# View API logs
gcloud logging read "resource.type=gce_instance AND resource.labels.instance_id=video-enhancer-api" --limit 50

# View worker logs
gcloud logging read "resource.type=gce_instance AND resource.labels.instance_id=video-enhancer-worker" --limit 50
```

### Cloud Monitoring

Set up alerts for:
- CPU usage > 80%
- Memory usage > 80%
- Disk usage > 80%
- API error rate
- Worker errors

---

## 🔒 Security Checklist

- [ ] Set strong passwords for all services
- [ ] Use service accounts with minimal permissions
- [ ] Enable VPC firewall rules
- [ ] Use private IPs where possible
- [ ] Enable Cloud SQL SSL
- [ ] Rotate JWT secrets regularly
- [ ] Set up Cloud Armor (DDoS protection)
- [ ] Enable audit logging

---

## 💰 Cost Estimation

| Service | Spec | Monthly Cost |
|---------|------|--------------|
| Cloud SQL | db-g1-small | $25 |
| Cloud Storage | 500GB | $10 |
| Pub/Sub | 1M messages | $1 |
| API VM | n1-standard-2 | $50 |
| GPU VM | n1-standard-4 + T4 | $350 |
| **Total** | | **~$436/month** |

💡 **Cost Optimization Tips:**
- Use preemptible GPU VMs (70% cheaper, but can be terminated)
- Scale down when not in use
- Use Cloud Functions for API (pay per request)
- Set up budget alerts

---

## 🔄 CI/CD (Optional)

Use Cloud Build for automated deployments:

```yaml
# cloudbuild.yaml
steps:
  - name: 'gcr.io/cloud-builders/npm'
    args: ['install']
    dir: 'backend/apps/api'
  
  - name: 'gcr.io/cloud-builders/npm'
    args: ['run', 'build']
    dir: 'backend/apps/api'
  
  - name: 'gcr.io/cloud-builders/docker'
    args: ['build', '-t', 'gcr.io/$PROJECT_ID/video-enhancer-api', '.']
    dir: 'backend/apps/api'
  
  - name: 'gcr.io/cloud-builders/docker'
    args: ['push', 'gcr.io/$PROJECT_ID/video-enhancer-api']
```

---

## 📞 Troubleshooting

### API won't start
- Check DATABASE_URL is correct
- Verify Cloud SQL instance is running
- Check firewall rules

### Worker not processing jobs
- Verify GPU with `nvidia-smi`
- Check Pub/Sub subscription exists
- Verify model weights are downloaded
- Check worker logs: `journalctl -u video-worker -f`

### High costs
- Check for stuck VMs
- Review Cloud Storage lifecycle policies
- Monitor Pub/Sub message retention

---

**Deployment complete! 🎉**

Your production backend is now running on GCP with GPU acceleration.
