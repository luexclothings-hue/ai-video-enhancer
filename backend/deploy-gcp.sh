#!/bin/bash

# AI Video Enhancer - GCP Production Deployment Script
# This script automates the deployment to Google Cloud Platform

set -e

# Configuration
PROJECT_ID=${PROJECT_ID:-"video-enhancer-prod"}
REGION=${REGION:-"us-central1"}
ZONE=${ZONE:-"us-central1-a"}
DB_INSTANCE_NAME="video-enhancer-db"
API_VM_NAME="video-enhancer-api"
WORKER_VM_NAME="video-enhancer-worker"

echo "🚀 Deploying AI Video Enhancer to Google Cloud Platform"
echo "======================================================"
echo "Project ID: $PROJECT_ID"
echo "Region: $REGION"
echo "Zone: $ZONE"
echo ""

# Check if gcloud is installed and authenticated
if ! command -v gcloud &> /dev/null; then
    echo "❌ gcloud CLI is not installed. Please install it first."
    exit 1
fi

# Check if user is authenticated
if ! gcloud auth list --filter=status:ACTIVE --format="value(account)" | grep -q .; then
    echo "❌ Not authenticated with gcloud. Please run: gcloud auth login"
    exit 1
fi

# Set project
echo "🔧 Setting up project..."
gcloud config set project $PROJECT_ID

# Enable required APIs
echo "🔌 Enabling required APIs..."
gcloud services enable \
    compute.googleapis.com \
    sqladmin.googleapis.com \
    storage.googleapis.com \
    pubsub.googleapis.com \
    cloudresourcemanager.googleapis.com \
    cloudbuild.googleapis.com

# Create Cloud SQL instance
echo "🗄️  Creating Cloud SQL instance..."
if ! gcloud sql instances describe $DB_INSTANCE_NAME &>/dev/null; then
    gcloud sql instances create $DB_INSTANCE_NAME \
        --database-version=POSTGRES_15 \
        --tier=db-g1-small \
        --region=$REGION \
        --root-password=$(openssl rand -base64 32) \
        --storage-size=20GB \
        --storage-auto-increase
    
    # Create database
    gcloud sql databases create video_enhancer \
        --instance=$DB_INSTANCE_NAME
    
    echo "✅ Cloud SQL instance created"
else
    echo "✅ Cloud SQL instance already exists"
fi

# Create Cloud Storage buckets
echo "🪣 Creating Cloud Storage buckets..."
gsutil mb -l $REGION gs://$PROJECT_ID-videos-raw || echo "Bucket already exists"
gsutil mb -l $REGION gs://$PROJECT_ID-videos-enhanced || echo "Bucket already exists"

# Set bucket lifecycle (delete after 30 days)
echo '{"lifecycle": {"rule": [{"action": {"type": "Delete"}, "condition": {"age": 30}}]}}' > /tmp/lifecycle.json
gsutil lifecycle set /tmp/lifecycle.json gs://$PROJECT_ID-videos-raw

# Create Pub/Sub topic and subscription
echo "📨 Creating Pub/Sub topic and subscription..."
gcloud pubsub topics create video-jobs || echo "Topic already exists"
gcloud pubsub subscriptions create video-jobs-subscription \
    --topic=video-jobs \
    --ack-deadline=600 \
    --message-retention-duration=7d || echo "Subscription already exists"

# Create service account for the application
echo "🔐 Creating service account..."
SERVICE_ACCOUNT_EMAIL="video-enhancer@$PROJECT_ID.iam.gserviceaccount.com"
gcloud iam service-accounts create video-enhancer \
    --display-name="Video Enhancer Service Account" || echo "Service account already exists"

# Grant necessary permissions
gcloud projects add-iam-policy-binding $PROJECT_ID \
    --member="serviceAccount:$SERVICE_ACCOUNT_EMAIL" \
    --role="roles/cloudsql.client"

gcloud projects add-iam-policy-binding $PROJECT_ID \
    --member="serviceAccount:$SERVICE_ACCOUNT_EMAIL" \
    --role="roles/storage.admin"

gcloud projects add-iam-policy-binding $PROJECT_ID \
    --member="serviceAccount:$SERVICE_ACCOUNT_EMAIL" \
    --role="roles/pubsub.editor"

# Create and download service account key
echo "🔑 Creating service account key..."
gcloud iam service-accounts keys create gcp-service-account.json \
    --iam-account=$SERVICE_ACCOUNT_EMAIL

# Build and push Docker images
echo "🐳 Building and pushing Docker images..."
cd apps/api
gcloud builds submit --tag gcr.io/$PROJECT_ID/video-enhancer-api .
cd ../worker
gcloud builds submit --tag gcr.io/$PROJECT_ID/video-enhancer-worker .
cd ../..

# Create API VM
echo "💻 Creating API VM..."
if ! gcloud compute instances describe $API_VM_NAME --zone=$ZONE &>/dev/null; then
    gcloud compute instances create $API_VM_NAME \
        --zone=$ZONE \
        --machine-type=n1-standard-2 \
        --image-family=cos-stable \
        --image-project=cos-cloud \
        --boot-disk-size=50GB \
        --tags=http-server,https-server \
        --service-account=$SERVICE_ACCOUNT_EMAIL \
        --scopes=cloud-platform \
        --metadata=startup-script='#!/bin/bash
            docker run -d \
                --name video-enhancer-api \
                --restart unless-stopped \
                -p 80:3000 \
                -e NODE_ENV=production \
                -e DATABASE_URL="postgresql://postgres:$(gcloud sql instances describe '$DB_INSTANCE_NAME' --format="value(rootPassword)")@/video_enhancer?host=/cloudsql/'$PROJECT_ID':'$REGION':'$DB_INSTANCE_NAME'" \
                -e JWT_SECRET="$(openssl rand -base64 32)" \
                -e GCP_PROJECT_ID='$PROJECT_ID' \
                -e GCS_BUCKET_VIDEOS_RAW='$PROJECT_ID'-videos-raw \
                -e GCS_BUCKET_VIDEOS_ENHANCED='$PROJECT_ID'-videos-enhanced \
                -e PUBSUB_TOPIC_VIDEO_JOBS=video-jobs \
                gcr.io/'$PROJECT_ID'/video-enhancer-api
        '
    echo "✅ API VM created"
else
    echo "✅ API VM already exists"
fi

# Create GPU Worker VM
echo "🎮 Creating GPU Worker VM..."
if ! gcloud compute instances describe $WORKER_VM_NAME --zone=$ZONE &>/dev/null; then
    gcloud compute instances create $WORKER_VM_NAME \
        --zone=$ZONE \
        --machine-type=n1-standard-4 \
        --accelerator=type=nvidia-tesla-t4,count=1 \
        --image-family=cos-stable \
        --image-project=cos-cloud \
        --boot-disk-size=100GB \
        --maintenance-policy=TERMINATE \
        --service-account=$SERVICE_ACCOUNT_EMAIL \
        --scopes=cloud-platform \
        --metadata=install-nvidia-driver=True,startup-script='#!/bin/bash
            # Install NVIDIA Docker runtime
            curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
            distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
            curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list
            sudo apt-get update && sudo apt-get install -y nvidia-docker2
            sudo systemctl restart docker
            
            # Run worker container
            docker run -d \
                --name video-enhancer-worker \
                --restart unless-stopped \
                --gpus all \
                -e DATABASE_URL="postgresql://postgres:$(gcloud sql instances describe '$DB_INSTANCE_NAME' --format="value(rootPassword)")@/video_enhancer?host=/cloudsql/'$PROJECT_ID':'$REGION':'$DB_INSTANCE_NAME'" \
                -e GCP_PROJECT_ID='$PROJECT_ID' \
                -e GCS_BUCKET_VIDEOS_RAW='$PROJECT_ID'-videos-raw \
                -e GCS_BUCKET_VIDEOS_ENHANCED='$PROJECT_ID'-videos-enhanced \
                -e PUBSUB_SUBSCRIPTION=video-jobs-subscription \
                -e DEVICE=cuda \
                -e BATCH_SIZE=8 \
                gcr.io/'$PROJECT_ID'/video-enhancer-worker
        '
    echo "✅ GPU Worker VM created"
else
    echo "✅ GPU Worker VM already exists"
fi

# Create firewall rules
echo "🔥 Creating firewall rules..."
gcloud compute firewall-rules create allow-video-enhancer-http \
    --allow tcp:80,tcp:443 \
    --target-tags http-server,https-server \
    --description "Allow HTTP and HTTPS traffic for Video Enhancer" || echo "Firewall rule already exists"

# Get external IP addresses
echo "🌐 Getting external IP addresses..."
API_IP=$(gcloud compute instances describe $API_VM_NAME --zone=$ZONE --format="value(networkInterfaces[0].accessConfigs[0].natIP)")
WORKER_IP=$(gcloud compute instances describe $WORKER_VM_NAME --zone=$ZONE --format="value(networkInterfaces[0].accessConfigs[0].natIP)")

echo ""
echo "✅ Deployment completed successfully!"
echo ""
echo "📋 Deployment Summary:"
echo "====================="
echo "Project ID: $PROJECT_ID"
echo "Region: $REGION"
echo "Zone: $ZONE"
echo ""
echo "🗄️  Database:"
echo "  Instance: $DB_INSTANCE_NAME"
echo "  Connection: $PROJECT_ID:$REGION:$DB_INSTANCE_NAME"
echo ""
echo "🪣 Storage Buckets:"
echo "  Raw Videos: gs://$PROJECT_ID-videos-raw"
echo "  Enhanced Videos: gs://$PROJECT_ID-videos-enhanced"
echo ""
echo "📨 Pub/Sub:"
echo "  Topic: video-jobs"
echo "  Subscription: video-jobs-subscription"
echo ""
echo "💻 Compute Instances:"
echo "  API Server: $API_VM_NAME ($API_IP)"
echo "  GPU Worker: $WORKER_VM_NAME ($WORKER_IP)"
echo ""
echo "🌐 API Endpoints:"
echo "  API: http://$API_IP"
echo "  Documentation: http://$API_IP/documentation"
echo ""
echo "🔧 Next Steps:"
echo "1. Wait 5-10 minutes for VMs to fully start"
echo "2. Test API health: curl http://$API_IP/health"
echo "3. Set up domain name and SSL certificate (optional)"
echo "4. Configure monitoring and alerting"
echo "5. Download Stream-DiffVSR model weights to worker VM"
echo ""
echo "💰 Estimated Monthly Cost: ~$400-500"
echo ""
echo "🔍 Monitoring Commands:"
echo "  gcloud compute ssh $API_VM_NAME --zone=$ZONE"
echo "  gcloud compute ssh $WORKER_VM_NAME --zone=$ZONE"
echo "  gcloud logging read 'resource.type=gce_instance' --limit=50"