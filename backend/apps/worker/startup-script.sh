#!/bin/bash

# GPU Worker Startup Script for Cost Optimization
# This script sets up auto-shutdown functionality to minimize costs

set -e

echo "Starting GPU Worker setup..."

# Install Docker if not present
if ! command -v docker &> /dev/null; then
    echo "Installing Docker..."
    curl -fsSL https://get.docker.com -o get-docker.sh
    sudo sh get-docker.sh
    sudo usermod -aG docker $USER
fi

# Install NVIDIA Container Toolkit
if ! command -v nvidia-docker &> /dev/null; then
    echo "Installing NVIDIA Container Toolkit..."
    distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
    curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
    curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list
    
    sudo apt-get update && sudo apt-get install -y nvidia-docker2
    sudo systemctl restart docker
fi

# Test GPU access
echo "Testing GPU access..."
sudo docker run --rm --gpus all nvidia/cuda:11.8-base-ubuntu22.04 nvidia-smi

# Clone repository and set up worker
echo "Setting up worker application..."
cd /opt
sudo rm -rf video-enhancer
sudo git clone https://github.com/luexclothings-hue/ai-video-enhancer.git video-enhancer
cd video-enhancer/backend/apps/worker

# Set up environment (you'll need to configure this)
sudo cp .env.example .env
echo "Please configure /opt/video-enhancer/backend/apps/worker/.env with your settings"

# Build worker container
echo "Building worker container..."
sudo docker build -t video-enhancer-worker .

# Create activity monitoring script
sudo tee /opt/check-activity.sh > /dev/null << 'EOF'
#!/bin/bash

# Check if worker container is running
if ! sudo docker ps | grep -q video-enhancer-worker; then
    echo "Worker container not running, exiting check"
    exit 0
fi

# Check for recent activity (last 30 minutes)
RECENT_ACTIVITY=$(sudo docker logs video-enhancer-worker --since=30m 2>/dev/null | grep -c "Processing video\|Job completed\|Model loaded" || echo "0")

if [ "$RECENT_ACTIVITY" -eq 0 ]; then
    echo "No recent activity detected"
    
    # Check if there are pending jobs in Pub/Sub
    # Note: This requires gcloud CLI to be configured
    PENDING_JOBS=$(gcloud pubsub subscriptions pull video-jobs-subscription --limit=1 --format="value(message.data)" 2>/dev/null | wc -l || echo "0")
    
    if [ "$PENDING_JOBS" -eq 0 ]; then
        echo "No pending jobs, shutting down to save costs..."
        
        # Stop the worker container gracefully
        sudo docker stop video-enhancer-worker 2>/dev/null || true
        
        # Shutdown the VM
        sudo shutdown -h +2 "Auto-shutdown: No video processing activity detected"
    else
        echo "Pending jobs found, keeping VM running"
    fi
else
    echo "Recent activity detected ($RECENT_ACTIVITY events), keeping VM running"
fi
EOF

sudo chmod +x /opt/check-activity.sh

# Set up cron job for activity monitoring (check every 30 minutes)
echo "*/30 * * * * /opt/check-activity.sh >> /var/log/activity-check.log 2>&1" | sudo crontab -

# Start the worker container
echo "Starting worker container..."
sudo docker run -d \
    --name video-enhancer-worker \
    --restart unless-stopped \
    --gpus all \
    -v /opt/video-enhancer/gcp-service-account.json:/app/gcp-service-account.json:ro \
    video-enhancer-worker

echo "GPU Worker setup complete!"
echo "Worker logs: sudo docker logs video-enhancer-worker -f"
echo "Activity check logs: sudo tail -f /var/log/activity-check.log"
echo "Auto-shutdown will trigger after 30 minutes of inactivity"