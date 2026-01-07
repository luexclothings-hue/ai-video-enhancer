# AI Video Enhancer Backend

Professional AI video enhancement service using Stream-DiffVSR. Optimized for production deployment with 720p resolution support and subscription-based billing.

## 🏗️ Architecture

- **API Server**: Node.js/Fastify REST API with JWT authentication
- **GPU Worker**: Python-based Stream-DiffVSR processor
- **Database**: PostgreSQL for metadata and billing
- **Queue**: Google Cloud Pub/Sub for job distribution
- **Storage**: Google Cloud Storage for video files

## 🚀 Quick Start

### Development Setup

```bash
# Clone and setup
git clone <your-repo-url>
cd ai-video-enhancer/backend

# Windows
setup-dev.bat

# Linux/Mac
chmod +x setup-dev.sh && ./setup-dev.sh

# Configure environment
nano apps/api/.env
nano apps/worker/.env

# Start services
docker-compose up
```

### Production Deployment

Follow the comprehensive [GCP Deployment Guide](GCP_DEPLOYMENT_GUIDE.md) for step-by-step instructions with UI screenshots and specific GPU recommendations.

## 📊 API Endpoints

- `POST /auth/register` - User registration
- `POST /auth/login` - User authentication
- `POST /videos/upload` - Video upload (requires width/height)
- `GET /videos` - List user videos
- `GET /videos/:id` - Get video with download URL
- `GET /jobs/:id` - Check processing status
- `GET /health` - System health check

## 💳 Subscription Plans

| Plan    | Videos    | Duration | Resolution | Monthly Cost |
| ------- | --------- | -------- | ---------- | ------------ |
| FREE    | 1         | 30s      | 720p max   | Free         |
| CREATOR | Unlimited | 2min     | 720p max   | $9.99        |
| PRO     | Unlimited | 5min     | 720p max   | $29.99       |

**Resolution Limits**: All plans limited to 720p (1280x720) for optimal performance.

## 🔧 Configuration

### API Environment

```env
NODE_ENV=production
DATABASE_URL=postgresql://user:pass@host:5432/db
JWT_SECRET=your-secret-key
GCP_PROJECT_ID=your-project-id
GCS_BUCKET_VIDEOS_RAW=your-raw-bucket
GCS_BUCKET_VIDEOS_ENHANCED=your-enhanced-bucket
```

### Worker Environment

```env
DATABASE_URL=postgresql://user:pass@host:5432/db
GCP_PROJECT_ID=your-project-id
DEVICE=cuda
BATCH_SIZE=8
ENABLE_TENSORRT=false
TARGET_HEIGHT=720
TARGET_WIDTH=1280
```

## 🧪 Testing

### Postman Collection

1. Import `docs/postman_collection.json`
2. Import `docs/postman_environments.json`
3. Select appropriate environment
4. Test all endpoints

### API Testing

```bash
# Health check
curl http://localhost:3000/health

# Register user
curl -X POST http://localhost:3000/auth/register \
  -H "Content-Type: application/json" \
  -d '{"email":"test@example.com","password":"password123","plan":"FREE"}'
```

## 🎮 GPU Requirements

### Recommended GPUs:

- **Tesla T4**: $350/month - Best cost/performance ratio
- **Tesla V100**: $800/month - High performance
- **Tesla A100**: $1200/month - Maximum performance

### Performance Expectations:

- **720p video**: 1-2 seconds per frame
- **5-minute video**: 30-90 minutes processing
- **Batch size**: Auto-optimized based on GPU memory

## 📈 Monitoring

```bash
# API logs
docker-compose logs -f api

# Worker logs
docker-compose logs -f worker

# GPU usage
nvidia-smi
```

## 🔒 Security

- JWT authentication (1-hour expiration)
- bcrypt password hashing (12 rounds)
- Input validation with Zod schemas
- Signed URLs for secure file access
- Resolution limits enforced per subscription

## 💰 Cost Estimation

### Development: Free (local resources)

### Production (GCP):

- Cloud SQL: $25/month
- Cloud Storage: $10/month
- Pub/Sub: $1/month
- API VM: $50/month
- GPU VM: $350/month
- **Total: ~$436/month**

## 🛠️ Development

### Project Structure

```
backend/
├── apps/
│   ├── api/                 # Node.js API server
│   └── worker/              # Python GPU worker
├── docs/                    # API documentation
├── docker-compose.yml       # Development environment
└── GCP_DEPLOYMENT_GUIDE.md  # Production deployment
```

### Key Features

- **Stream-DiffVSR Integration**: Real model from HuggingFace Hub
- **Resolution Validation**: Enforced 720p limits
- **Dynamic Batch Sizing**: GPU memory optimization
- **Subscription Billing**: Usage tracking and limits
- **Professional Error Handling**: Comprehensive logging

## 📄 License

Proprietary - All rights reserved

---

**Built for production-scale AI video enhancement**
