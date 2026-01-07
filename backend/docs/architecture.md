# System Architecture

## Overview

The AI Video Enhancement system is a production-ready, scalable backend for processing video enhancement requests from a Flutter mobile app using **Stream-DiffVSR** deep learning model.

## Architecture Diagram

```
┌─────────────┐
│   Flutter   │
│  Mobile App │
└──────┬──────┘
       │ HTTPS/REST
       │
┌──────▼──────────────────────────────────────────┐
│         Fastify API (Node.js)                   │
│  ┌──────────────────────────────────────────┐   │
│  │  Auth   │  Users  │  Videos  │  Jobs    │   │
│  └──────────────────────────────────────────┘   │
│              │                                   │
│  ┌───────────▼────────┐   ┌──────────────────┐  │
│  │  PostgreSQL (Prisma)│   │  Cloud Storage  │  │
│  └────────────────────┘   └──────────────────┘  │
└───────────────┬──────────────────────────────────┘
                │
        ┌───────▼────────┐
        │  Cloud Pub/Sub │
        │  (Job Queue)   │
        └───────┬────────┘
                │
┌───────────────▼──────────────────────────────────┐
│       GPU Worker (Python)                        │
│  ┌──────────────────────────────────────────┐    │
│  │ Pub/Sub Consumer → Video Processor       │    │
│  │                                           │    │
│  │  1. Download video (Cloud Storage)       │    │
│  │  2. Extract frames (FFmpeg)              │    │
│  │  3. Enhance frames (Stream-DiffVSR)      │    │
│  │  4. Re-encode video (FFmpeg)             │    │
│  │  5. Upload result (Cloud Storage)        │    │
│  │  6. Update job status (PostgreSQL)       │    │
│  │  7. Charge minutes (idempotent)          │    │
│  └──────────────────────────────────────────┘    │
│                                                   │
│  Hardware: NVIDIA GPU (T4/V100/A100)              │
│  Software: CUDA 11.8+, PyTorch, FFmpeg           │
└───────────────────────────────────────────────────┘
```

## Data Flow

### 1. Video Upload Flow

```
User (Flutter) → API: POST /videos/upload
                 │
                 ├─ Validate subscription limits
                 ├─ Check remaining minutes
                 ├─ Generate signed upload URL
                 ├─ Create Video record (DB)
                 ├─ Create Job record (status: UPLOADED)
                 ├─ Publish message to Pub/Sub
                 └─ Return uploadUrl, videoId, jobId

User uploads file → Cloud Storage (signed URL)

Worker (Pub/Sub) ← Message received
                 │
                 └─ Process video (see Processing Flow)
```

### 2. Video Processing Flow

```
Worker receives Pub/Sub message
  │
  ├─ Update Job: status = PROCESSING
  │
  ├─ Download video from Cloud Storage
  ├─ Extract frames with FFmpeg
  │   └─ frame_000001.png, frame_000002.png, ...
  │
  ├─ Load Stream-DiffVSR model (GPU)
  │
  ├─ Enhance frames in batches
  │   └─ Input: LQ frames → Model → Output: HQ frames
  │
  ├─ Re-encode enhanced frames to video (FFmpeg)
  │   └─ CRF 18, slow preset for quality
  │
  ├─ Merge audio from original video
  │
  ├─ Upload enhanced video to Cloud Storage
  │
  ├─ Update Video: outputStoragePath
  │
  ├─ Charge minutes (idempotent transaction)
  │   └─ Update Job: minutesCharged
  │   └─ Update User: minutesUsedThisMonth += charged
  │
  └─ Update Job: status = COMPLETED
```

### 3. Video Retrieval Flow

```
User (Flutter) → API: GET /videos/:id
                 │
                 ├─ Fetch Video + Jobs from DB
                 ├─ If outputStoragePath exists:
                 │   └─ Generate signed download URL (60 min expiry)
                 └─ Return video details + downloadUrl
```

## Component Details

### Fastify API

**Responsibilities:**

- User authentication (JWT)
- Subscription management and validation
- Video metadata storage
- Job status tracking
- Cloud Storage signed URL generation
- Pub/Sub job publishing

**Tech Stack:**

- Node.js 20+ with TypeScript
- Fastify (HTTP server)
- Prisma ORM (PostgreSQL)
- @google-cloud/storage
- @google-cloud/pubsub
- bcrypt (password hashing)
- Zod (validation)

**Scaling:**

- Stateless design (horizontal scaling)
- Connection pooling (database)
- JWT for auth (no session storage)

### Python GPU Worker

**Responsibilities:**

- Consume Pub/Sub messages
- Download/upload videos
- Frame extraction and encoding (FFmpeg)
- AI enhancement (Stream-DiffVSR)
- Job status updates
- Minute charging

**Tech Stack:**

- Python 3.10+
- PyTorch 2.1+ with CUDA
- Stream-DiffVSR (pretrained)
- FFmpeg (via ffmpeg-python)
- Google Cloud SDKs
- psycopg2 (PostgreSQL)

**Scaling:**

- Multiple worker instances (horizontal)
- Pub/Sub distributes load
- Each worker processes one job at a time
- GPU batching for efficiency

### Database Schema

**Users Table:**

- `id` (UUID, primary key)
- `email` (unique)
- `passwordHash`
- `plan` (FREE, CREATOR, PRO)
- `minutesUsedThisMonth`
- `billingCycleStart`

**Videos Table:**

- `id` (UUID, primary key)
- `userId` (foreign key)
- `originalFilename`
- `durationSeconds`
- `resolution`
- `storagePath` (GCS URI)
- `outputStoragePath` (GCS URI, nullable)

**Jobs Table:**

- `id` (UUID, primary key)
- `userId` (foreign key)
- `videoId` (foreign key)
- `status` (UPLOADED, QUEUED, PROCESSING, COMPLETED, FAILED)
- `progressPercent`
- `errorMessage` (nullable)
- `minutesCharged` (nullable, for idempotency)
- `startedAt`, `completedAt`

### Cloud Infrastructure

**Cloud Storage:**

- Bucket: `video-enhancer-raw` (input videos)
- Bucket: `video-enhancer-enhanced` (output videos)
- Access: Signed URLs (time-limited)

**Cloud Pub/Sub:**

- Topic: `video-jobs`
- Subscription: `video-jobs-subscription`
- Message format: JSON with jobId, videoId, userId, inputPath, priority

**Compute:**

- API: Compute Engine VM (n1-standard-2) or Cloud Run
- Worker: Compute Engine VM with GPU (n1-standard-4 + T4)

## Subscription Plans

| Plan    | Videos/Lifetime | Max Duration | Minutes/Month | Resolution | Queue Priority |
| ------- | --------------- | ------------ | ------------- | ---------- | -------------- |
| FREE    | 1               | 30s          | -             | 720p       | Low            |
| CREATOR | Unlimited       | 2m           | 30            | 1080p      | Normal         |
| PRO     | Unlimited       | 5m           | 120           | 1080p      | High           |

**Minute Calculation:**

- Charged minutes = `ceil(video_duration_seconds / 60)`
- Deducted from `minutesUsedThisMonth` after processing
- Billing cycle: 30 days from `billingCycleStart`

## Security

**Authentication:**

- JWT tokens (1 hour expiration)
- Passwords hashed with bcrypt (12 rounds)
- Bearer token in Authorization header

**Authorization:**

- Users can only access their own resources
- Middleware checks `userId` from JWT

**Data Protection:**

- Signed URLs for Cloud Storage (time-limited)
- HTTPS only in production
- SQL injection prevention (Prisma ORM)
- Input validation (Zod schemas)

## Error Handling

**API Errors:**

- `400 Bad Request` - Invalid input
- `401 Unauthorized` - Missing/invalid token
- `403 Forbidden` - Insufficient quota
- `404 Not Found` - Resource not found
- `500 Internal Server Error` - Unexpected errors

**Worker Errors:**

- Job status set to `FAILED`
- Error message stored in database
- Pub/Sub message nacked (retry)
- Logs sent to Cloud Logging

## Monitoring & Observability

**Logging:**

- Structured JSON logs (Pino)
- Request IDs for tracing
- Worker logs to stdout (captured by Cloud Logging)

**Metrics:**

- Job completion rate
- Average processing time
- Error rates
- GPU utilization
- API latency

**Alerts:**

- High error rate
- Worker crashes
- Database connection failures
- Storage quota exceeded

## Scalability Considerations

**API Scaling:**

- Horizontal: Add more API instances
- Database: Read replicas for analytics
- Cache: Redis for frequently accessed data (future)

**Worker Scaling:**

- Add more GPU VMs for parallel processing
- Pub/Sub automatically distributes load
- Each instance can process ~10-20 videos/hour (depending on length)

**Cost vs Performance:**

- Free tier: Process sequentially
- Paid tiers: Higher priority in queue
- Auto-scaling based on queue depth (future)

## Future Enhancements

1. **Webhooks** - Notify app when job completes
2. **Real-time progress** - WebSocket updates
3. **Batch processing** - Process multiple videos together
4. **Model selection** - Different enhancement models
5. **Video preview** - Generate thumbnail/preview
6. **Analytics dashboard** - Usage stats for users
7. **CDN integration** - Faster video delivery
8. **Auto-scaling** - Dynamic worker scaling

---

**Architecture Version:** 1.0  
**Last Updated:** 2026-01-07
