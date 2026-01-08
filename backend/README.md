# AI Video Enhancer - Backend

A Node.js (Fastify) backend for the AI Video Enhancer application. Handles video uploads, queues jobs for processing, and manages status via PostgreSQL.

## Features
- **Fast & Low Overhead**: Built with [Fastify](https://www.fastify.io/).
- **Cloud Native**: Integrated with Google Cloud Storage (GCS) and Cloud Pub/Sub.
- **Scalable**: Stateless design ready for Cloud Run.

## Local Development

### Prerequisites
- Node.js (v18+)
- PostgreSQL (Local or Cloud Proxy)
- Google Cloud Service Account with Storage Admin & Pub/Sub Editor roles.

### Setup
1. **Install Dependencies**:
   ```bash
   pnpm install
   ```

2. **Environment Variables**:
   Copy `.env.example` to `.env` and fill in your details.
   ```bash
   cp .env.example .env
   ```
   *Note: For local dev, `GCP_PROJECT_ID` is required. `keyFilename` in `storage.js` is optional if you have `gcloud auth application-default login` set up.*

3. **Database**:
   Run the init script to create the table:
   ```bash
   psql "YOUR_CONNECTION_STRING" -f init_db.sql
   ```

4. **Run Server**:
   ```bash
   pnpm dev
   ```

## API Endpoints

### `POST /api/v1/videos/upload`
Uploads a video file (`multipart/form-data`).
- **Body**: `file` (Binary video file)
- **Response**:
  ```json
  {
    "success": true,
    "job": {
      "id": 1,
      "status": "PENDING",
      "created_at": "..."
    }
  }
  ```

---

## Deployment to Google Cloud Run

We recommend deploying to Cloud Run for autoscaling and zero-maintenance.

### 1. Build and Deploy
You can deploy directly from source using the gcloud CLI.

```bash
gcloud run deploy video-enhancer-backend \
  --source . \
  --region us-central1 \
  --allow-unauthenticated \
  --set-env-vars GCP_PROJECT_ID=your-project-id \
  --set-env-vars GCS_BUCKET_RAW=your-raw-bucket \
  --set-env-vars GCS_BUCKET_ENHANCED=your-enhanced-bucket \
  --set-env-vars PUBSUB_TOPIC_NAME=your-topic-name
```

### 2. Connect to Cloud SQL (Option A: Socket)
For production, use the Unix socket connection.
1. Go to Cloud Run console > Edit Service.
2. **Connections** tab > **Cloud SQL connections**: Select your instance.
3. **Variables** tab: Set `DATABASE_URL` to:
   ```
   postgresql://USER:PASSWORD@/video_enhancer?host=/cloudsql/INSTANCE_CONNECTION_NAME
   ```
   *(Replace `INSTANCE_CONNECTION_NAME` with the string like `project:region:instance`)*

---

## Google Cloud Setup

### 1. Pub/Sub Setup
Create a topic for the worker queue:
```bash
gcloud pubsub topics create video-enhancement-jobs
```

### 2. Storage Buckets
Ensure you have two buckets created:
- `raw-videos-[suffix]` (Set lifecycle rule: Delete after 30 days)
- `enhanced-videos-[suffix]`