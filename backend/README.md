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

### 1. Prerequisites (One-time Setup)
Ensure you have the Google Cloud SDK installed and authenticated.
```bash
gcloud auth login
gcloud config set project [YOUR_PROJECT_ID]
```

Enable necessary APIs:
```bash
gcloud services enable run.googleapis.com sqladmin.googleapis.com
```

### 2. Prepare Environment Variables
You will need the following values ready:
- `GCP_PROJECT_ID`: Your project ID.
- `GCS_BUCKET_RAW`: Bucket for uploads.
- `GCS_BUCKET_ENHANCED`: Bucket for results.
- `PUBSUB_TOPIC_NAME`: The Pub/Sub topic name (e.g., `video-enhancement-jobs`).
- `DB_INSTANCE_CONNECTION_NAME`: Find this in Cloud SQL > Overview (format: `project:region:instance`).
- `DB_USER` / `DB_PASSWORD`: Your Cloud SQL credentials.
- `DB_NAME`: Your database name.

### 3. Deploy Command
Run this command from the `backend/` directory.

> **Important**: Replace the bracketed values `[...]` with your actual configuration.

```bash
gcloud run deploy video-enhancer-backend \
  --source . \
  --region us-central1 \
  --allow-unauthenticated \
  --set-env-vars GCP_PROJECT_ID=[YOUR_PROJECT_ID] \
  --set-env-vars GCS_BUCKET_RAW=[GCS_BUCKET_RAW] \
  --set-env-vars GCS_BUCKET_ENHANCED=[GCS_BUCKET_ENHANCED] \
  --set-env-vars PUBSUB_TOPIC_NAME=[PUBSUB_TOPIC_NAME] \
  --set-env-vars DATABASE_URL="postgresql://[DB_USER]:[DB_PASSWORD]@/video_enhancer?host=/cloudsql/[DB_INSTANCE_CONNECTION_NAME]" \
  --add-cloudsql-instances [DB_INSTANCE_CONNECTION_NAME]
```

**breakdown of flags:**
- `--source .`: Uploads your code and builds it automatically (no Dockerfile manual build needed).
- `--allow-unauthenticated`: Makes the API public (remove this if you want to put it behind an API Gateway/Load Balancer later).
- `--add-cloudsql-instances`: Automatically mounts the Cloud SQL socket.
- `DATABASE_URL`: Note the format! It uses the socket path provided by Google.

### 4. Database Migration
Since Cloud Run is serverless, running migrations is best done separately. You can connect via Cloud Shell or from your local machine using the Auth Proxy (see Local Development section) to run the `init_db.sql` script.

```bash
# Example from local machine with Proxy running on port 5432
psql "postgresql://[DB_USER]:[DB_PASSWORD]@127.0.0.1:5432/[DB_NAME]" -f init_db.sql
```

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