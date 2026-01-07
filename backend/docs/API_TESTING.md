# API Testing Guide

Quick guide to test the API endpoints with cURL and sample data.

## Setup

1. Start the API server:
```bash
cd backend/apps/api
npm run dev
```

2. Ensure PostgreSQL is running and migrations are applied.

## Test Flow

### 1. Health Check

```bash
curl http://localhost:3000/health
```

Expected response:
```json
{
  "status": "ok",
  "timestamp": "2026-01-07T10:00:00.000Z"
}
```

### 2. Register User

```bash
curl -X POST http://localhost:3000/auth/register \
  -H "Content-Type: application/json" \
  -d '{
    "email": "test@example.com",
    "password": "SecurePassword123"
  }'
```

Expected response (201):
```json
{
  "token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
  "user": {
    "id": "550e8400-e29b-41d4-a716-446655440000",
    "email": "test@example.com",
    "plan": "FREE",
    "createdAt": "2026-01-07T10:00:00.000Z"
  }
}
```

Save the token for subsequent requests.

### 3. Login

```bash
curl -X POST http://localhost:3000/auth/login \
  -H "Content-Type: application/json" \
  -d '{
    "email": "test@example.com",
    "password": "SecurePassword123"
  }'
```

### 4. Get Current User

```bash
curl http://localhost:3000/me \
  -H "Authorization: Bearer YOUR_TOKEN_HERE"
```

### 5. Get Subscription Details

```bash
curl http://localhost:3000/me/subscription \
  -H "Authorization: Bearer YOUR_TOKEN_HERE"
```

Expected response:
```json
{
  "plan": "FREE",
  "limits": {
    "videosLimit": 1,
    "durationLimitSeconds": 30,
    "minutesLimit": null,
    "resolution": "720p",
    "priority": "low"
  },
  "usage": {
    "minutesUsed": 0,
    "minutesRemaining": null,
    "videosProcessed": 0
  },
  "billingCycle": {
    "start": "2026-01-07T10:00:00.000Z",
    "end": "2026-02-06T10:00:00.000Z"
  }
}
```

### 6. Upload Video

```bash
curl -X POST http://localhost:3000/videos/upload \
  -H "Authorization: Bearer YOUR_TOKEN_HERE" \
  -H "Content-Type: application/json" \
  -d '{
    "filename": "test_video.mp4",
    "durationSeconds": 25,
    "resolution": "720p"
  }'
```

Expected response (201):
```json
{
  "videoId": "660e8400-e29b-41d4-a716-446655440000",
  "jobId": "770e8400-e29b-41d4-a716-446655440000",
  "uploadUrl": "https://storage.googleapis.com/video-enhancer-raw/...",
  "message": "Upload URL generated. Upload your video to this URL, then processing will begin."
}
```

### 7. Upload Video File to GCS

Use the `uploadUrl` from previous response:

```bash
curl -X PUT "UPLOAD_URL_FROM_RESPONSE" \
  -H "Content-Type: video/mp4" \
  --upload-file /path/to/your/video.mp4
```

### 8. List Videos

```bash
curl http://localhost:3000/videos?limit=10&offset=0 \
  -H "Authorization: Bearer YOUR_TOKEN_HERE"
```

### 9. Get Video by ID

```bash
curl http://localhost:3000/videos/VIDEO_ID \
  -H "Authorization: Bearer YOUR_TOKEN_HERE"
```

### 10. List Jobs

```bash
curl http://localhost:3000/jobs?limit=10 \
  -H "Authorization: Bearer YOUR_TOKEN_HERE"
```

Filter by status:
```bash
curl "http://localhost:3000/jobs?status=PROCESSING" \
  -H "Authorization: Bearer YOUR_TOKEN_HERE"
```

### 11. Get Job by ID

```bash
curl http://localhost:3000/jobs/JOB_ID \
  -H "Authorization: Bearer YOUR_TOKEN_HERE"
```

## Error Responses

### 400 Bad Request
```json
{
  "error": {
    "message": "Validation error",
    "statusCode": 400,
    "details": [...]
  }
}
```

### 401 Unauthorized
```json
{
  "error": {
    "message": "Invalid or expired token",
    "statusCode": 401
  }
}
```

### 403 Forbidden (Quota Exceeded)
```json
{
  "error": {
    "message": "Free plan allows only 1 video. Please upgrade.",
    "statusCode": 403
  }
}
```

### 404 Not Found
```json
{
  "error": {
    "message": "Resource not found",
    "statusCode": 404
  }
}
```

## Using Postman

Import the Postman collection:
```
docs/postman_collection.json
```

Set environment variables:
- `base_url`: http://localhost:3000
- `auth_token`: (will be auto-set after login)

## Next Steps

1. Set up the Python worker to process jobs
2. Monitor job status until COMPLETED
3. Download enhanced video using the signed URL

---

**Last Updated:** 2026-01-07
