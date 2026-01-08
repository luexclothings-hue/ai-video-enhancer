const util = require('util');
const { pipeline } = require('stream');
const pump = util.promisify(pipeline);
const db = require('../config/db');
const { rawBucket } = require('../config/storage');
const { publishMessage } = require('../config/pubsub');

class VideoService {
    async uploadVideo(fileStream, filename, mimeType) {
        // 1. Upload to GCS
        const gcsFileName = `${Date.now()}-${filename}`;
        const file = rawBucket.file(gcsFileName);

        // Use pump for efficient stream piping
        await pump(fileStream, file.createWriteStream({
            contentType: mimeType,
            resumable: false // For simple uploads
        }));

        const gcsPath = `gs://${rawBucket.name}/${gcsFileName}`;

        // 2. Save metadata to DB
        const insertQuery = `
      INSERT INTO videos (filename, gcs_raw_path, status)
      VALUES ($1, $2, 'PENDING', 0)
      RETURNING id, created_at, status, progress
    `;
        const result = await db.query(insertQuery, [filename, gcsPath]);
        const job = result.rows[0];

        // 3. Publish to Pub/Sub
        const messagePayload = {
            jobId: job.id,
            gcsRawPath: gcsPath,
            fileName: gcsFileName
        };
        await publishMessage(messagePayload);

        return job;
    }
}

module.exports = new VideoService();
