const { Storage } = require('@google-cloud/storage');
require('dotenv').config();

const storage = new Storage({
    projectId: process.env.GCP_PROJECT_ID,
});

const rawBucket = storage.bucket(process.env.GCS_BUCKET_RAW);
const enhancedBucket = storage.bucket(process.env.GCS_BUCKET_ENHANCED);

module.exports = {
    storage,
    rawBucket,
    enhancedBucket,
};
