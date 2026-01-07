import { Storage } from '@google-cloud/storage';
import { config } from '../config';

export class StorageService {
  private storage: Storage;

  constructor() {
    this.storage = new Storage({
      projectId: config.GCP_PROJECT_ID,
    });
  }

  /**
   * Upload a file to Google Cloud Storage
   */
  async uploadFile(
    bucketName: string,
    destination: string,
    fileBuffer: Buffer,
    contentType?: string
  ): Promise<string> {
    const bucket = this.storage.bucket(bucketName);
    const file = bucket.file(destination);

    await file.save(fileBuffer, {
      metadata: {
        contentType: contentType || 'application/octet-stream',
      },
    });

    return `gs://${bucketName}/${destination}`;
  }

  /**
   * Generate a signed URL for uploading
   */
  async generateUploadUrl(
    bucketName: string,
    destination: string,
    expiresInMinutes: number = 15
  ): Promise<string> {
    const bucket = this.storage.bucket(bucketName);
    const file = bucket.file(destination);

    const [url] = await file.getSignedUrl({
      version: 'v4',
      action: 'write',
      expires: Date.now() + expiresInMinutes * 60 * 1000,
      contentType: 'video/mp4',
    });

    return url;
  }

  /**
   * Generate a signed URL for downloading
   */
  async generateDownloadUrl(
    bucketName: string,
    filePath: string,
    expiresInMinutes: number = 60
  ): Promise<string> {
    const bucket = this.storage.bucket(bucketName);
    const file = bucket.file(filePath);

    const [url] = await file.getSignedUrl({
      version: 'v4',
      action: 'read',
      expires: Date.now() + expiresInMinutes * 60 * 1000,
    });

    return url;
  }

  /**
   * Delete a file from Google Cloud Storage
   */
  async deleteFile(bucketName: string, filePath: string): Promise<void> {
    const bucket = this.storage.bucket(bucketName);
    const file = bucket.file(filePath);
    await file.delete();
  }

  /**
   * Check if a file exists
   */
  async fileExists(bucketName: string, filePath: string): Promise<boolean> {
    const bucket = this.storage.bucket(bucketName);
    const file = bucket.file(filePath);
    const [exists] = await file.exists();
    return exists;
  }

  /**
   * Parse GCS path (gs://bucket/path) into bucket and file path
   */
  parseGcsPath(gcsPath: string): { bucket: string; path: string } {
    const match = gcsPath.match(/^gs:\/\/([^/]+)\/(.+)$/);
    if (!match) {
      throw new Error(`Invalid GCS path: ${gcsPath}`);
    }
    return { bucket: match[1], path: match[2] };
  }
}

export const storageService = new StorageService();
