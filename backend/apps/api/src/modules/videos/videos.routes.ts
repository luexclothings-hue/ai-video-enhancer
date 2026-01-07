import { FastifyInstance } from 'fastify';
import { ZodTypeProvider } from 'fastify-type-provider-zod';
import { authenticate } from '../../middleware/auth.middleware';
import { VideoService } from './videos.service';
import { JobService } from '../jobs/jobs.service';
import { storageService } from '../../services/storage.service';
import { pubsubService } from '../../services/pubsub.service';
import { config } from '../../config';
import { uploadVideoSchema } from './videos.schema';
import logger from '../../utils/logger';
import { z } from 'zod';

export async function videoRoutes(fastify: FastifyInstance) {
  const server = fastify.withTypeProvider<ZodTypeProvider>();
  const videoService = new VideoService(fastify.prisma);
  const jobService = new JobService(fastify.prisma);

  // POST /videos/upload
  server.post(
    '/videos/upload',
    {
      onRequest: [authenticate],
      schema: {
        body: uploadVideoSchema,
        response: {
          201: {
            type: 'object',
            properties: {
              videoId: { type: 'string' },
              jobId: { type: 'string' },
              uploadUrl: { type: 'string' },
              message: { type: 'string' },
            },
          },
        },
        tags: ['Videos'],
        description: 'Upload a video for enhancement',
        security: [{ bearerAuth: [] }],
      },
    },
    async (request, reply) => {
      const { filename, durationSeconds, width, height } = request.body;
      const userId = request.user!.userId;

      // Validate subscription limits including resolution
      const { limits } = await videoService.validateVideoUpload(
        userId,
        durationSeconds,
        width,
        height
      );

      // Generate unique storage path
      const timestamp = Date.now();
      const storagePath = `users/${userId}/raw/${timestamp}-${filename}`;

      // Create video record
      const video = await videoService.createVideo(
        userId,
        filename,
        durationSeconds,
        `gs://${config.GCS_BUCKET_VIDEOS_RAW}/${storagePath}`,
        width,
        height
      );

      // Create job
      const job = await jobService.createJob(userId, video.id);

      // Generate signed upload URL (15 minutes validity)
      const uploadUrl = await storageService.generateUploadUrl(
        config.GCS_BUCKET_VIDEOS_RAW,
        storagePath,
        15
      );

      // Publish job to Pub/Sub
      await pubsubService.publishVideoJob({
        jobId: job.id,
        videoId: video.id,
        userId: userId,
        inputPath: video.storagePath,
        priority: limits.priority as 'low' | 'normal' | 'high',
      });

      // Update job status to QUEUED
      await jobService.updateJobStatus(job.id, 'QUEUED');

      logger.info(
        { videoId: video.id, jobId: job.id, resolution: `${width}x${height}` },
        'Video upload initiated'
      );

      return reply.code(201).send({
        videoId: video.id,
        jobId: job.id,
        uploadUrl,
        message: 'Upload URL generated. Upload your video to this URL, then processing will begin.',
      });
    }
  );

  // GET /videos
  server.get(
    '/videos',
    {
      onRequest: [authenticate],
      schema: {
        querystring: z.object({
          limit: z.string().transform(Number).optional(),
          offset: z.string().transform(Number).optional(),
        }),
        response: {
          200: {
            type: 'object',
            properties: {
              videos: {
                type: 'array',
                items: {
                  type: 'object',
                  properties: {
                    id: { type: 'string' },
                    originalFilename: { type: 'string' },
                    durationSeconds: { type: 'number' },
                    resolution: { type: ['string', 'null'] },
                    storagePath: { type: 'string' },
                    outputStoragePath: { type: ['string', 'null'] },
                    createdAt: { type: 'string' },
                    jobs: { type: 'array' },
                  },
                },
              },
              total: { type: 'number' },
              limit: { type: 'number' },
              offset: { type: 'number' },
            },
          },
        },
        tags: ['Videos'],
        description: 'Get list of user videos',
        security: [{ bearerAuth: [] }],
      },
    },
    async (request) => {
      const userId = request.user!.userId;
      const limit = request.query.limit || 50;
      const offset = request.query.offset || 0;

      const { videos, total } = await videoService.getUserVideos(userId, limit, offset);

      return {
        videos,
        total,
        limit,
        offset,
      };
    }
  );

  // GET /videos/:id
  server.get(
    '/videos/:id',
    {
      onRequest: [authenticate],
      schema: {
        params: z.object({
          id: z.string().uuid(),
        }),
        response: {
          200: {
            type: 'object',
            properties: {
              id: { type: 'string' },
              originalFilename: { type: 'string' },
              durationSeconds: { type: 'number' },
              resolution: { type: ['string', 'null'] },
              storagePath: { type: 'string' },
              outputStoragePath: { type: ['string', 'null'] },
              createdAt: { type: 'string' },
              jobs: { type: 'array' },
              downloadUrl: { type: ['string', 'null'] },
            },
          },
        },
        tags: ['Videos'],
        description: 'Get video by ID',
        security: [{ bearerAuth: [] }],
      },
    },
    async (request) => {
      const userId = request.user!.userId;
      const { id } = request.params;

      const video = await videoService.getVideoById(id, userId);

      // Generate download URL if enhanced video is available
      let downloadUrl = null;
      if (video.outputStoragePath) {
        const { bucket, path } = storageService.parseGcsPath(video.outputStoragePath);
        downloadUrl = await storageService.generateDownloadUrl(bucket, path, 60);
      }

      return {
        ...video,
        downloadUrl,
      };
    }
  );
}
