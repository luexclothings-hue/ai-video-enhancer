import { FastifyInstance, FastifyRequest } from 'fastify';
import { authenticate, AuthUser } from '../../middleware/auth.middleware';
import { JobService } from './jobs.service';
import { z } from 'zod';

interface AuthenticatedRequest extends FastifyRequest {
  user: AuthUser;
}

export async function jobRoutes(fastify: FastifyInstance) {
  const jobService = new JobService(fastify.prisma);

  // GET /jobs
  fastify.get(
    '/jobs',
    {
      onRequest: [authenticate],
      schema: {
        querystring: z.object({
          status: z.enum(['UPLOADED', 'QUEUED', 'PROCESSING', 'COMPLETED', 'FAILED']).optional(),
          limit: z.string().transform(Number).optional(),
          offset: z.string().transform(Number).optional(),
        }),
        response: {
          200: {
            type: 'object',
            properties: {
              jobs: {
                type: 'array',
                items: {
                  type: 'object',
                  properties: {
                    id: { type: 'string' },
                    status: { type: 'string' },
                    progressPercent: { type: 'number' },
                    errorMessage: { type: ['string', 'null'] },
                    minutesCharged: { type: ['number', 'null'] },
                    createdAt: { type: 'string' },
                    startedAt: { type: ['string', 'null'] },
                    completedAt: { type: ['string', 'null'] },
                    video: { type: 'object' },
                  },
                },
              },
              total: { type: 'number' },
              limit: { type: 'number' },
              offset: { type: 'number' },
            },
          },
        },
        tags: ['Jobs'],
        description: 'Get list of user jobs',
        security: [{ bearerAuth: [] }],
      },
    },
    async (request: AuthenticatedRequest) => {
      const userId = request.user.userId;
      const query = request.query as { status?: string; limit?: number; offset?: number };
      const status = query.status as
        | 'UPLOADED'
        | 'QUEUED'
        | 'PROCESSING'
        | 'COMPLETED'
        | 'FAILED'
        | undefined;
      const limit = query.limit || 50;
      const offset = query.offset || 0;

      const { jobs, total } = await jobService.getUserJobs(userId, status, limit, offset);

      return {
        jobs,
        total,
        limit,
        offset,
      };
    }
  );

  // GET /jobs/:id
  fastify.get(
    '/jobs/:id',
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
              status: { type: 'string' },
              progressPercent: { type: 'number' },
              errorMessage: { type: ['string', 'null'] },
              minutesCharged: { type: ['number', 'null'] },
              createdAt: { type: 'string' },
              startedAt: { type: ['string', 'null'] },
              completedAt: { type: ['string', 'null'] },
              video: {
                type: 'object',
                properties: {
                  id: { type: 'string' },
                  originalFilename: { type: 'string' },
                  durationSeconds: { type: 'number' },
                  resolution: { type: ['string', 'null'] },
                  storagePath: { type: 'string' },
                  outputStoragePath: { type: ['string', 'null'] },
                  createdAt: { type: 'string' },
                },
              },
            },
          },
        },
        tags: ['Jobs'],
        description: 'Get job by ID',
        security: [{ bearerAuth: [] }],
      },
    },
    async (request: AuthenticatedRequest) => {
      const userId = request.user.userId;
      const params = request.params as { id: string };

      const job = await jobService.getJobById(params.id, userId);

      return job;
    }
  );
}
