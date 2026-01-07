import { FastifyInstance, FastifyRequest } from 'fastify';
import { authenticate, AuthUser } from '../../middleware/auth.middleware';
import { AuthService } from '../auth/auth.service';
import { SUBSCRIPTION_LIMITS } from '../../config';
import { NotFoundError } from '../../utils/errors';

interface AuthenticatedRequest extends FastifyRequest {
  user: AuthUser;
}

type PlanType = 'FREE' | 'CREATOR' | 'PRO';

export async function userRoutes(fastify: FastifyInstance) {
  const authService = new AuthService(fastify.prisma);

  // GET /me
  fastify.get(
    '/me',
    {
      onRequest: [authenticate],
      schema: {
        response: {
          200: {
            type: 'object',
            properties: {
              id: { type: 'string' },
              email: { type: 'string' },
              plan: { type: 'string' },
              minutesUsedThisMonth: { type: 'number' },
              billingCycleStart: { type: 'string' },
              createdAt: { type: 'string' },
            },
          },
        },
        tags: ['User'],
        description: 'Get current user information',
        security: [{ bearerAuth: [] }],
      },
    },
    async (request: AuthenticatedRequest) => {
      const user = await authService.getUserById(request.user.userId);

      if (!user) {
        throw new NotFoundError('User not found');
      }

      return user;
    }
  );

  // GET /me/subscription
  fastify.get(
    '/me/subscription',
    {
      onRequest: [authenticate],
      schema: {
        response: {
          200: {
            type: 'object',
            properties: {
              plan: { type: 'string' },
              limits: {
                type: 'object',
                properties: {
                  videosLimit: { type: ['number', 'null'] },
                  durationLimitSeconds: { type: 'number' },
                  minutesLimit: { type: ['number', 'null'] },
                  resolution: { type: 'string' },
                  priority: { type: 'string' },
                },
              },
              usage: {
                type: 'object',
                properties: {
                  minutesUsed: { type: 'number' },
                  minutesRemaining: { type: ['number', 'null'] },
                  videosProcessed: { type: 'number' },
                },
              },
              billingCycle: {
                type: 'object',
                properties: {
                  start: { type: 'string' },
                  end: { type: 'string' },
                },
              },
            },
          },
        },
        tags: ['User'],
        description: 'Get subscription details and usage',
        security: [{ bearerAuth: [] }],
      },
    },
    async (request: AuthenticatedRequest) => {
      const user = await authService.getUserById(request.user.userId);

      if (!user) {
        throw new NotFoundError('User not found');
      }

      const limits = SUBSCRIPTION_LIMITS[user.plan as PlanType];

      // Calculate billing cycle end (30 days from start)
      const billingCycleEnd = new Date(user.billingCycleStart);
      billingCycleEnd.setDate(billingCycleEnd.getDate() + 30);

      // Count videos processed in current billing cycle
      const videosProcessed = await fastify.prisma.video.count({
        where: {
          userId: user.id,
          createdAt: {
            gte: user.billingCycleStart,
          },
        },
      });

      const minutesRemaining = limits.minutesLimit
        ? limits.minutesLimit - user.minutesUsedThisMonth
        : null;

      return {
        plan: user.plan,
        limits,
        usage: {
          minutesUsed: user.minutesUsedThisMonth,
          minutesRemaining,
          videosProcessed,
        },
        billingCycle: {
          start: user.billingCycleStart.toISOString(),
          end: billingCycleEnd.toISOString(),
        },
      };
    }
  );
}
