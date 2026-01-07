import { FastifyInstance } from 'fastify';
import { ZodTypeProvider } from 'fastify-type-provider-zod';
import { AuthService } from './auth.service';
import { registerSchema, loginSchema } from './auth.schema';
import logger from '../../utils/logger';

export async function authRoutes(fastify: FastifyInstance) {
  const server = fastify.withTypeProvider<ZodTypeProvider>();
  const authService = new AuthService(fastify.prisma);

  // POST /auth/register
  server.post(
    '/auth/register',
    {
      schema: {
        body: registerSchema,
        response: {
          201: {
            type: 'object',
            properties: {
              token: { type: 'string' },
              user: {
                type: 'object',
                properties: {
                  id: { type: 'string' },
                  email: { type: 'string' },
                  plan: { type: 'string' },
                  createdAt: { type: 'string' },
                },
              },
            },
          },
        },
        tags: ['Auth'],
        description: 'Register a new user',
      },
    },
    async (request, reply) => {
      const user = await authService.register(request.body);

      const token = fastify.jwt.sign({
        userId: user.id,
        email: user.email,
      });

      logger.info({ userId: user.id }, 'User registered successfully');

      return reply.code(201).send({
        token,
        user,
      });
    }
  );

  // POST /auth/login
  server.post(
    '/auth/login',
    {
      schema: {
        body: loginSchema,
        response: {
          200: {
            type: 'object',
            properties: {
              token: { type: 'string' },
              user: {
                type: 'object',
                properties: {
                  id: { type: 'string' },
                  email: { type: 'string' },
                  plan: { type: 'string' },
                  createdAt: { type: 'string' },
                },
              },
            },
          },
        },
        tags: ['Auth'],
        description: 'Login with email and password',
      },
    },
    async (request, reply) => {
      const user = await authService.login(request.body);

      const token = fastify.jwt.sign({
        userId: user.id,
        email: user.email,
      });

      logger.info({ userId: user.id }, 'User logged in successfully');

      return reply.send({
        token,
        user,
      });
    }
  );
}
