import Fastify, { FastifyError, FastifyReply, FastifyRequest } from 'fastify';
import cors from '@fastify/cors';
import jwt from '@fastify/jwt';
import multipart from '@fastify/multipart';
import swagger from '@fastify/swagger';
import swaggerUi from '@fastify/swagger-ui';
import { config } from './config';
import logger from './utils/logger';
import prismaPlugin from './plugins/prisma';
import { AppError } from './utils/errors';

// Routes
import { authRoutes } from './modules/auth/auth.routes';
import { userRoutes } from './modules/users/users.routes';
import { videoRoutes } from './modules/videos/videos.routes';
import { jobRoutes } from './modules/jobs/jobs.routes';

export async function buildApp() {
    const app = Fastify({
        logger: logger,
        requestIdLogLabel: 'requestId',
        disableRequestLogging: false,
        requestIdHeader: 'x-request-id',
    });

    // Register plugins
    await app.register(cors, {
        origin: true, // In production, specify exact origins
        credentials: true,
    });

    await app.register(jwt, {
        secret: config.JWT_SECRET,
        sign: {
            expiresIn: config.JWT_EXPIRES_IN,
        },
    });

    await app.register(multipart, {
        limits: {
            fileSize: config.MAX_FILE_SIZE_MB * 1024 * 1024,
        },
    });

    // Register Swagger
    await app.register(swagger, {
        swagger: {
            info: {
                title: 'AI Video Enhancer API',
                description: 'Production-ready backend for AI video enhancement SaaS',
                version: '1.0.0',
            },
            host: `localhost:${config.PORT}`,
            schemes: ['http', 'https'],
            consumes: ['application/json'],
            produces: ['application/json'],
            tags: [
                { name: 'Auth', description: 'Authentication endpoints' },
                { name: 'User', description: 'User management endpoints' },
                { name: 'Videos', description: 'Video upload and management' },
                { name: 'Jobs', description: 'Job tracking and status' },
            ],
            securityDefinitions: {
                bearerAuth: {
                    type: 'apiKey',
                    name: 'Authorization',
                    in: 'header',
                    description: 'Enter your JWT token in format: Bearer <token>',
                },
            },
        },
    });

    await app.register(swaggerUi, {
        routePrefix: '/documentation',
        uiConfig: {
            docExpansion: 'list',
            deepLinking: true,
        },
    });

    // Register database
    await app.register(prismaPlugin);

    // Health check
    app.get('/health', async () => {
        return { status: 'ok', timestamp: new Date().toISOString() };
    });

    // Register routes
    await app.register(authRoutes);
    await app.register(userRoutes);
    await app.register(videoRoutes);
    await app.register(jobRoutes);

    // Global error handler
    app.setErrorHandler((error: FastifyError | AppError, request: FastifyRequest, reply: FastifyReply) => {
        if (error instanceof AppError) {
            request.log.warn({ err: error, statusCode: error.statusCode }, error.message);
            return reply.status(error.statusCode).send({
                error: {
                    message: error.message,
                    statusCode: error.statusCode,
                },
            });
        }

        // Validation errors
        if (error.validation) {
            return reply.status(400).send({
                error: {
                    message: 'Validation error',
                    statusCode: 400,
                    details: error.validation,
                },
            });
        }

        // Unexpected errors
        request.log.error({ err: error }, 'Unexpected error');
        return reply.status(500).send({
            error: {
                message: 'Internal server error',
                statusCode: 500,
            },
        });
    });

    // 404 handler
    app.setNotFoundHandler((request, reply) => {
        reply.status(404).send({
            error: {
                message: 'Route not found',
                statusCode: 404,
                path: request.url,
            },
        });
    });

    return app;
}
