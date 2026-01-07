import { FastifyRequest, FastifyReply } from 'fastify';
import { AppError } from '../utils/errors';
import logger from '../utils/logger';

export interface AuthUser {
    userId: string;
    email: string;
}

declare module 'fastify' {
    interface FastifyRequest {
        user?: AuthUser;
    }
}

export const authenticate = async (request: FastifyRequest, reply: FastifyReply) => {
    try {
        const decoded = await request.jwtVerify<AuthUser>();
        request.user = decoded;
    } catch (error) {
        logger.error({ error }, 'JWT verification failed');
        throw new AppError('Invalid or expired token', 401);
    }
};
