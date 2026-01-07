import { FastifyRequest } from 'fastify';
import { AppError } from '../utils/errors';

export interface AuthUser {
  userId: string;
  email: string;
}

declare module '@fastify/jwt' {
  interface FastifyJWT {
    payload: AuthUser;
    user: AuthUser;
  }
}

export const authenticate = async (request: FastifyRequest) => {
  try {
    await request.jwtVerify();
  } catch {
    throw new AppError('Invalid or expired token', 401);
  }
};
