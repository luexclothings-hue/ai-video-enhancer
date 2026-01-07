import bcrypt from 'bcrypt';
import { PrismaClient } from '@prisma/client';
import { ConflictError, UnauthorizedError } from '../../utils/errors';
import { RegisterInput, LoginInput } from './auth.schema';
import { config } from '../../config';

export class AuthService {
    constructor(private prisma: PrismaClient) { }

    async register(input: RegisterInput) {
        // Check if user already exists
        const existingUser = await this.prisma.user.findUnique({
            where: { email: input.email },
        });

        if (existingUser) {
            throw new ConflictError('User with this email already exists');
        }

        // Hash password
        const passwordHash = await bcrypt.hash(input.password, config.BCRYPT_ROUNDS);

        // Create user
        const user = await this.prisma.user.create({
            data: {
                email: input.email,
                passwordHash,
                plan: 'FREE',
                minutesUsedThisMonth: 0,
                billingCycleStart: new Date(),
            },
            select: {
                id: true,
                email: true,
                plan: true,
                createdAt: true,
            },
        });

        return user;
    }

    async login(input: LoginInput) {
        // Find user
        const user = await this.prisma.user.findUnique({
            where: { email: input.email },
        });

        if (!user) {
            throw new UnauthorizedError('Invalid email or password');
        }

        // Verify password
        const isPasswordValid = await bcrypt.compare(input.password, user.passwordHash);

        if (!isPasswordValid) {
            throw new UnauthorizedError('Invalid email or password');
        }

        return {
            id: user.id,
            email: user.email,
            plan: user.plan,
            createdAt: user.createdAt,
        };
    }

    async getUserById(userId: string) {
        const user = await this.prisma.user.findUnique({
            where: { id: userId },
            select: {
                id: true,
                email: true,
                plan: true,
                minutesUsedThisMonth: true,
                billingCycleStart: true,
                createdAt: true,
            },
        });

        return user;
    }
}
