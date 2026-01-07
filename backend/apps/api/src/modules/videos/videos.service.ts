import { PrismaClient, UserPlan } from '@prisma/client';
import { SUBSCRIPTION_LIMITS } from '../../config';
import {
    SubscriptionLimitError,
    InsufficientMinutesError,
    NotFoundError,
} from '../../utils/errors';

export class VideoService {
    constructor(private prisma: PrismaClient) { }

    /**
     * Validate if user can upload a video based on subscription limits
     */
    async validateVideoUpload(userId: string, durationSeconds: number, width: number, height: number) {
        const user = await this.prisma.user.findUnique({
            where: { id: userId },
        });

        if (!user) {
            throw new NotFoundError('User not found');
        }

        const limits = SUBSCRIPTION_LIMITS[user.plan];
        const durationMinutes = Math.ceil(durationSeconds / 60);

        // Check video duration limit
        if (durationSeconds > limits.durationLimitSeconds) {
            throw new SubscriptionLimitError(
                `Video duration exceeds ${limits.durationLimitSeconds}s limit for ${user.plan} plan`
            );
        }

        // Check resolution limits
        if (width > limits.maxWidth || height > limits.maxHeight) {
            throw new SubscriptionLimitError(
                `Video resolution ${width}x${height} exceeds ${limits.maxResolution} limit for ${user.plan} plan`
            );
        }

        // Check if free plan has reached video limit
        if (user.plan === 'FREE' && limits.videosLimit !== null) {
            const videosCount = await this.prisma.video.count({
                where: { userId },
            });

            if (videosCount >= limits.videosLimit) {
                throw new SubscriptionLimitError(
                    `Free plan allows only ${limits.videosLimit} video. Please upgrade.`
                );
            }
        }

        // Check remaining minutes for paid plans
        if (limits.minutesLimit !== null) {
            const minutesRemaining = limits.minutesLimit - user.minutesUsedThisMonth;

            if (durationMinutes > minutesRemaining) {
                throw new InsufficientMinutesError(durationMinutes, minutesRemaining);
            }
        }

        return {
            user,
            limits,
            durationMinutes,
        };
    }

    /**
     * Create a video record
     */
    async createVideo(
        userId: string,
        filename: string,
        durationSeconds: number,
        storagePath: string,
        width: number,
        height: number
    ) {
        const video = await this.prisma.video.create({
            data: {
                userId,
                originalFilename: filename,
                durationSeconds,
                resolution: `${width}x${height}`,
                storagePath,
            },
        });

        return video;
    }

    /**
     * Get user's videos
     */
    async getUserVideos(userId: string, limit: number = 50, offset: number = 0) {
        const [videos, total] = await Promise.all([
            this.prisma.video.findMany({
                where: { userId },
                include: {
                    jobs: {
                        orderBy: { createdAt: 'desc' },
                        take: 1,
                    },
                },
                orderBy: { createdAt: 'desc' },
                take: limit,
                skip: offset,
            }),
            this.prisma.video.count({ where: { userId } }),
        ]);

        return { videos, total };
    }

    /**
     * Get single video by ID
     */
    async getVideoById(videoId: string, userId: string) {
        const video = await this.prisma.video.findFirst({
            where: {
                id: videoId,
                userId,
            },
            include: {
                jobs: {
                    orderBy: { createdAt: 'desc' },
                },
            },
        });

        if (!video) {
            throw new NotFoundError('Video not found');
        }

        return video;
    }
}
