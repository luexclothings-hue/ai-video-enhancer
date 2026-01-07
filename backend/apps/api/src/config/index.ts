import dotenv from 'dotenv';
import { z } from 'zod';

dotenv.config();

const configSchema = z.object({
    // Server
    NODE_ENV: z.enum(['development', 'production', 'test']).default('development'),
    PORT: z.string().transform(Number).default('3000'),
    HOST: z.string().default('0.0.0.0'),

    // Database
    DATABASE_URL: z.string(),

    // JWT
    JWT_SECRET: z.string(),
    JWT_EXPIRES_IN: z.string().default('1h'),

    // Google Cloud Platform
    GCP_PROJECT_ID: z.string(),
    GCS_BUCKET_VIDEOS_RAW: z.string(),
    GCS_BUCKET_VIDEOS_ENHANCED: z.string(),
    PUBSUB_TOPIC_VIDEO_JOBS: z.string(),

    // Application
    MAX_FILE_SIZE_MB: z.string().transform(Number).default('500'),
    BCRYPT_ROUNDS: z.string().transform(Number).default('12'),

    // Subscription Plans
    PLAN_FREE_VIDEOS_LIMIT: z.string().transform(Number).default('1'),
    PLAN_FREE_DURATION_LIMIT_SECONDS: z.string().transform(Number).default('30'),
    PLAN_CREATOR_MINUTES_LIMIT: z.string().transform(Number).default('30'),
    PLAN_CREATOR_VIDEO_DURATION_LIMIT_SECONDS: z.string().transform(Number).default('120'),
    PLAN_PRO_MINUTES_LIMIT: z.string().transform(Number).default('120'),
    PLAN_PRO_VIDEO_DURATION_LIMIT_SECONDS: z.string().transform(Number).default('300'),
});

const parseConfig = () => {
    try {
        return configSchema.parse(process.env);
    } catch (error) {
        if (error instanceof z.ZodError) {
            const missingVars = error.issues.map((issue) => issue.path.join('.')).join(', ');
            throw new Error(`Missing or invalid environment variables: ${missingVars}`);
        }
        throw error;
    }
};

export const config = parseConfig();

export const SUBSCRIPTION_LIMITS = {
    FREE: {
        videosLimit: config.PLAN_FREE_VIDEOS_LIMIT,
        durationLimitSeconds: config.PLAN_FREE_DURATION_LIMIT_SECONDS,
        minutesLimit: null,
        maxResolution: '720p',
        maxWidth: 1280,
        maxHeight: 720,
        priority: 'low',
    },
    CREATOR: {
        videosLimit: null,
        durationLimitSeconds: config.PLAN_CREATOR_VIDEO_DURATION_LIMIT_SECONDS,
        minutesLimit: config.PLAN_CREATOR_MINUTES_LIMIT,
        maxResolution: '720p',
        maxWidth: 1280,
        maxHeight: 720,
        priority: 'normal',
    },
    PRO: {
        videosLimit: null,
        durationLimitSeconds: config.PLAN_PRO_VIDEO_DURATION_LIMIT_SECONDS,
        minutesLimit: config.PLAN_PRO_MINUTES_LIMIT,
        maxResolution: '720p',
        maxWidth: 1280,
        maxHeight: 720,
        priority: 'high',
    },
} as const;
