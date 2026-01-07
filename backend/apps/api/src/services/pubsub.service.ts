import { PubSub } from '@google-cloud/pubsub';
import { config } from '../config';
import logger from '../utils/logger';

interface VideoJobMessage {
    jobId: string;
    videoId: string;
    userId: string;
    inputPath: string;
    priority: 'low' | 'normal' | 'high';
}

export class PubSubService {
    private pubsub: PubSub;

    constructor() {
        this.pubsub = new PubSub({
            projectId: config.GCP_PROJECT_ID,
        });
    }

    /**
     * Publish a video processing job to Pub/Sub
     */
    async publishVideoJob(message: VideoJobMessage): Promise<string> {
        try {
            const topic = this.pubsub.topic(config.PUBSUB_TOPIC_VIDEO_JOBS);
            const messageBuffer = Buffer.from(JSON.stringify(message));

            const messageId = await topic.publishMessage({
                data: messageBuffer,
                attributes: {
                    priority: message.priority,
                    userId: message.userId,
                },
            });

            logger.info({ messageId, jobId: message.jobId }, 'Published video job to Pub/Sub');
            return messageId;
        } catch (error) {
            logger.error({ error, message }, 'Failed to publish video job');
            throw error;
        }
    }

    /**
     * Create topic if it doesn't exist (for setup)
     */
    async ensureTopicExists(topicName: string): Promise<void> {
        try {
            const topic = this.pubsub.topic(topicName);
            const [exists] = await topic.exists();

            if (!exists) {
                await topic.create();
                logger.info({ topicName }, 'Created Pub/Sub topic');
            }
        } catch (error) {
            logger.error({ error, topicName }, 'Failed to ensure topic exists');
            throw error;
        }
    }
}

export const pubsubService = new PubSubService();
