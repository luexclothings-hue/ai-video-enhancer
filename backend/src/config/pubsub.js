const { PubSub } = require('@google-cloud/pubsub');

const pubSubClient = new PubSub({
    projectId: process.env.GCP_PROJECT_ID,
});

const topicName = process.env.PUBSUB_TOPIC_NAME;

async function publishMessage(data) {
    const dataBuffer = Buffer.from(JSON.stringify(data));

    try {
        const messageId = await pubSubClient.topic(topicName).publishMessage({ data: dataBuffer });
        console.log(`Message ${messageId} published.`);
        return messageId;
    } catch (error) {
        console.error(`Received error while publishing: ${error.message}`);
        throw error;
    }
}

module.exports = {
    pubSubClient,
    publishMessage,
};
