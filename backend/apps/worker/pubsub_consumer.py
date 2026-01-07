from google.cloud import pubsub_v1
import json
from typing import Callable
from logger import logger
import config

class PubSubConsumer:
    """Pub/Sub consumer for video processing jobs"""
    
    def __init__(self, callback: Callable):
        self.subscriber = pubsub_v1.SubscriberClient()
        self.subscription_path = self.subscriber.subscription_path(
            config.GCP_PROJECT_ID,
            config.PUBSUB_SUBSCRIPTION_ID
        )
        self.callback = callback
    
    def message_callback(self, message: pubsub_v1.subscriber.message.Message) -> None:
        """Handle incoming Pub/Sub message"""
        try:
            logger.info(f'Received message: {message.message_id}')
            
            # Parse message data
            data = json.loads(message.data.decode('utf-8'))
            
            logger.info(f'Processing job: {data.get("jobId")}')
            
            # Process the job
            self.callback(data)
            
            # Acknowledge the message
            message.ack()
            logger.info(f'Message {message.message_id} acknowledged')
            
        except Exception as e:
            logger.error(f'Error processing message {message.message_id}: {e}')
            # Nack the message to retry later
            message.nack()
    
    def start_consuming(self) -> None:
        """Start consuming messages from Pub/Sub"""
        logger.info(f'Starting to consume messages from {self.subscription_path}')
        
        streaming_pull_future = self.subscriber.subscribe(
            self.subscription_path,
            callback=self.message_callback
        )
        
        logger.info('Listening for messages...')
        
        try:
            # Block and wait for messages
            streaming_pull_future.result()
        except KeyboardInterrupt:
            logger.info('Received interrupt signal, stopping...')
            streaming_pull_future.cancel()
        except Exception as e:
            logger.error(f'Subscriber error: {e}')
            streaming_pull_future.cancel()
            raise
