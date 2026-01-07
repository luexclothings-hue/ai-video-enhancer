#!/usr/bin/env python3
"""
AI Video Enhancer - GPU Worker
Main entry point for the video processing worker
"""

import signal
import sys
from logger import logger
from pubsub_consumer import PubSubConsumer
from video_processor import processor

# Global flag for graceful shutdown
shutdown_requested = False

def signal_handler(signum, frame):
    """Handle shutdown signals"""
    global shutdown_requested
    logger.info(f'Received signal {signum}, initiating graceful shutdown...')
    shutdown_requested = True
    sys.exit(0)

def process_job(job_data: dict):
    """Callback for processing video jobs"""
    try:
        processor.process_video(job_data)
    except Exception as e:
        logger.error(f'Failed to process job: {e}')
        # Error is already logged in processor, just propagate

def main():
    """Main entry point"""
    logger.info('=' * 60)
    logger.info('AI Video Enhancer - GPU Worker Starting')
    logger.info('=' * 60)
    
    # Register signal handlers for graceful shutdown
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    try:
        # Create Pub/Sub consumer
        consumer = PubSubConsumer(callback=process_job)
        
        # Start consuming messages
        consumer.start_consuming()
        
    except Exception as e:
        logger.error(f'Worker failed: {e}')
        sys.exit(1)

if __name__ == '__main__':
    main()
