import functions_framework
import json
import base64
from google.cloud import compute_v1
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
PROJECT_ID = 'your-project-id'  # Replace with your project ID
ZONE = 'us-central1-a'
INSTANCE_NAME = 'video-enhancer-worker'

@functions_framework.cloud_event
def start_worker_vm(cloud_event):
    """
    Cloud Function triggered by Pub/Sub messages to start GPU worker VM
    when video processing jobs are queued.
    """
    try:
        # Decode the Pub/Sub message
        pubsub_message = base64.b64decode(cloud_event.data["message"]["data"]).decode()
        job_data = json.loads(pubsub_message)
        
        logger.info(f"Received job: {job_data}")
        
        # Initialize Compute Engine client
        compute_client = compute_v1.InstancesClient()
        
        # Check current instance status
        try:
            instance = compute_client.get(
                project=PROJECT_ID,
                zone=ZONE,
                instance=INSTANCE_NAME
            )
            
            current_status = instance.status
            logger.info(f"Worker VM current status: {current_status}")
            
            if current_status == 'TERMINATED':
                # Start the instance
                logger.info("Starting worker VM...")
                operation = compute_client.start(
                    project=PROJECT_ID,
                    zone=ZONE,
                    instance=INSTANCE_NAME
                )
                logger.info(f"Start operation initiated: {operation.name}")
                return f"Worker VM start initiated: {operation.name}"
                
            elif current_status == 'RUNNING':
                logger.info("Worker VM is already running")
                return "Worker VM is already running"
                
            else:
                logger.info(f"Worker VM is in {current_status} state, waiting...")
                return f"Worker VM is in {current_status} state"
                
        except Exception as e:
            logger.error(f"Error checking/starting worker VM: {str(e)}")
            return f"Error: {str(e)}"
            
    except Exception as e:
        logger.error(f"Error processing Pub/Sub message: {str(e)}")
        return f"Error processing message: {str(e)}"


@functions_framework.http
def start_worker_http(request):
    """
    HTTP endpoint to manually start the worker VM (for testing)
    """
    try:
        compute_client = compute_v1.InstancesClient()
        
        # Start the instance
        operation = compute_client.start(
            project=PROJECT_ID,
            zone=ZONE,
            instance=INSTANCE_NAME
        )
        
        return {
            'status': 'success',
            'message': f'Worker VM start initiated: {operation.name}'
        }
        
    except Exception as e:
        logger.error(f"Error starting worker VM: {str(e)}")
        return {
            'status': 'error',
            'message': str(e)
        }, 500