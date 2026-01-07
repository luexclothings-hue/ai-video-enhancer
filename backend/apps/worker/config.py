import os
from dotenv import load_dotenv

load_dotenv()

# Google Cloud Platform
GCP_PROJECT_ID = os.getenv('GCP_PROJECT_ID')
PUBSUB_SUBSCRIPTION_ID = os.getenv('PUBSUB_SUBSCRIPTION_ID', 'video-jobs-subscription')
GCS_BUCKET_VIDEOS_RAW = os.getenv('GCS_BUCKET_VIDEOS_RAW')
GCS_BUCKET_VIDEOS_ENHANCED = os.getenv('GCS_BUCKET_VIDEOS_ENHANCED')

# Database
DATABASE_URL = os.getenv('DATABASE_URL')

# Model Configuration
DEVICE = os.getenv('DEVICE', 'cuda')
BATCH_SIZE = int(os.getenv('BATCH_SIZE', '4'))
MODEL_NUM_INFERENCE_STEPS = int(os.getenv('MODEL_NUM_INFERENCE_STEPS', '4'))

# Performance Optimizations
ENABLE_TENSORRT = os.getenv('ENABLE_TENSORRT', 'false').lower() == 'true'
TARGET_HEIGHT = int(os.getenv('TARGET_HEIGHT', '720'))
TARGET_WIDTH = int(os.getenv('TARGET_WIDTH', '1280'))

# Processing
TEMP_DIR = os.getenv('TEMP_DIR', '/tmp/video_processing')
FRAME_EXTRACTION_FPS = int(os.getenv('FRAME_EXTRACTION_FPS', '30'))
OUTPUT_VIDEO_CRF = int(os.getenv('OUTPUT_VIDEO_CRF', '18'))
OUTPUT_VIDEO_PRESET = os.getenv('OUTPUT_VIDEO_PRESET', 'slow')

# Ensure temp directory exists
os.makedirs(TEMP_DIR, exist_ok=True)
