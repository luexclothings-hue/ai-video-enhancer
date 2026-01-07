import os
import shutil
from typing import Dict
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm

from logger import logger
from ffmpeg_pipeline import FFmpegPipeline
from storage_client import storage_client
from database_client import db_client
from stream_diffvsr_integration import stream_diffvsr_processor
import config

class VideoProcessor:
    """Main video processing pipeline using Stream-DiffVSR"""
    
    def __init__(self):
        self.ffmpeg = FFmpegPipeline()
        self.device = config.DEVICE
        
    def load_model(self):
        """Load Stream-DiffVSR model"""
        try:
            logger.info('Loading Stream-DiffVSR model...')
            stream_diffvsr_processor.load_model()
            logger.info(f'Model loaded successfully on {self.device}')
        except Exception as e:
            logger.error(f'Failed to load model: {e}')
            raise
    
    def enhance_frames(self, input_dir: str, output_dir: str, total_frames: int) -> None:
        """
        Enhance video frames using Stream-DiffVSR
        
        Args:
            input_dir: Directory containing input frames
            output_dir: Directory for enhanced frames
            total_frames: Total number of frames
        """
        try:
            os.makedirs(output_dir, exist_ok=True)
            
            logger.info(f'Enhancing {total_frames} frames...')
            
            # Use the real Stream-DiffVSR batch processing
            stream_diffvsr_processor.enhance_frames_batch(input_dir, output_dir)
            
            logger.info('Frame enhancement completed')
            
        except Exception as e:
            logger.error(f'Frame enhancement failed: {e}')
            raise
    
    def process_video(self, job_data: Dict) -> None:
        """
        Main video processing pipeline
        
        Args:
            job_data: Job information from Pub/Sub message
        """
        job_id = job_data['jobId']
        video_id = job_data['videoId']
        user_id = job_data['userId']
        input_gcs_path = job_data['inputPath']
        
        temp_input = None
        temp_frames_input = None
        temp_frames_output = None
        temp_output_no_audio = None
        temp_output_final = None
        
        try:
            logger.info(f'Starting processing for job {job_id}')
            
            # Update status to PROCESSING
            db_client.update_job_status(job_id, 'PROCESSING', 0)
            
            # Create temp directories
            job_temp_dir = os.path.join(config.TEMP_DIR, job_id)
            os.makedirs(job_temp_dir, exist_ok=True)
            
            temp_input = os.path.join(job_temp_dir, 'input.mp4')
            temp_frames_input = os.path.join(job_temp_dir, 'frames_input')
            temp_frames_output = os.path.join(job_temp_dir, 'frames_output')
            temp_output_no_audio = os.path.join(job_temp_dir, 'output_no_audio.mp4')
            temp_output_final = os.path.join(job_temp_dir, 'output_final.mp4')
            
            # Step 1: Download video from GCS
            logger.info('Step 1: Downloading video from Cloud Storage')
            storage_client.download_file(input_gcs_path, temp_input)
            db_client.update_job_status(job_id, 'PROCESSING', 10)
            
            # Step 2: Get video info
            logger.info('Step 2: Extracting video metadata')
            video_info = self.ffmpeg.get_video_info(temp_input)
            duration_seconds = int(video_info['duration'])
            fps = int(video_info['fps'])
            logger.info(f'Video info: {video_info}')
            
            # Step 3: Extract frames
            logger.info('Step 3: Extracting frames')
            frame_count, _ = self.ffmpeg.extract_frames(
                temp_input,
                temp_frames_input,
                fps=config.FRAME_EXTRACTION_FPS
            )
            db_client.update_job_status(job_id, 'PROCESSING', 30)
            
            # Step 4: Load model and enhance frames
            logger.info('Step 4: Enhancing frames with Stream-DiffVSR')
            self.load_model()
            self.enhance_frames(temp_frames_input, temp_frames_output, frame_count)
            db_client.update_job_status(job_id, 'PROCESSING', 70)
            
            # Step 5: Encode enhanced frames to video
            logger.info('Step 5: Encoding enhanced frames to video')
            frames_pattern = os.path.join(temp_frames_output, 'frame_%06d.png')
            self.ffmpeg.encode_frames_to_video(
                frames_pattern,
                temp_output_no_audio,
                fps=fps,
                crf=config.OUTPUT_VIDEO_CRF,
                preset=config.OUTPUT_VIDEO_PRESET
            )
            db_client.update_job_status(job_id, 'PROCESSING', 85)
            
            # Step 6: Add audio from original video
            logger.info('Step 6: Adding audio to enhanced video')
            self.ffmpeg.add_audio_to_video(
                temp_output_no_audio,
                temp_input,
                temp_output_final
            )
            db_client.update_job_status(job_id, 'PROCESSING', 90)
            
            # Step 7: Upload enhanced video to GCS
            logger.info('Step 7: Uploading enhanced video to Cloud Storage')
            timestamp = os.path.basename(input_gcs_path).split('-')[0]
            output_gcs_path = f'gs://{config.GCS_BUCKET_VIDEOS_ENHANCED}/users/{user_id}/enhanced/{timestamp}-enhanced.mp4'
            
            storage_client.upload_file(
                temp_output_final,
                output_gcs_path,
                content_type='video/mp4'
            )
            db_client.update_job_status(job_id, 'PROCESSING', 95)
            
            # Step 8: Update video record with output path
            logger.info('Step 8: Updating video record')
            db_client.update_video_output_path(video_id, output_gcs_path)
            
            # Step 9: Charge minutes (idempotent)
            logger.info('Step 9: Charging minutes')
            minutes_charged = max(1, (duration_seconds + 59) // 60)  # Round up
            db_client.charge_minutes(job_id, minutes_charged)
            
            # Step 10: Mark job as completed
            logger.info('Step 10: Marking job as completed')
            db_client.update_job_status(job_id, 'COMPLETED', 100)
            
            logger.info(f'Job {job_id} completed successfully')
            
        except Exception as e:
            logger.error(f'Job {job_id} failed: {e}')
            db_client.update_job_status(job_id, 'FAILED', error_message=str(e))
            raise
            
        finally:
            # Cleanup temp files
            if job_temp_dir and os.path.exists(job_temp_dir):
                logger.info('Cleaning up temporary files')
                shutil.rmtree(job_temp_dir, ignore_errors=True)

processor = VideoProcessor()
