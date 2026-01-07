from google.cloud import storage
from typing import Optional
import os
from logger import logger
import config

class CloudStorageClient:
    """Google Cloud Storage client wrapper"""
    
    def __init__(self):
        self.client = storage.Client(project=config.GCP_PROJECT_ID)
    
    def download_file(self, gcs_path: str, local_path: str) -> str:
        """
        Download file from GCS to local path
        
        Args:
            gcs_path: GCS path (gs://bucket/path or just bucket/path)
            local_path: Local file path
            
        Returns:
            Local file path
        """
        try:
            bucket_name, blob_path = self.parse_gcs_path(gcs_path)
            
            bucket = self.client.bucket(bucket_name)
            blob = bucket.blob(blob_path)
            
            # Ensure directory exists
            os.makedirs(os.path.dirname(local_path), exist_ok=True)
            
            logger.info(f'Downloading {gcs_path} to {local_path}')
            blob.download_to_filename(local_path)
            logger.info(f'Download completed: {local_path}')
            
            return local_path
            
        except Exception as e:
            logger.error(f'Failed to download file from GCS: {e}')
            raise
    
    def upload_file(self, local_path: str, gcs_path: str, content_type: Optional[str] = None) -> str:
        """
        Upload file to GCS
        
        Args:
            local_path: Local file path
            gcs_path: GCS destination path (gs://bucket/path or bucket/path)
            content_type: MIME type
            
        Returns:
            Full GCS path (gs://bucket/path)
        """
        try:
            bucket_name, blob_path = self.parse_gcs_path(gcs_path)
            
            bucket = self.client.bucket(bucket_name)
            blob = bucket.blob(blob_path)
            
            logger.info(f'Uploading {local_path} to {gcs_path}')
            
            if content_type:
                blob.upload_from_filename(local_path, content_type=content_type)
            else:
                blob.upload_from_filename(local_path)
            
            full_path = f'gs://{bucket_name}/{blob_path}'
            logger.info(f'Upload completed: {full_path}')
            
            return full_path
            
        except Exception as e:
            logger.error(f'Failed to upload file to GCS: {e}')
            raise
    
    @staticmethod
    def parse_gcs_path(gcs_path: str) -> tuple[str, str]:
        """
        Parse GCS path into bucket and blob path
        
        Args:
            gcs_path: gs://bucket/path or bucket/path
            
        Returns:
            (bucket_name, blob_path)
        """
        if gcs_path.startswith('gs://'):
            gcs_path = gcs_path[5:]
        
        parts = gcs_path.split('/', 1)
        if len(parts) != 2:
            raise ValueError(f'Invalid GCS path: {gcs_path}')
        
        return parts[0], parts[1]

storage_client = CloudStorageClient()
