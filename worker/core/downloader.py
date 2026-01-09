import os
from google.cloud import storage

def download_video(gcs_path, local_dir="/tmp"):
    """
    Downloads a video from GCS to a local directory.
    Args:
        gcs_path (str): The full GCS path (e.g., gs://bucket/file.mp4)
        local_dir (str): Directory to save the file.
    Returns:
        str: Application-local path to the downloaded file.
    """
    # Parse bucket and filename
    # gcs_path format: gs://bucket-name/filename
    if not gcs_path.startswith("gs://"):
        raise ValueError("Invalid GCS path. Must start with gs://")
    
    parts = gcs_path[5:].split("/", 1)
    bucket_name = parts[0]
    blob_name = parts[1]
    
    # Initialize client
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    
    # Ensure local dir exists
    os.makedirs(local_dir, exist_ok=True)
    
    # Define local path
    local_filename = blob_name.split("/")[-1] # Handle nested paths if any
    local_path = os.path.join(local_dir, local_filename)
    
    # Download
    print(f"Downloading {gcs_path} to {local_path}...")
    blob.download_to_filename(local_path)
    print("Download complete.")
    
    return local_path

def upload_video(local_path, gcs_bucket_name, blob_name):
    """
    Uploads a local video to GCS.
    """
    storage_client = storage.Client()
    bucket = storage_client.bucket(gcs_bucket_name)
    blob = bucket.blob(blob_name)
    
    print(f"Uploading {local_path} to gs://{gcs_bucket_name}/{blob_name}...")
    blob.upload_from_filename(local_path)
    print("Upload complete.")
    
    return f"gs://{gcs_bucket_name}/{blob_name}"
