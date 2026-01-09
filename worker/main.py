import os
import base64
import json
import logging
from flask import Flask, request
from dotenv import load_dotenv

# Load env
load_dotenv()

# Import core modules
from core.downloader import download_video, upload_video
from core.inference import InferenceEngine
from core.db import update_job_status, update_job_progress

app = Flask(__name__)

# Global Inference Engine (Loads models once)
inference_engine = InferenceEngine()

# Configure logging
logging.basicConfig(level=logging.INFO)

@app.route("/process", methods=["POST"])
def process_job():
    """Receive and serve Pub/Sub messages."""
    envelope = request.get_json()
    if not envelope:
        msg = "no Pub/Sub message received"
        logging.error(msg)
        return f"Bad Request: {msg}", 400

    if not isinstance(envelope, dict) or "message" not in envelope:
        msg = "invalid Pub/Sub message format"
        logging.error(msg)
        return f"Bad Request: {msg}", 400

    pubsub_message = envelope["message"]

    if isinstance(pubsub_message, dict) and "data" in pubsub_message:
        job_id = None
        try:
            data = base64.b64decode(pubsub_message["data"]).decode("utf-8").strip()
            job_data = json.loads(data)
            
            job_id = job_data.get("jobId")
            gcs_raw_path = job_data.get("gcsRawPath")
            
            logging.info(f"Received Job ID: {job_id}")
            
            # 0. Update Status to PROCESSING
            update_job_status(job_id, 'PROCESSING')

            # 1. Download Video
            try:
                local_path = download_video(gcs_raw_path)
                logging.info(f"Video downloaded to: {local_path}")
            except Exception as e:
                logging.error(f"Download failed: {e}")
                update_job_status(job_id, 'FAILED')
                return (f"Download Error: {e}", 500)
            
            # 2. Run Inference
            try:
                enhanced_video_path = inference_engine.enhance_video(
                    local_path, 
                    job_id, 
                    update_callback=update_job_progress
                )
                logging.info(f"Inference complete. Output: {enhanced_video_path}")
            except Exception as e:
                logging.error(f"Inference failed: {e}")
                update_job_status(job_id, 'FAILED')
                return (f"Inference Error: {e}", 500)

            # 3. Upload Result
            try:
                # Construct output GCS path (e.g. enhanced/filename.mp4)
                filename = os.path.basename(enhanced_video_path)
                bucket_name = os.environ.get("GCS_BUCKET_ENHANCED", "video-enhancer-enhanced") # Default or Env
                
                gcs_enhanced_path = upload_video(enhanced_video_path, bucket_name, filename)
            except Exception as e:
                logging.error(f"Upload failed: {e}")
                update_job_status(job_id, 'FAILED')
                return (f"Upload Error: {e}", 500)

            # 4. Mark Completed
            update_job_status(job_id, 'COMPLETED', gcs_enhanced_path)
            logging.info(f"Job {job_id} Completed Successfully.")
            
            return ("Success", 200)
            
        except Exception as e:
            logging.error(f"Critical error processing message: {e}")
            if job_id:
                update_job_status(job_id, 'FAILED')
            return (f"Internal Server Error: {e}", 500)

    return ("No data found", 400)

@app.route("/", methods=["GET"])
def health():
    return "Worker is running", 200

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(debug=True, host="0.0.0.0", port=port)
