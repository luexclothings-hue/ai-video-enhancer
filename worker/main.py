import os
import base64
import json
import logging
from flask import Flask, request
from dotenv import load_dotenv

# Load env
load_dotenv()

# Import core modules
from core.downloader import download_video

app = Flask(__name__)

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
        try:
            data = base64.b64decode(pubsub_message["data"]).decode("utf-8").strip()
            job_data = json.loads(data)
            
            job_id = job_data.get("jobId")
            gcs_raw_path = job_data.get("gcsRawPath")
            
            logging.info(f"Processing Job ID: {job_id}")
            
            # 1. Download Video
            try:
                local_path = download_video(gcs_raw_path)
                logging.info(f"Video downloaded to: {local_path}")
            except Exception as e:
                logging.error(f"Download failed: {e}")
                return (f"Download Error: {e}", 500)
            
            # TODO: Inference Logic will go here
            
            return ("Success", 200)
            
        except Exception as e:
            logging.error(f"Error processing message: {e}")
            return (f"Internal Server Error: {e}", 500)

    return ("No data found", 400)

@app.route("/", methods=["GET"])
def health():
    return "Worker is running", 200

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(debug=True, host="0.0.0.0", port=port)
