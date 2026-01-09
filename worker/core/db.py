import os
import psycopg2
import logging
from urllib.parse import urlparse

# Global DB Connection (reused in Cloud Run)
_db_pool = None

def get_db_connection():
    """
    Establishes a connection to the PostgreSQL database.
    Supports both TCP (Local) and Unix Socket (Cloud Run) via DATABASE_URL.
    """
    db_url = os.environ.get("DATABASE_URL")
    if not db_url:
        raise ValueError("DATABASE_URL environment variable is not set")

    try:
        conn = psycopg2.connect(db_url)
        return conn
    except Exception as e:
        logging.error(f"DB Connection failed: {e}")
        raise

def update_job_progress(job_id, progress_percent):
    """
    Updates the progress of a video job in the database.
    """
    conn = None
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        
        query = "UPDATE videos SET progress = %s WHERE id = %s"
        cur.execute(query, (progress_percent, job_id))
        
        conn.commit()
        cur.close()
    except Exception as e:
        logging.error(f"Failed to update progress for job {job_id}: {e}")
    finally:
        if conn:
            conn.close()

def update_job_status(job_id, status, gcs_enhanced_path=None):
    """
    Updates the status (and optionally the enhanced path) of a job.
    """
    conn = None
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        
        if gcs_enhanced_path:
            query = "UPDATE videos SET status = %s, gcs_enhanced_path = %s, progress = 100 WHERE id = %s"
            cur.execute(query, (status, gcs_enhanced_path, job_id))
        else:
            query = "UPDATE videos SET status = %s WHERE id = %s"
            cur.execute(query, (status, job_id))
            
        conn.commit()
        cur.close()
    except Exception as e:
        logging.error(f"Failed to update status for job {job_id}: {e}")
    finally:
        if conn:
            conn.close()
