import psycopg2
from psycopg2.extras import RealDictCursor
from contextlib import contextmanager
from logger import logger
import config

class DatabaseClient:
    """PostgreSQL database client"""
    
    def __init__(self):
        self.connection_string = config.DATABASE_URL
    
    @contextmanager
    def get_connection(self):
        """Context manager for database connections"""
        conn = None
        try:
            conn = psycopg2.connect(self.connection_string)
            yield conn
            conn.commit()
        except Exception as e:
            if conn:
                conn.rollback()
            logger.error(f'Database error: {e}')
            raise
        finally:
            if conn:
                conn.close()
    
    def update_job_status(self, job_id: str, status: str, progress_percent: int = None, error_message: str = None):
        """Update job status in database"""
        try:
            with self.get_connection() as conn:
                with conn.cursor() as cursor:
                    update_fields = ['status = %s']
                    params = [status]
                    
                    if progress_percent is not None:
                        update_fields.append('progress_percent = %s')
                        params.append(progress_percent)
                    
                    if error_message:
                        update_fields.append('error_message = %s')
                        params.append(error_message)
                    
                    if status == 'PROCESSING':
                        update_fields.append('started_at = NOW()')
                    
                    if status in ['COMPLETED', 'FAILED']:
                        update_fields.append('completed_at = NOW()')
                    
                    params.append(job_id)
                    
                    query = f"UPDATE jobs SET {', '.join(update_fields)} WHERE id = %s"
                    cursor.execute(query, params)
                    
                    logger.info(f'Updated job {job_id} status to {status}')
        except Exception as e:
            logger.error(f'Failed to update job status: {e}')
            raise
    
    def update_video_output_path(self, video_id: str, output_path: str):
        """Update video output storage path"""
        try:
            with self.get_connection() as conn:
                with conn.cursor() as cursor:
                    cursor.execute(
                        "UPDATE videos SET output_storage_path = %s WHERE id = %s",
                        (output_path, video_id)
                    )
                    logger.info(f'Updated video {video_id} output path')
        except Exception as e:
            logger.error(f'Failed to update video output path: {e}')
            raise
    
    def charge_minutes(self, job_id: str, minutes_charged: int):
        """
        Charge minutes for a job (idempotent)
        Only charges if job.minutes_charged is NULL
        """
        try:
            with self.get_connection() as conn:
                with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                    # Check if already charged
                    cursor.execute(
                        "SELECT minutes_charged, user_id FROM jobs WHERE id = %s",
                        (job_id,)
                    )
                    job = cursor.fetchone()
                    
                    if not job:
                        logger.warning(f'Job {job_id} not found')
                        return
                    
                    if job['minutes_charged'] is not None:
                        logger.info(f'Job {job_id} already charged {job["minutes_charged"]} minutes, skipping')
                        return
                    
                    # Update job and user in transaction
                    cursor.execute(
                        "UPDATE jobs SET minutes_charged = %s WHERE id = %s",
                        (minutes_charged, job_id)
                    )
                    
                    cursor.execute(
                        "UPDATE users SET minutes_used_this_month = minutes_used_this_month + %s WHERE id = %s",
                        (minutes_charged, job['user_id'])
                    )
                    
                    logger.info(f'Charged {minutes_charged} minutes for job {job_id}')
        except Exception as e:
            logger.error(f'Failed to charge minutes: {e}')
            raise

db_client = DatabaseClient()
