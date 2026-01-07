import ffmpeg
import os
from typing import Tuple, Dict
from logger import logger

class FFmpegPipeline:
    """FFmpeg utilities for video processing"""
    
    @staticmethod
    def get_video_info(video_path: str) -> Dict:
        """Extract video metadata using ffprobe"""
        try:
            probe = ffmpeg.probe(video_path)
            video_stream = next((stream for stream in probe['streams'] if stream['codec_type'] == 'video'), None)
            
            if not video_stream:
                raise ValueError('No video stream found')
            
            return {
                'duration': float(probe['format']['duration']),
                'width': int(video_stream['width']),
                'height': int(video_stream['height']),
                'fps': eval(video_stream['r_frame_rate']),  # e.g., "30/1" -> 30.0
                'codec': video_stream['codec_name'],
            }
        except Exception as e:
            logger.error(f'Failed to get video info: {e}')
            raise
    
    @staticmethod
    def extract_frames(video_path: str, output_dir: str, fps: int = 30) -> Tuple[int, str]:
        """
        Extract frames from video
        Returns: (frame_count, frames_pattern)
        """
        try:
            os.makedirs(output_dir, exist_ok=True)
            frames_pattern = os.path.join(output_dir, 'frame_%06d.png')
            
            logger.info(f'Extracting frames from {video_path} at {fps} fps')
            
            (
                ffmpeg
                .input(video_path)
                .filter('fps', fps=fps)
                .output(frames_pattern, format='image2', start_number=0)
                .overwrite_output()
                .run(capture_stdout=True, capture_stderr=True)
            )
            
            # Count extracted frames
            frame_count = len([f for f in os.listdir(output_dir) if f.startswith('frame_')])
            
            logger.info(f'Extracted {frame_count} frames')
            return frame_count, frames_pattern
            
        except ffmpeg.Error as e:
            logger.error(f'FFmpeg error: {e.stderr.decode()}')
            raise
    
    @staticmethod
    def encode_frames_to_video(
        frames_pattern: str,
        output_path: str,
        fps: int = 30,
        crf: int = 18,
        preset: str = 'slow'
    ) -> None:
        """
        Encode frames back to video
        
        Args:
            frames_pattern: Path pattern for input frames (e.g., 'frame_%06d.png')
            output_path: Output video path
            fps: Frame rate
            crf: Constant Rate Factor (0-51, lower = better quality)
            preset: Encoding preset (ultrafast, superfast, veryfast, faster, fast, medium, slow, slower, veryslow)
        """
        try:
            logger.info(f'Encoding frames to video: {output_path}')
            
            (
                ffmpeg
                .input(frames_pattern, format='image2', framerate=fps)
                .output(
                    output_path,
                    vcodec='libx264',
                    crf=crf,
                    preset=preset,
                    pix_fmt='yuv420p'
                )
                .overwrite_output()
                .run(capture_stdout=True, capture_stderr=True)
            )
            
            logger.info(f'Video encoded successfully: {output_path}')
            
        except ffmpeg.Error as e:
            logger.error(f'FFmpeg encoding error: {e.stderr.decode()}')
            raise
    
    @staticmethod
    def add_audio_to_video(
        video_path: str,
        audio_source: str,
        output_path: str
    ) -> None:
        """
        Add audio from source video to processed video
        """
        try:
            logger.info('Adding audio to processed video')
            
            video_input = ffmpeg.input(video_path)
            audio_input = ffmpeg.input(audio_source)
            
            (
                ffmpeg
                .output(
                    video_input,
                    audio_input.audio,
                    output_path,
                    vcodec='copy',
                    acodec='aac',
                    shortest=None
                )
                .overwrite_output()
                .run(capture_stdout=True, capture_stderr=True)
            )
            
            logger.info('Audio added successfully')
            
        except ffmpeg.Error as e:
            logger.error(f'FFmpeg audio merge error: {e.stderr.decode()}')
            raise
