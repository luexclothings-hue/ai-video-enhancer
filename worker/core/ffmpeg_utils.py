import os
import subprocess
import logging

def extract_frames(video_path, output_dir):
    """
    Extracts frames from a video file into a directory using FFmpeg.
    Returns the FPS of the source video.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 1. Get FPS
    try:
        fps_cmd = [
            "ffprobe", 
            "-v", "error", 
            "-select_streams", "v:0", 
            "-show_entries", "stream=r_frame_rate", 
            "-of", "default=noprint_wrappers=1:nokey=1", 
            video_path
        ]
        result = subprocess.run(fps_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True)
        fps_str = result.stdout.strip()
        
        # dynamic eval of "30000/1001" or "30"
        if '/' in fps_str:
            num, den = map(int, fps_str.split('/'))
            fps = num / den
        else:
            fps = float(fps_str)
            
        logging.info(f"Detected FPS: {fps}")

    except Exception as e:
        logging.error(f"Failed to detect FPS: {e}")
        fps = 30.0 # Fallback

    # 2. Extract Frames
    # %05d.png guarantees correct sorting for up to 99999 frames
    output_pattern = os.path.join(output_dir, "%05d.png")
    
    cmd = [
        "ffmpeg",
        "-i", video_path,
        "-vsync", "0",
        output_pattern,
        "-y" # Overwrite
    ]

    try:
        logging.info(f"Extracting frames from {video_path} to {output_dir}")
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except subprocess.CalledProcessError as e:
        logging.error(f"FFmpeg extraction failed: {e.stderr}")
        raise RuntimeError("Frame extraction failed")

    return fps

def stitch_frames(frames_dir, output_video_path, fps):
    """
    Stitches frames from a directory back into a video file using FFmpeg.
    """
    # Pattern must match extract_frames (e.g. %05d.png)
    input_pattern = os.path.join(frames_dir, "%05d.png")
    
    cmd = [
        "ffmpeg",
        "-framerate", str(fps),
        "-i", input_pattern,
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-crf", "18", # High quality
        "-preset", "slow",
        output_video_path,
        "-y"
    ]

    try:
        logging.info(f"Stitching frames from {frames_dir} to {output_video_path}")
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except subprocess.CalledProcessError as e:
        logging.error(f"FFmpeg stitching failed: {e.stderr}")
        raise RuntimeError("Frame stitching failed")
    
    return output_video_path
