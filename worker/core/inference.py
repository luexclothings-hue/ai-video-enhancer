import os
import sys
import torch
import logging
from PIL import Image
from diffusers import DDIMScheduler
from transformers import CLIPTextModel, CLIPTokenizer

# Add Stream-DiffVSR to path
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "Stream-DiffVSR"))

from pipeline.stream_diffvsr_pipeline import StreamDiffVSRPipeline, ControlNetModel, UNet2DConditionModel
from temporal_autoencoder.autoencoder_tiny import TemporalAutoencoderTiny
from torchvision.models.optical_flow import raft_large, Raft_Large_Weights

from core.ffmpeg_utils import extract_frames, stitch_frames

class InferenceEngine:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.pipeline = None
        self.of_model = None
        
        # Model IDs
        self.model_id = "stabilityai/stable-diffusion-x4-upscaler"
        self.cache_dir = "/models" # For Cloud Run persistence if using volumes, else default

    def load_models(self):
        """Loads models into memory. Call this once at startup."""
        logging.info("Loading AI Models...")
        
        # 1. Load Components
        controlnet = ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-canny", torch_dtype=torch.float16)
        unet = UNet2DConditionModel.from_pretrained(self.model_id, subfolder="unet", torch_dtype=torch.float16)
        vae = TemporalAutoencoderTiny.from_pretrained("madebyollin/taesd", torch_dtype=torch.float16)
        scheduler = DDIMScheduler.from_pretrained(self.model_id, subfolder="scheduler")
        
        # 2. Build Pipeline
        self.pipeline = StreamDiffVSRPipeline.from_pretrained(
            self.model_id,
            controlnet=controlnet,
            vae=vae,
            unet=unet,
            scheduler=scheduler,
            torch_dtype=torch.float16
        )
        
        self.pipeline.to(self.device)
        self.pipeline.enable_xformers_memory_efficient_attention()
        
        # 3. Load Optical Flow
        self.of_model = raft_large(weights=Raft_Large_Weights.DEFAULT).to(self.device).eval()
        self.of_model.requires_grad_(False)
        
        logging.info("Models loaded successfully.")

    def enhance_video(self, local_video_path, job_id, update_callback=None):
        """
        Runs the enhancement pipeline on a video file.
        Returns the path to the enhanced video.
        """
        if not self.pipeline:
            self.load_models()

        base_dir = os.path.dirname(local_video_path)
        frames_dir = os.path.join(base_dir, "frames_raw")
        enhanced_frames_dir = os.path.join(base_dir, "frames_enhanced")
        os.makedirs(enhanced_frames_dir, exist_ok=True)

        # 1. Extract Frames
        logging.info(f"Extracting frames for job {job_id}...")
        fps = extract_frames(local_video_path, frames_dir)
        
        # 2. Load Frames to Memory
        frame_files = sorted(os.listdir(frames_dir))
        frames = [Image.open(os.path.join(frames_dir, f)).convert("RGB") for f in frame_files]
        total_frames = len(frames)
        logging.info(f"Loaded {total_frames} frames. Starting inference...")

        # 3. Run Inference
        def internal_callback(current, total):
            percent = int((current / total) * 100)
            logging.info(f"Job {job_id}: Processed frame {current}/{total} ({percent}%)")
            if update_callback:
                update_callback(job_id, percent)

        # Run pipeline
        output = self.pipeline(
            prompt="", 
            images=frames, 
            num_inference_steps=10, # Adjustable for speed vs quality
            guidance_scale=0, 
            of_model=self.of_model,
            frame_callback=internal_callback
        )
        
        enhanced_frames = output.images # List[List[PIL]] or List[PIL] depending on output
        # Flatten if needed (StreamDiffVSR usually returns list of single-image lists)
        if isinstance(enhanced_frames[0], list):
             enhanced_frames = [f[0] for f in enhanced_frames]

        # 4. Save Enhanced Frames
        for idx, frame in enumerate(enhanced_frames):
            frame.save(os.path.join(enhanced_frames_dir, f"{idx:05d}.png"))
            
        # 5. Stitch Video
        output_video_path = os.path.join(base_dir, f"enhanced_{job_id}.mp4")
        stitch_frames(enhanced_frames_dir, output_video_path, fps)
        
        return output_video_path
