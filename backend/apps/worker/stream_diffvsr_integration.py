"""
Stream-DiffVSR Integration Module
Optimized for production video enhancement service
"""

import os
import sys
import torch
from typing import List
from PIL import Image
from torchvision.models.optical_flow import raft_large, Raft_Large_Weights

from logger import logger
import config

# Add Stream-DiffVSR to Python path
STREAM_DIFFVSR_PATH = os.path.join(os.path.dirname(__file__), 'Stream-DiffVSR')
if STREAM_DIFFVSR_PATH not in sys.path:
    sys.path.insert(0, STREAM_DIFFVSR_PATH)

try:
    from pipeline.stream_diffvsr_pipeline import StreamDiffVSRPipeline, ControlNetModel, UNet2DConditionModel
    from diffusers import DDIMScheduler
    from temporal_autoencoder.autoencoder_tiny import TemporalAutoencoderTiny
except ImportError as e:
    logger.error(f"Failed to import Stream-DiffVSR: {e}")
    raise

class StreamDiffVSRProcessor:
    """Production-optimized Stream-DiffVSR processor"""
    
    def __init__(self, model_id: str = 'Jamichsu/Stream-DiffVSR'):
        self.model_id = model_id
        self.pipeline = None
        self.of_model = None
        self.device = torch.device(config.DEVICE)
        self.is_loaded = False
        
        # Performance optimizations
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        
    def load_model(self):
        """Load Stream-DiffVSR pipeline with performance optimizations"""
        if self.is_loaded:
            return
            
        try:
            logger.info(f'Loading Stream-DiffVSR model from {self.model_id}...')
            
            # Load components from HuggingFace Hub
            controlnet = ControlNetModel.from_pretrained(self.model_id, subfolder="controlnet")
            unet = UNet2DConditionModel.from_pretrained(self.model_id, subfolder="unet")
            vae = TemporalAutoencoderTiny.from_pretrained(self.model_id, subfolder="vae")
            scheduler = DDIMScheduler.from_pretrained(self.model_id, subfolder="scheduler")
            
            # Create optimized pipeline
            pipeline_kwargs = {"custom_pipeline": None}
            
            # Enable TensorRT if configured
            if config.ENABLE_TENSORRT:
                pipeline_kwargs.update({
                    "custom_pipeline": "/acceleration/tensorrt/sd_with_controlnet_ST",
                    "image_height": config.TARGET_HEIGHT,
                    "image_width": config.TARGET_WIDTH,
                })
                logger.info("TensorRT acceleration enabled")
            
            self.pipeline = StreamDiffVSRPipeline.from_pretrained(
                self.model_id,
                controlnet=controlnet,
                vae=vae,
                unet=unet,
                scheduler=scheduler,
                **pipeline_kwargs
            )
            
            # Performance optimizations
            self.pipeline = self.pipeline.to(self.device)
            self.pipeline.enable_xformers_memory_efficient_attention()
            
            if config.ENABLE_TENSORRT:
                self.pipeline.set_cached_folder("Jamichsu/Stream-DiffVSR")
            
            # Load optical flow model
            logger.info('Loading RAFT optical flow model...')
            self.of_model = raft_large(weights=Raft_Large_Weights.DEFAULT).to(self.device).eval()
            self.of_model.requires_grad_(False)
            
            self.is_loaded = True
            logger.info(f'Stream-DiffVSR loaded successfully on {self.device}')
            
        except Exception as e:
            logger.error(f'Failed to load Stream-DiffVSR: {e}')
            raise
    
    def enhance_frame_sequence(self, frame_paths: List[str], output_dir: str) -> List[str]:
        """
        Enhance a sequence of frames using Stream-DiffVSR
        
        Args:
            frame_paths: List of input frame file paths
            output_dir: Directory to save enhanced frames
            
        Returns:
            List of enhanced frame file paths
        """
        if not self.is_loaded:
            self.load_model()
        
        try:
            os.makedirs(output_dir, exist_ok=True)
            
            # Load frames as PIL Images
            frames = []
            for frame_path in frame_paths:
                if not os.path.exists(frame_path):
                    raise FileNotFoundError(f"Frame not found: {frame_path}")
                frames.append(Image.open(frame_path))
            
            logger.info(f'Enhancing {len(frames)} frames with Stream-DiffVSR...')
            
            # Run inference using the official API
            with torch.no_grad():
                output = self.pipeline(
                    prompt='',  # Empty prompt as per official usage
                    images=frames,
                    num_inference_steps=config.MODEL_NUM_INFERENCE_STEPS,
                    guidance_scale=0,  # No guidance as per official usage
                    of_model=self.of_model
                )
            
            # Extract enhanced frames
            enhanced_frames = output.images
            frames_to_save = [frame[0] for frame in enhanced_frames]
            
            # Save enhanced frames
            output_paths = []
            for i, (enhanced_frame, original_path) in enumerate(zip(frames_to_save, frame_paths)):
                # Keep original filename
                original_name = os.path.basename(original_path)
                output_path = os.path.join(output_dir, original_name)
                enhanced_frame.save(output_path)
                output_paths.append(output_path)
            
            logger.info(f'Successfully enhanced {len(output_paths)} frames')
            
            # Cleanup
            del frames
            del frames_to_save
            torch.cuda.empty_cache()
            
            return output_paths
            
        except Exception as e:
            logger.error(f'Frame enhancement failed: {e}')
            raise
    
    def enhance_frames_batch(self, input_dir: str, output_dir: str) -> None:
        """Enhance all frames in directory with optimized batching"""
        if not self.is_loaded:
            self.load_model()
        
        try:
            os.makedirs(output_dir, exist_ok=True)
            
            # Get frame files
            frame_files = sorted([
                f for f in os.listdir(input_dir) 
                if f.lower().endswith(('.png', '.jpg', '.jpeg')) and f.startswith('frame_')
            ])
            
            if not frame_files:
                raise ValueError(f"No frames found in {input_dir}")
            
            logger.info(f'Enhancing {len(frame_files)} frames...')
            
            # Dynamic batch size based on GPU memory
            batch_size = self._get_optimal_batch_size()
            
            # Process in optimized batches
            for i in range(0, len(frame_files), batch_size):
                batch_files = frame_files[i:i + batch_size]
                batch_paths = [os.path.join(input_dir, f) for f in batch_files]
                
                # Load frames
                frames = [Image.open(path) for path in batch_paths]
                
                # Run inference
                with torch.no_grad():
                    output = self.pipeline(
                        prompt='',
                        images=frames,
                        num_inference_steps=config.MODEL_NUM_INFERENCE_STEPS,
                        guidance_scale=0,
                        of_model=self.of_model
                    )
                
                # Save enhanced frames
                enhanced_frames = [frame[0] for frame in output.images]
                for enhanced_frame, original_file in zip(enhanced_frames, batch_files):
                    output_path = os.path.join(output_dir, original_file)
                    enhanced_frame.save(output_path)
                
                # Memory cleanup
                del frames, enhanced_frames
                torch.cuda.empty_cache()
                
                # Progress logging
                progress = int((i + len(batch_files)) / len(frame_files) * 100)
                logger.info(f'Enhancement progress: {progress}%')
            
            logger.info('Frame enhancement completed')
            
        except Exception as e:
            logger.error(f'Frame enhancement failed: {e}')
            raise
    
    def _get_optimal_batch_size(self) -> int:
        """Determine optimal batch size based on available GPU memory"""
        try:
            if torch.cuda.is_available():
                gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                if gpu_memory_gb >= 24:  # RTX4090, A100
                    return min(16, config.BATCH_SIZE * 2)
                elif gpu_memory_gb >= 12:  # RTX3080, RTX4080
                    return min(8, config.BATCH_SIZE)
                else:  # T4, RTX3060
                    return min(4, config.BATCH_SIZE)
            return config.BATCH_SIZE
        except:
            return config.BATCH_SIZE

# Global processor instance
stream_diffvsr_processor = StreamDiffVSRProcessor()

if __name__ == '__main__':
    # Test model loading
    try:
        logger.info('Testing Stream-DiffVSR model loading...')
        stream_diffvsr_processor.load_model()
        logger.info('Model loading test successful!')
    except Exception as e:
        logger.error(f'Model loading test failed: {e}')