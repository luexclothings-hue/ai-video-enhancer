# Stream-DiffVSR Integration Guide

This guide explains how to integrate the Stream-DiffVSR model into the worker once you have the repository set up.

## Setup

### 1. Clone Stream-DiffVSR

```bash
cd backend/apps/worker
git clone https://github.com/jamichss/Stream-DiffVSR.git stream_diffvsr
cd stream_diffvsr
```

### 2. Download Pretrained Weights

Visit the [Stream-DiffVSR project page](https://jamichss.github.io/stream-diffvsr-project-page/) to download pretrained weights.

Typically, you'll download:
- `streamdiffvsr_pretrained.pth`

Place in:
```
stream_diffvsr/checkpoints/streamdiffvsr_pretrained.pth
```

### 3. Install Model Dependencies

The model may have additional dependencies. Check `stream_diffvsr/requirements.txt` and install:

```bash
pip install -r stream_diffvsr/requirements.txt
```

## Integration

### Update `video_processor.py`

Replace the placeholder model loading and inference code with actual Stream-DiffVSR API calls.

#### Example Integration

```python
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'stream_diffvsr'))

# Import Stream-DiffVSR modules
from stream_diffvsr.inference import StreamDiffVSR  # Adjust based on actual API
import torch

class VideoProcessor:
    def __init__(self):
        self.ffmpeg = FFmpegPipeline()
        self.device = config.DEVICE
        self.model = None
        
    def load_model(self):
        """Load Stream-DiffVSR model"""
        if self.model is not None:
            return
        
        try:
            logger.info('Loading Stream-DiffVSR model...')
            
            # Load model (adjust based on actual API)
            self.model = StreamDiffVSR(
                checkpoint_path=config.MODEL_CHECKPOINT_PATH,
                device=self.device
            )
            self.model.eval()
            
            logger.info(f'Model loaded successfully on {self.device}')
            
        except Exception as e:
            logger.error(f'Failed to load model: {e}')
            raise
    
    def enhance_frames(self, input_dir: str, output_dir: str, total_frames: int):
        """Enhance video frames using Stream-DiffVSR"""
        try:
            os.makedirs(output_dir, exist_ok=True)
            
            logger.info(f'Enhancing {total_frames} frames...')
            
            frame_files = sorted([
                f for f in os.listdir(input_dir) if f.startswith('frame_')
            ])
            
            # Process frames
            for i in tqdm(range(0, len(frame_files), config.BATCH_SIZE)):
                batch_files = frame_files[i:i + config.BATCH_SIZE]
                
                # Load batch as tensors
                batch_tensors = []
                for frame_file in batch_files:
                    img = Image.open(os.path.join(input_dir, frame_file))
                    # Convert to tensor and normalize (adjust based on model input requirements)
                    tensor = torch.from_numpy(np.array(img)).float() / 255.0
                    tensor = tensor.permute(2, 0, 1)  # HWC -> CHW
                    batch_tensors.append(tensor)
                
                batch = torch.stack(batch_tensors).to(self.device)
                
                # Run inference
                with torch.no_grad():
                    enhanced = self.model(batch)
                
                # Save enhanced frames
                for j, frame_file in enumerate(batch_files):
                    # Convert tensor back to image
                    output_tensor = enhanced[j].cpu()
                    output_array = (output_tensor.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
                    output_img = Image.fromarray(output_array)
                    output_img.save(os.path.join(output_dir, frame_file))
            
            logger.info('Frame enhancement completed')
            
        except Exception as e:
            logger.error(f'Frame enhancement failed: {e}')
            raise
```

### Important Notes

1. **API Variations**: The actual Stream-DiffVSR API may differ. Check the repository's README and example code.

2. **Input Format**: Verify the expected input format:
   - Image normalization (0-1 or 0-255)
   - Tensor shape (NCHW vs NHWC)
   - Color space (RGB vs BGR)

3. **Batch Processing**: Adjust `BATCH_SIZE` based on GPU memory:
   - T4 (16GB): batch_size = 4-8
   - V100 (32GB): batch_size = 16-32

4. **Temporal Consistency**: Stream-DiffVSR may process frames sequentially for temporal consistency. Check if the model requires previous frames as context.

## Testing

Test the integration with a short video:

```bash
cd backend/apps/worker

# Set test environment
export DEVICE=cuda
export MODEL_CHECKPOINT_PATH=./stream_diffvsr/checkpoints/streamdiffvsr_pretrained.pth

# Run a test
python -c "
from video_processor import processor
processor.load_model()
print('Model loaded successfully!')
"
```

## Troubleshooting

### CUDA Out of Memory
- Reduce `BATCH_SIZE` in `.env`
- Process lower resolution
- Use mixed precision (FP16)

### Model Loading Errors
- Verify checkpoint path
- Check CUDA version compatibility
- Ensure all dependencies installed

### Quality Issues
- Check input preprocessing
- Verify output postprocessing
- Adjust model parameters

## Performance Optimization

1. **Mixed Precision**: Use `torch.cuda.amp` for faster inference
2. **TensorRT**: Convert model to TensorRT for 2-3x speedup
3. **Frame Sampling**: Process every Nth frame for preview mode
4. **Async I/O**: Overlap file I/O with GPU processing

## Reference

- Stream-DiffVSR GitHub: https://github.com/jamichss/Stream-DiffVSR
- Project Page: https://jamichss.github.io/stream-diffvsr-project-page/
- Paper: [Link to paper if available]

---

**Last Updated:** 2026-01-07
