try:
    import transnetv2_pytorch as transnet
except ImportError:
    raise ImportError("transnetv2_pytorch not installed. Install it the optional dependency aidan-lib[transnet]")

from .load_batched_frames import ConstrainedScene

import cv2
from pathlib import Path
import torch

def get_transnet_model(device: str | torch.device = "cuda") -> transnet.TransNetV2:
    original_dtype = torch.get_default_dtype()
    torch.set_default_dtype(torch.float32)
    try:
        model = transnet.TransNetV2()
        model = model.to(device, dtype=torch.float32)
        model.eval()
        return model
    finally:
        torch.set_default_dtype(original_dtype)

def get_constrained_scenes(
    video_path: str | Path,
    model: "transnet.TransNetV2 | None" = None,
    threshold: float = 0.2
) -> list[ConstrainedScene]:
    """
    Extract scenes from a video.
    
    Args:
        video_path: Path to the video file.
        model: An optional, pre-loaded TransNetV2 model instance.
        threshold: Scene detection threshold.
        
    Returns:
        A list of ConstrainedScene objects.
    """
    if model is None:
        model = get_transnet_model()

    original_dtype = torch.get_default_dtype()
    torch.set_default_dtype(torch.float32)
    try:
        with torch.autocast(device_type="cuda", enabled=False):
            scenes = model.detect_scenes(str(video_path), threshold=threshold)
    finally:
        torch.set_default_dtype(original_dtype)
    
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    
    constrained_scenes = []
    for scene in scenes:
        start_frame = scene['start_frame']
        end_frame = scene['end_frame']
        
        constrained_scenes.append(ConstrainedScene(
            shot_id=scene['shot_id'],
            start_frame=start_frame,
            end_frame=end_frame,
            start_time=float(start_frame / fps) if fps else 0.0,
            end_time=float(end_frame / fps) if fps else 0.0,
            probability=float(scene['probability'])
        ))
            
    return constrained_scenes