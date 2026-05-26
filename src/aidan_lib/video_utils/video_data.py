from dataclasses import dataclass
from pathlib import Path
import cv2

@dataclass
class GenericVideoData:
    fps: float

def get_video_data(vid_path: Path) -> GenericVideoData:
    video = cv2.VideoCapture(str(vid_path))

    try:
        fps = video.get(cv2.CAP_PROP_FPS)
    finally:
        video.release()

    return GenericVideoData(
        fps=fps
    )
