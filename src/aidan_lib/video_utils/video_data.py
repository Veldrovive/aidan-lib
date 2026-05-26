from dataclasses import dataclass
from pathlib import Path
import cv2

@dataclass
class GenericVideoData:
    fps: float
    width: int
    height: int
    frame_count: int

def get_video_data(vid_path: Path) -> GenericVideoData:
    video = cv2.VideoCapture(str(vid_path))

    try:
        fps = video.get(cv2.CAP_PROP_FPS)
        width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    finally:
        video.release()

    return GenericVideoData(
        fps=fps,
        width=width,
        height=height,
        frame_count=frame_count
    )
