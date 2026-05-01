import cv2
import math
from pathlib import Path
from PIL import Image
from typing import Iterator, List, Dict, Any, TypedDict, Union
import numpy as np

class ConstrainedScene(TypedDict):
    shot_id: Union[str, int]
    start_frame: int
    end_frame: int
    start_time: float
    end_time: float
    probability: float


def load_batched_frames(
    vid_path: Path, 
    batch_size: int = 120, 
    skip_frames: int | None = None, 
    convert_pil: bool = True,
    overlap: int = 0
) -> Iterator[tuple[list[np.ndarray] | list[Image.Image], list[int]]]:
    if overlap >= batch_size:
        raise ValueError("overlap must be less than batch_size")

    cap = cv2.VideoCapture(vid_path.absolute().as_posix())

    if not cap.isOpened():
        raise FileNotFoundError(f"{vid_path} does not exist")

    global_frame = -1
    batch = []
    batch_frames = []
    has_unyielded_frames = False
    
    while cap.isOpened():
        global_frame += 1
        # ret is a boolean (True if frame is read), frame is the image data
        ret, frame = cap.read()

        if not ret:
            # Then we finished iterating
            break

        if skip_frames is not None and global_frame % skip_frames != 0:
            continue

        # Convert to a PIL image
        if convert_pil:
            color_converted = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = Image.fromarray(color_converted)

        batch.append(frame)
        batch_frames.append(global_frame)
        has_unyielded_frames = True

        if len(batch) >= batch_size:
            yield batch, batch_frames
            has_unyielded_frames = False
            if overlap > 0:
                batch = batch[-overlap:]
                batch_frames = batch_frames[-overlap:]
            else:
                batch, batch_frames = [], []
    
    if has_unyielded_frames:
        yield batch, batch_frames


def load_constrained_batched_frames(
    vid_path: Path, 
    constrained_scenes: List[ConstrainedScene],
    batch_size: int = 120, 
    skip_frames: int | None = None, 
    convert_pil: bool = True,
    overlap: int = 0,
    overlap_internal_only: bool = True
) -> Iterator[tuple[list[np.ndarray] | list[Image.Image], list[int], bool]]:
    if overlap >= batch_size:
        raise ValueError("overlap must be less than batch_size")

    cap = cv2.VideoCapture(vid_path.absolute().as_posix())

    if not cap.isOpened():
        raise FileNotFoundError(f"{vid_path} does not exist")

    global_frame = -1
    batch = []
    batch_frames = []
    has_unyielded_frames = False
    
    if not constrained_scenes:
        return
        
    scene_index = 0
    current_scene = constrained_scenes[scene_index]

    def get_total_frames(start: int, end: int, skip: int | None) -> int:
        if skip is None:
            return max(0, end - start + 1)
        first_frame = start + (skip - (start % skip)) % skip
        last_frame = end - (end % skip)
        if first_frame > last_frame:
            return 0
        return (last_frame - first_frame) // skip + 1

    def get_target_batch_size(total: int, read: int, max_size: int, overlap_val: int, current_len: int) -> int:
        remaining = total - read
        if remaining <= 0:
            return max_size
            
        first_batch_capacity = max_size - current_len
        if remaining <= first_batch_capacity:
            return current_len + remaining
            
        max_new = max(1, max_size - overlap_val)
        num_batches = 1 + math.ceil((remaining - first_batch_capacity) / max_new)
        
        target_new = math.ceil(remaining / num_batches)
        
        return min(max_size, current_len + target_new)

    scene_frames_total = get_total_frames(current_scene['start_frame'], current_scene['end_frame'], skip_frames)
    scene_frames_read = 0
    current_target_batch_size = get_target_batch_size(scene_frames_total, scene_frames_read, batch_size, overlap, len(batch))

    while cap.isOpened():
        global_frame += 1
        ret, frame = cap.read()

        if not ret:
            break

        while global_frame > current_scene['end_frame']:
            if has_unyielded_frames:
                yield batch, batch_frames, True
                has_unyielded_frames = False
                
                if overlap > 0 and not overlap_internal_only:
                    batch = batch[-overlap:]
                    batch_frames = batch_frames[-overlap:]
                else:
                    batch, batch_frames = [], []
                
            scene_index += 1
            if scene_index < len(constrained_scenes):
                current_scene = constrained_scenes[scene_index]
                scene_frames_total = get_total_frames(current_scene['start_frame'], current_scene['end_frame'], skip_frames)
                scene_frames_read = 0
                current_target_batch_size = get_target_batch_size(scene_frames_total, scene_frames_read, batch_size, overlap, len(batch))
            else:
                break
                
        if scene_index >= len(constrained_scenes):
            break
            
        if global_frame < current_scene['start_frame']:
            continue

        if skip_frames is not None and global_frame % skip_frames != 0:
            continue

        if convert_pil:
            color_converted = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = Image.fromarray(color_converted)

        batch.append(frame)
        batch_frames.append(global_frame)
        scene_frames_read += 1
        has_unyielded_frames = True

        if len(batch) >= current_target_batch_size:
            is_end_of_scene = scene_frames_read >= scene_frames_total
            yield batch, batch_frames, is_end_of_scene
            has_unyielded_frames = False
            
            if overlap > 0 and (not is_end_of_scene or not overlap_internal_only):
                batch = batch[-overlap:]
                batch_frames = batch_frames[-overlap:]
            else:
                batch, batch_frames = [], []
                
            current_target_batch_size = get_target_batch_size(scene_frames_total, scene_frames_read, batch_size, overlap, len(batch))
    
    if has_unyielded_frames:
        yield batch, batch_frames, True