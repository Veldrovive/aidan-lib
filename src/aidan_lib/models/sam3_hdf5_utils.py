from typing import Iterable
from pathlib import Path
import numpy as np
import h5py
import json
from tqdm import tqdm

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from aidan_lib.models.sam3_lib import SAM3VideoOutput, SAM3FrameOutput

class SegmentationsView:
    """A list-like wrapper to access HDF5 datasets lazily."""
    def __init__(self, frames_group: h5py.Group):
        self.frames_group = frames_group
        self.num_frames = len(frames_group.keys())

    def __len__(self) -> int:
        return self.num_frames

    def __getitem__(self, idx: int) -> h5py.Dataset:
        if idx < 0:
            idx += self.num_frames
        if idx < 0 or idx >= self.num_frames:
            raise IndexError(f"Frame index {idx} out of range.")
        return self.frames_group[f"frame_{idx:04d}"]


class LazySAM3Reader:
    """Context manager for lazily reading SAM3 HDF5 files."""
    def __init__(self, file_path: Path | str):
        self.file_path = Path(file_path)
        self._file: h5py.File | None = None
        
        # These will be populated into memory when the context opens
        self.confidences: list[dict[int, float]] = []
        self.prompt_to_obj_ids: dict[str, list[int]] = {}
        self.video_frame_indices: list[int] = []

    def __enter__(self):
        self._file = h5py.File(self.file_path, "r")
        self.frames_group = self._file["frames"]

        # 1. Load the global prompt map into memory
        self.prompt_to_obj_ids = json.loads(self._file.attrs.get("prompt_to_obj_ids", "{}"))
        
        if "video_frame_indices" in self._file.attrs:
            self.video_frame_indices = self._file.attrs["video_frame_indices"].tolist()
        else:
            # Fallback for old files
            self.video_frame_indices = list(range(len(self.frames_group.keys())))

        # 2. Pre-load all confidences into a memory-bound list
        self.confidences = []
        for i in range(len(self.frames_group.keys())):
            dset = self.frames_group[f"frame_{i:04d}"]
            raw_conf = json.loads(dset.attrs.get("confidences", "{}"))
            self.confidences.append({int(k): v for k, v in raw_conf.items()})

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._file:
            self._file.close()

    @property
    def segmentations(self) -> SegmentationsView:
        """Returns the lazy list-like wrapper for accessing h5py datasets."""
        if not self._file:
            raise RuntimeError("File is closed. Must be used within a 'with' statement.")
        return SegmentationsView(self.frames_group)


class SAM3HDF5Manager:
    def __init__(self, file_path: Path | str):
        self.file_path = Path(file_path)

    def save_video(self, video_output: "SAM3VideoOutput"):
        """Saves an entire pre-computed SAM3VideoOutput object to the file."""
        with h5py.File(self.file_path, "w") as f:
            frames_group = f.create_group("frames")

            # Store the global prompt map at the root of the file
            f.attrs["prompt_to_obj_ids"] = json.dumps(video_output.prompt_to_obj_ids)
            f.attrs["video_frame_indices"] = np.array(video_output.video_frame_indices, dtype=np.int32)

            for i, seg_tensor in enumerate(video_output.segmentation):
                # Move to CPU and convert to numpy
                seg_array = seg_tensor.cpu().numpy()
                dset = frames_group.create_dataset(
                    name=f"frame_{i:04d}",
                    data=seg_array,
                    compression="gzip"
                )
                dset.attrs["confidences"] = json.dumps(video_output.confidences[i])
                dset.attrs["video_frame_index"] = video_output.video_frame_indices[i]

    def save_stream(self, frame_stream: Iterable["SAM3FrameOutput"], progress_bar: bool = False):
        """Iterates through a stream of SAM3FrameOutputs, appending them sequentially."""
        with h5py.File(self.file_path, "w") as f:
            frames_group = f.create_group("frames")
            global_prompts: dict[str, set[int]] = {}
            global_indices: list[int] = []

            if progress_bar:
                frame_stream = tqdm(frame_stream, desc="Saving SAM3 video")

            for i, frame_output in enumerate(frame_stream):
                seg_array = frame_output.segmentation.cpu().numpy()
                dset = frames_group.create_dataset(
                    name=f"frame_{i:04d}",
                    data=seg_array,
                    compression="gzip"
                )
                dset.attrs["confidences"] = json.dumps(frame_output.confidences)
                dset.attrs["video_frame_index"] = frame_output.video_frame_index
                
                global_indices.append(frame_output.video_frame_index)

                # Accumulate new object IDs natively using sets
                for prompt, obj_ids in frame_output.prompt_to_obj_ids.items():
                    if prompt not in global_prompts:
                        global_prompts[prompt] = set()
                    global_prompts[prompt].update(obj_ids)

                # Overwrite the global map continuously. 
                # Doing this per-frame ensures data is preserved if the stream crashes halfway.
                current_prompts = {k: sorted(list(v)) for k, v in global_prompts.items()}
                f.attrs["prompt_to_obj_ids"] = json.dumps(current_prompts)
                f.attrs["video_frame_indices"] = np.array(global_indices, dtype=np.int32)

    def read(self) -> LazySAM3Reader:
        """Returns a context manager for safely reading the file."""
        return LazySAM3Reader(self.file_path)