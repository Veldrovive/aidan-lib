from transformers.generation.continuous_batching import input_outputs
import os
import tempfile
import itertools
import json
from pathlib import Path
from typing import Iterable, Generator, NamedTuple, TypedDict
import cv2
import imageio.v3 as iio
import numpy as np
import h5py
import torch
import kornia.morphology as morph
import kornia.contrib as contrib
from jaxtyping import Int, Float, Bool
from dataclasses import dataclass
from tqdm import tqdm
from PIL import Image
import time
from abc import ABC, abstractmethod
from multiprocessing import shared_memory, resource_tracker
from multiprocessing.connection import Listener, Client
import contextlib
from dataclasses import dataclass

try:
    from sam3.model_builder import build_sam3_multiplex_video_predictor
except ImportError:
    raise ImportError("Meta SAM3 not installed. Make sure you installed it via 'pip install git+https://github.com/facebookresearch/sam3.git'")

SEG_ID_TYPE = np.int16
MAX_SEG_ID = np.iinfo(SEG_ID_TYPE).max
BACKGROUND_SEG_ID = MAX_SEG_ID

TextPrompt = str

VideoSegmentation = Int[np.ndarray, "n_frames height width"]
VideoConfidences = list[dict[int, float]]  # Length equals n_frames
VideoPrompts = dict[int, str]  # Maps from tracklet id to prompt text

@dataclass
class SAM3VideoOutput:
    background_index = BACKGROUND_SEG_ID
    segmentation: VideoSegmentation
    confidences: VideoConfidences
    obj_id_to_prompt: VideoPrompts
    video_frame_indices: list[int]


FrameSegmentation = Int[np.ndarray, "height width"]
FrameConfidences = dict[int, float]

@dataclass
class SAM3FrameOutput:
    background_index = BACKGROUND_SEG_ID
    segmentation: FrameSegmentation
    confidences: FrameConfidences
    video_frame_index: int

class BaseSAM3Harness(ABC):
    @abstractmethod
    def close_session(self, session_id: str | None = None):
        pass

    @abstractmethod
    def reset_session(self, session_id: str | None = None):
        pass

    @abstractmethod
    def start_session(self, video: Path | list[Image.Image], frame_numbers: list[int] | None = None, offload_state_to_cpu: bool | None = None, store_session: bool = True) -> str:
        pass

    @abstractmethod
    def add_prompt(self, prompt: TextPrompt, frame_index: int = 0, session_id: str | None = None):
        pass

    @abstractmethod
    def propagate_session(self, session_id: str | None = None) -> SAM3VideoOutput:
        pass

    def __call__(
        self,
        prompt: TextPrompt | list[TextPrompt],
        video: Path | list[Image.Image],
        prompt_frame: int = 0,
        frame_numbers: list[int] | None = None,
        offload_state_to_cpu: bool | None = None
    ) -> SAM3VideoOutput:
        session_id = self.start_session(video, frame_numbers, offload_state_to_cpu, store_session=False)
        try:
            if isinstance(prompt, list):
                for p in prompt:
                    self.add_prompt(p, prompt_frame, session_id)
            else:
                self.add_prompt(prompt, prompt_frame, session_id)
            output = self.propagate_session(session_id)
        finally:
            self.close_session(session_id)

        return output

class SAM3Harness(BaseSAM3Harness):
    def __init__(
        self, 
        device: str | torch.device = "cuda",
        dtype: torch.dtype = torch.bfloat16,
        inference_device: str | torch.device | None = None,
        processing_device: str | torch.device = "cpu",
        video_storage_device: str | torch.device = "cpu",
        max_num_objects: int = 16,
        compile: bool = False,
        warm_up: bool = False,
        score_threshold: float = 0.5,
    ):
        self.device = torch.device(device)
        self.dtype = dtype
        self.inference_device = torch.device(inference_device) if inference_device is not None else self.device
        self.processing_device = torch.device(processing_device)
        self.video_storage_device = torch.device(video_storage_device)
        
        # Initialize Meta's multiplex video predictor
        print(f"Constructing multiplex predictor")
        print(f"WARNING: DISABLING FA3")
        self.predictor = build_sam3_multiplex_video_predictor(use_fa3=False, compile=compile, warm_up=warm_up, max_num_objects=max_num_objects, default_output_prob_thresh=score_threshold)
        print(f"Created multiplex predictor")

        self.main_session_id: str | None = None

        self.session_metadata: dict[str, SAM3Harness.SessionMetadata] = {}

    @dataclass
    class SessionMetadata:
        session_id: str
        offload_state_to_cpu: bool | None
        frame_numbers: list[int] | None
        current_prompt: str | None

    def start_session(self, video: Path | list[Image.Image], frame_numbers: list[int] | None = None, offload_state_to_cpu: bool | None = None, store_session: bool = True) -> str:
        if self.main_session_id is not None:
            raise ValueError("Session already started. Please call close_session first.")

        response = self.predictor.handle_request(
            request=dict(
                type="start_session",
                resource_path=video,
                offload_state_to_cpu=None
            )
        )
        new_session_id = response["session_id"]
        if store_session:
            self.main_session_id = new_session_id

        session_meta = SAM3Harness.SessionMetadata(
            session_id=new_session_id,
            offload_state_to_cpu=offload_state_to_cpu,
            frame_numbers=frame_numbers,
            current_prompt=None
        )
        self.session_metadata[new_session_id] = session_meta

        return new_session_id
    
    def reset_session(self, session_id: str | None = None):
        if session_id is None:
            session_id = self.main_session_id
            if session_id is None:
                raise ValueError("No session_id provided and no active main session.")
        
        _ = self.predictor.handle_request(
            request=dict(
                type="reset_session",
                session_id=session_id,
            )
        )
        self.session_metadata[session_id].current_prompt = None

    def add_prompt(self, prompt: TextPrompt, frame_index: int = 0, session_id: str | None = None):
        if session_id is None:
            session_id = self.main_session_id
            if session_id is None:
                raise ValueError("No session_id provided and no active main session.")

        resp = self.predictor.handle_request(
            request=dict(
                type="add_prompt",
                session_id=session_id,
                frame_index=frame_index,
                text=prompt,
            )
        )
        self.session_metadata[session_id].current_prompt = prompt
        return resp

    def close_session(self, session_id: str | None = None):
        if session_id is None:
            session_id = self.main_session_id
            if session_id is None:
                raise ValueError("No session_id provided and no active main session.")
        
        _ = self.predictor.handle_request(
            request=dict(
                type="close_session",
                session_id=session_id,
            )
        )
        if session_id == self.main_session_id:
            self.main_session_id = None

        if session_id in self.session_metadata:
            del self.session_metadata[session_id]

    class SAMOutType(NamedTuple):
        masks: np.ndarray
        scores: list[float]
        out_obj_ids: list[int]
        bboxs: np.ndarray

    def propagate_session(self, session_id: str | None = None):
        if session_id is None:
            session_id = self.main_session_id
            if session_id is None:
                raise ValueError("No session_id provided and no active main session.")
        
        current_meta = self.session_metadata[session_id]
        frame_number_map = current_meta.frame_numbers
        current_prompt = current_meta.current_prompt
        assert current_prompt is not None, "No prompt added to session."

        outputs: dict[int, SAM3Harness.SAMOutType] = {}
        for response in self.predictor.handle_stream_request(
            request=dict(
                type="propagate_in_video",
                session_id=session_id,
            )
        ):
            input_frame_num = response["frame_index"]

            model_outputs = response["outputs"]
            masks = model_outputs["out_binary_masks"]
            scores = model_outputs["out_probs"]
            out_obj_ids = model_outputs["out_obj_ids"]
            bboxs = model_outputs["out_boxes_xywh"]

            for out_obj_id in out_obj_ids:
                if out_obj_id > MAX_SEG_ID:
                    raise ValueError("Segmentation id exceeds maximum allowed.")
                if out_obj_id == BACKGROUND_SEG_ID:
                    raise ValueError("Segmentation id equals background id.")

            if frame_number_map is None:
                frame_num = input_frame_num
            else:
                frame_num = frame_number_map[input_frame_num]

            outputs[frame_num] = SAM3Harness.SAMOutType(masks, scores, out_obj_ids, bboxs)

        # Pack into a video output
        n_frames = len(outputs)
        assert n_frames > 0, "No frames were processed."
        first_frame_out = next(iter(outputs.values()))
        height, width = first_frame_out.masks.shape[1:]

        segmentation = np.full((n_frames, height, width), fill_value=BACKGROUND_SEG_ID, dtype=SEG_ID_TYPE)
        confidences = [dict() for _ in range(n_frames)]
        video_frame_indices = []

        output_frame_indices = sorted(outputs.keys())

        prompts = {}
        for input_frame_num, output_frame_num in enumerate(output_frame_indices):
            frame_out = outputs[output_frame_num]
            video_frame_indices.append(output_frame_num)
            
            for i, obj_id in enumerate(frame_out.out_obj_ids):
                mask = frame_out.masks[i]
                conf = frame_out.scores[i]

                confidences[input_frame_num][obj_id] = conf
                
                mask_locations = np.where(mask)
                segmentation[input_frame_num, mask_locations[0], mask_locations[1]] = obj_id
                prompts[obj_id] = current_prompt

        video_output = SAM3VideoOutput(
            segmentation=segmentation,
            confidences=confidences,
            obj_id_to_prompt=prompts,
            video_frame_indices=video_frame_indices,
        )

        return video_output


class FrameSegmentationInfo(NamedTuple):
    global_frame_num: int
    frame: np.ndarray | Image.Image
    segmentation: np.ndarray
    background_index: int
    obj_id_to_prompt: dict[int, str]

def compute_overlap_ids(
    overlap_frames: list[int], 
    last_out: SAM3VideoOutput, 
    cur_out: SAM3VideoOutput, 
    iou_thresh: float = 0.9
) -> list[tuple[int, int]]:
    """
    Returns a list of equality tuples.
    (last_obj_id, cur_obj_id) where we know that these refer to the same object
    """

    equality_counts = {}
    for global_frame in overlap_frames:
        last_frame_index = last_out.video_frame_indices.index(global_frame)
        cur_frame_index = cur_out.video_frame_indices.index(global_frame)

        last_seg = last_out.segmentation[last_frame_index]
        cur_seg = cur_out.segmentation[cur_frame_index]

        last_visible_obj_ids = [obj_id for obj_id in np.unique(last_seg) if obj_id != last_out.background_index]
        cur_visible_obj_ids = [obj_id for obj_id in np.unique(cur_seg) if obj_id != cur_out.background_index]

        for last_obj_id in last_visible_obj_ids:
            last_mask = last_seg == last_obj_id
            for cur_obj_id in cur_visible_obj_ids:
                # Only check for overlap if the prompts match
                if last_out.obj_id_to_prompt.get(last_obj_id) != cur_out.obj_id_to_prompt.get(cur_obj_id):
                    continue

                key = (int(last_obj_id), int(cur_obj_id))
                if key not in equality_counts:
                    equality_counts[key] = 0

                cur_mask = cur_seg == cur_obj_id

                intersection = np.count_nonzero(last_mask & cur_mask)
                
                if intersection == 0:
                    continue

                union = np.count_nonzero(last_mask | cur_mask)
                iou = intersection / union

                if iou > iou_thresh:
                    equality_counts[key] += 1

    equalities = []
    for key, equality_count in equality_counts.items():
        if equality_count == len(overlap_frames):
            equalities.append(key)

    return equalities

def generate_video_segmentation(
    harness: BaseSAM3Harness,
    prompts: list[str] | str,
    batch_frame_loader: Iterable,
    iou_thresh: float = 0.9
) -> Generator[FrameSegmentationInfo, None, None]:
    if isinstance(prompts, str):
        prompts = [prompts]

    last_frame_numbers_set: set | None = None
    last_out: SAM3VideoOutput | None = None
    next_unique_id = 0
    last_obj_id_unique_id_assignments: dict | None = None
    global_unique_id_to_prompt: dict[int, str] = {}

    last_global_frame = -1
    for frame_batch, frame_numbers, scene_done in batch_frame_loader:
        new_frame_numbers_set = set(frame_numbers)

        session_id = harness.start_session(frame_batch, frame_numbers=frame_numbers, offload_state_to_cpu=None, store_session=False)
        
        combined_out = None
        next_combined_id = 0

        try:
            for prompt in prompts:
                harness.add_prompt(prompt, session_id=session_id)
                out = harness.propagate_session(session_id=session_id)
                
                if combined_out is None:
                    combined_out = SAM3VideoOutput(
                        segmentation=np.full_like(out.segmentation, out.background_index),
                        confidences=[dict() for _ in range(len(out.confidences))],
                        obj_id_to_prompt={},
                        video_frame_indices=out.video_frame_indices,
                    )
                
                unique_ids = np.unique(out.segmentation)
                unique_ids = unique_ids[unique_ids != out.background_index]
                
                if len(unique_ids) > 0:
                    obj_id_to_new_id = {}
                    for obj_id in unique_ids:
                        obj_id_to_new_id[obj_id] = next_combined_id
                        next_combined_id += 1
                        
                    for frame_idx in range(len(out.segmentation)):
                        seg = out.segmentation[frame_idx]
                        for obj_id in unique_ids:
                            new_id = obj_id_to_new_id[obj_id]
                            mask = (seg == obj_id)
                            combined_out.segmentation[frame_idx][mask] = new_id
                            
                            if obj_id in out.confidences[frame_idx]:
                                combined_out.confidences[frame_idx][new_id] = out.confidences[frame_idx][obj_id]
                            
                            combined_out.obj_id_to_prompt[new_id] = prompt

                harness.reset_session(session_id=session_id)
        finally:
            harness.close_session(session_id=session_id)
            
        out = combined_out
        if out is None:
            raise RuntimeError("No outputs generated. Are prompts empty?")

        if last_frame_numbers_set:
            overlap = last_frame_numbers_set.intersection(new_frame_numbers_set)
            assert last_out is not None
            equalities = compute_overlap_ids(list(overlap), last_out, out, iou_thresh=iou_thresh)
            cur_obj_id_to_prev_obj_id = {cur_obj_id: last_obj_id for (last_obj_id, cur_obj_id) in equalities}
        else:
            cur_obj_id_to_prev_obj_id = {}

        # Now we need to assign the object ids to unique ids
        seg = out.segmentation
        obj_ids = [int(obj_id) for obj_id in np.unique(seg) if obj_id != out.background_index]
        obj_id_to_unique_id_assignments = {}
        for cur_obj_id in obj_ids:
            # First, we see if there is a match to an old segmentation
            last_obj_id = cur_obj_id_to_prev_obj_id.get(cur_obj_id, None)
            if last_obj_id is None:
                unique_id = next_unique_id
                next_unique_id += 1
            else:
                assert last_obj_id_unique_id_assignments is not None
                unique_id = last_obj_id_unique_id_assignments[last_obj_id]
            obj_id_to_unique_id_assignments[cur_obj_id] = unique_id

        max_obj_id = int(np.max(seg)) if len(obj_ids) > 0 else out.background_index
        
        for cur_obj_id, unique_id in obj_id_to_unique_id_assignments.items():
            if unique_id not in global_unique_id_to_prompt:
                global_unique_id_to_prompt[unique_id] = out.obj_id_to_prompt.get(cur_obj_id, "UNKNOWN")

        # Create a lookup table initialized to map to itself by default
        lookup_table_size = max(max_obj_id, out.background_index) + 1
        lookup_table = np.arange(lookup_table_size, dtype=np.int32)
        
        # Populate the lookup table with the dictionary assignments
        for old_id, new_id in obj_id_to_unique_id_assignments.items():
            lookup_table[old_id] = new_id

        for i in range(len(frame_batch)):
            global_frame_num = frame_numbers[i]
            if global_frame_num == last_global_frame:
                continue
            last_global_frame = global_frame_num
            
            frame = frame_batch[i]
            frame_seg = out.segmentation[i]
            
            # Map the segmentation which uses obj ids to unique ids 
            # This applies the mapping to the entire 2D mask instantly
            unique_frame_seg = lookup_table[frame_seg]

            yield FrameSegmentationInfo(global_frame_num, frame, unique_frame_seg, out.background_index, global_unique_id_to_prompt)

        if scene_done:
            last_frame_numbers_set = None
            last_out = None
            last_obj_id_unique_id_assignments = None
        else:
            last_frame_numbers_set = new_frame_numbers_set
            last_out = out
            last_obj_id_unique_id_assignments = obj_id_to_unique_id_assignments