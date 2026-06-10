from transformers.generation.continuous_batching import input_outputs
import os
import tempfile
import itertools
import json
from pathlib import Path
from typing import Iterable, Generator, NamedTuple, TypedDict, Literal
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
import traceback

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

@dataclass
class SAM3VideoOutput:
    background_index = BACKGROUND_SEG_ID
    segmentation: VideoSegmentation
    confidences: VideoConfidences
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
            prompts = prompt if isinstance(prompt, list) else [prompt]
            has_objects = False
            
            for p in prompts:
                resp = self.add_prompt(p, prompt_frame, session_id)
                if resp is not None and "outputs" in resp:
                    if len(resp["outputs"].get("out_obj_ids", [])) > 0:
                        has_objects = True
            
            if has_objects:
                output = self.propagate_session(session_id)
            else:
                if isinstance(video, list):
                    frame0 = video[0]
                    width, height = frame0.size
                    n_frames = len(video)
                else:
                    import cv2
                    cap = cv2.VideoCapture(str(video))
                    ret, frame = cap.read()
                    if not ret:
                        raise ValueError(f"Could not read video {video}")
                    height, width = frame.shape[:2]
                    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    cap.release()
                
                if frame_numbers is None:
                    frame_numbers = list(range(n_frames))
                
                output = SAM3VideoOutput(
                    segmentation=np.full((n_frames, height, width), SAM3VideoOutput.background_index, dtype=SEG_ID_TYPE),
                    confidences=[dict() for _ in range(n_frames)],
                    video_frame_indices=list(frame_numbers),
                )
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

        for input_frame_num, output_frame_num in enumerate(output_frame_indices):
            frame_out = outputs[output_frame_num]
            video_frame_indices.append(output_frame_num)
            
            for i, obj_id in enumerate(frame_out.out_obj_ids):
                mask = frame_out.masks[i]
                conf = frame_out.scores[i]

                confidences[input_frame_num][obj_id] = conf
                
                mask_locations = np.where(mask)
                segmentation[input_frame_num, mask_locations[0], mask_locations[1]] = obj_id

        video_output = SAM3VideoOutput(
            segmentation=segmentation,
            confidences=confidences,
            video_frame_indices=video_frame_indices,
        )

        return video_output


class FrameSegmentationInfo(NamedTuple):
    global_frame_num: int
    frame: np.ndarray | Image.Image
    segmentations: dict[str, np.ndarray]
    background_index: int

def compute_overlap_ids(
    overlap_frames: list[int], 
    last_out: SAM3VideoOutput, 
    cur_out: SAM3VideoOutput, 
    iou_thresh: float = 0.5
) -> list[tuple[int, int]]:
    """
    Returns a list of equality tuples.
    (last_obj_id, cur_obj_id) where we know that these refer to the same object
    """

    intersections = {}
    unions = {}
    
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
                key = (int(last_obj_id), int(cur_obj_id))
                cur_mask = cur_seg == cur_obj_id

                intersection = int(np.count_nonzero(last_mask & cur_mask))
                union = int(np.count_nonzero(last_mask | cur_mask))
                
                intersections[key] = intersections.get(key, 0) + intersection
                unions[key] = unions.get(key, 0) + union

    # Compute 3D (volumetric) IoU for all candidate pairs
    ious = {}
    for key in intersections.keys():
        if unions[key] > 0:
            iou = intersections[key] / unions[key]
            if iou > iou_thresh:
                ious[key] = iou

    # Greedy 1-to-1 matching to prevent multiple assignments
    equalities = []
    matched_last = set()
    matched_cur = set()
    
    # Sort by IoU descending
    sorted_pairs = sorted(ious.items(), key=lambda x: x[1], reverse=True)
    
    for (last_obj_id, cur_obj_id), iou in sorted_pairs:
        if last_obj_id not in matched_last and cur_obj_id not in matched_cur:
            equalities.append((last_obj_id, cur_obj_id))
            matched_last.add(last_obj_id)
            matched_cur.add(cur_obj_id)

    return equalities

def generate_video_segmentation(
    harness: BaseSAM3Harness,
    prompts: list[str] | str,
    batch_frame_loader: Iterable,
    num_prompt_applications: int = 1,
    prompt_frame_spacing: Literal["space_around", "space_between"] = "space_between",
    iou_thresh: float = 0.5
) -> Generator[FrameSegmentationInfo, None, None]:
    if isinstance(prompts, str):
        prompts = [prompts]

    last_frame_numbers_set: set | None = None
    last_outs: dict[str, SAM3VideoOutput] | None = None
    next_unique_id = 0
    last_obj_id_unique_id_assignments: dict[str, dict] | None = None

    last_global_frame = -1
    for frame_batch, frame_numbers, scene_done in batch_frame_loader:
        new_frame_numbers_set = set(frame_numbers)

        session_id = harness.start_session(frame_batch, frame_numbers=frame_numbers, offload_state_to_cpu=None, store_session=False)
        
        outs: dict[str, SAM3VideoOutput] = {}

        try:
            n = len(frame_batch)
            if prompt_frame_spacing == "space_around":
                prompt_indices = [int(n * (i + 1) / (num_prompt_applications + 1)) for i in range(num_prompt_applications)]
            elif prompt_frame_spacing == "space_between":
                if num_prompt_applications == 1:
                    prompt_indices = [0]
                else:
                    prompt_indices = [int((n - 1) * i / (num_prompt_applications - 1)) for i in range(num_prompt_applications)]
            else:
                raise ValueError(f"Unknown prompt_frame_spacing: {prompt_frame_spacing}")

            for prompt in prompts:
                has_objects = False
                for idx in prompt_indices:
                    resp = harness.add_prompt(prompt, frame_index=idx, session_id=session_id)
                    if resp is not None and "outputs" in resp:
                        if len(resp["outputs"].get("out_obj_ids", [])) > 0:
                            has_objects = True
                
                if has_objects:
                    try:
                        out = harness.propagate_session(session_id=session_id)
                    except RuntimeError as e:
                        # This is ugly, but we want to check specifically for a string in the error
                        # "No points are provided; please add points first"
                        # I don't know what causes this, but we can ignore it
                        if "add points" in str(e):
                            traceback.print_exc()
                            print(f"WARNING: {e}")
                            out = None
                        else:
                            raise e
                else:
                    out = None
                
                if out is not None:
                    outs[prompt] = out

                harness.reset_session(session_id=session_id)
        finally:
            harness.close_session(session_id=session_id)
            
        frame0 = frame_batch[0]
        if isinstance(frame0, Image.Image):
            width, height = frame0.size
        else:
            height, width = frame0.shape[:2]
            
        for prompt in prompts:
            if prompt not in outs:
                outs[prompt] = SAM3VideoOutput(
                    segmentation=np.full((len(frame_batch), height, width), SAM3VideoOutput.background_index, dtype=SEG_ID_TYPE),
                    confidences=[dict() for _ in range(len(frame_batch))],
                    video_frame_indices=list(frame_numbers),
                )

        if last_frame_numbers_set:
            overlap = last_frame_numbers_set.intersection(new_frame_numbers_set)
            assert last_outs is not None
            cur_obj_id_to_prev_obj_id: dict[str, dict] = {}
            for prompt in prompts:
                equalities = compute_overlap_ids(list(overlap), last_outs[prompt], outs[prompt], iou_thresh=iou_thresh)
                cur_obj_id_to_prev_obj_id[prompt] = {cur_obj_id: last_obj_id for (last_obj_id, cur_obj_id) in equalities}
        else:
            cur_obj_id_to_prev_obj_id = {prompt: {} for prompt in prompts}

        # Now we need to assign the object ids to unique ids
        obj_id_to_unique_id_assignments: dict[str, dict] = {}
        lookup_tables: dict[str, np.ndarray] = {}
        for prompt in prompts:
            out = outs[prompt]
            seg = out.segmentation
            obj_ids = [int(obj_id) for obj_id in np.unique(seg) if obj_id != out.background_index]
            assignments = {}
            for cur_obj_id in obj_ids:
                # First, we see if there is a match to an old segmentation
                last_obj_id = cur_obj_id_to_prev_obj_id[prompt].get(cur_obj_id, None)
                if last_obj_id is None:
                    unique_id = next_unique_id
                    next_unique_id += 1
                else:
                    assert last_obj_id_unique_id_assignments is not None
                    unique_id = last_obj_id_unique_id_assignments[prompt][last_obj_id]
                assignments[cur_obj_id] = unique_id

            max_obj_id = int(np.max(seg)) if len(obj_ids) > 0 else out.background_index
            
            # Create a lookup table initialized to map to itself by default
            lookup_table_size = max(max_obj_id, out.background_index) + 1
            lookup_table = np.arange(lookup_table_size, dtype=np.int32)
            
            # Populate the lookup table with the dictionary assignments
            for old_id, new_id in assignments.items():
                lookup_table[old_id] = new_id
                
            obj_id_to_unique_id_assignments[prompt] = assignments
            lookup_tables[prompt] = lookup_table

        for i in range(len(frame_batch)):
            global_frame_num = frame_numbers[i]
            if global_frame_num <= last_global_frame:
                continue
            last_global_frame = global_frame_num
            
            frame = frame_batch[i]
            
            frame_segmentations = {}
            for prompt in prompts:
                frame_seg = outs[prompt].segmentation[i]
                frame_segmentations[prompt] = lookup_tables[prompt][frame_seg]

            yield FrameSegmentationInfo(global_frame_num, frame, frame_segmentations, SAM3VideoOutput.background_index)

        if scene_done:
            last_frame_numbers_set = None
            last_outs = None
            last_obj_id_unique_id_assignments = None
        else:
            last_frame_numbers_set = new_frame_numbers_set
            last_outs = outs
            last_obj_id_unique_id_assignments = obj_id_to_unique_id_assignments