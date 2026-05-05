from transformers.generation.continuous_batching import input_outputs
import os
import tempfile
import itertools
import json
from pathlib import Path
from typing import Iterable, Generator, NamedTuple
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

try:
    from sam3.model_builder import build_sam3_multiplex_video_predictor
except ImportError:
    raise ImportError("Meta SAM3 not installed. Make sure you installed it via 'pip install git+https://github.com/facebookresearch/sam3.git'")

SEG_ID_TYPE = np.int8
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

class SAM3Harness:
    def __init__(
        self, 
        checkpoint: str = "facebook/sam3", # Kept for API compatibility 
        device: str | torch.device = "cuda",
        dtype: torch.dtype = torch.bfloat16,
        inference_device: str | torch.device | None = None,
        processing_device: str | torch.device = "cpu",
        video_storage_device: str | torch.device = "cpu",
        compile: bool = True,
        warm_up: bool = True,
    ):
        self.device = torch.device(device)
        self.dtype = dtype
        self.inference_device = torch.device(inference_device) if inference_device is not None else self.device
        self.processing_device = torch.device(processing_device)
        self.video_storage_device = torch.device(video_storage_device)
        
        # Initialize Meta's multiplex video predictor
        print(f"Constructing multiplex predictor")
        print(f"WARNING: DISABLING FA3")
        self.predictor = build_sam3_multiplex_video_predictor(use_fa3=False, compile=compile, warm_up=warm_up)
        print(f"Created multiplex predictor")

        self.main_session_id = None

    def start_session(self, video: Path | list[Image.Image], offload_state_to_cpu: bool | None = None, store_session: bool = True) -> str:
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

    class SAMOutType(NamedTuple):
        masks: np.ndarray
        scores: list[float]
        out_obj_ids: list[int]
        bboxs: np.ndarray

    def propogate_session(self, frame_number_map: list[int] | None = None, session_id: str | None = None):
        if session_id is None:
            session_id = self.main_session_id
            if session_id is None:
                raise ValueError("No session_id provided and no active main session.")
        
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

        segmentation = np.zeros((n_frames, height, width), dtype=SEG_ID_TYPE)
        confidences = [dict() for _ in range(n_frames)]
        video_frame_indices = []

        output_frame_indices = sorted(outputs.keys())

        for output_frame_num in output_frame_indices:
            frame_out = outputs[output_frame_num]
            video_frame_indices.append(output_frame_num)
            
            for i, obj_id in enumerate(frame_out.out_obj_ids):
                mask = frame_out.masks[i]
                conf = frame_out.scores[i]

                confidences[output_frame_num][obj_id] = conf
                
                mask_locations = np.where(mask)
                segmentation[output_frame_num, mask_locations[0], mask_locations[1]] = obj_id

        video_output = SAM3VideoOutput(
            segmentation=segmentation,
            confidences=confidences,
            video_frame_indices=video_frame_indices
        )

        return video_output
