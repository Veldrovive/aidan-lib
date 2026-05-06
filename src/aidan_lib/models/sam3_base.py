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
from abc import ABC, abstractmethod
from multiprocessing import shared_memory, resource_tracker
from multiprocessing.connection import Listener, Client
import contextlib

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
        prompt: TextPrompt,
        video: Path | list[Image.Image],
        prompt_frame: int = 0,
        frame_numbers: list[int] | None = None,
        offload_state_to_cpu: bool | None = None
    ) -> SAM3VideoOutput:
        session_id = self.start_session(video, frame_numbers, offload_state_to_cpu, store_session=False)
        try:
            self.add_prompt(prompt, prompt_frame, session_id)
            output = self.propagate_session(session_id)
        finally:
            self.close_session(session_id)

        return output

class SAM3Harness(BaseSAM3Harness):
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

        self.main_session_id: str | None = None

        self.session_metadata: dict[str, SAM3Harness.SessionMetadata] = {}

    class SessionMetadata(NamedTuple):
        session_id: str
        offload_state_to_cpu: bool | None
        frame_numbers: list[int] | None

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
            frame_numbers=frame_numbers
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
        
        frame_number_map = self.session_metadata[session_id].frame_numbers

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
            video_frame_indices=video_frame_indices
        )

        return video_output

DEFAULT_SAM3_FRAME_SHARED_MEMORY_NAME = "SAM3_HARNESS_FRAMES"
DEFAULT_SAM3_SEGMENTATION_SHARED_MEMORY_NAME = "SAM3_HARNESS_SEGMENTATIONS"
DEFAULT_SAM3_SERVER_ADDRESS = ("localhost", 26000)
def _get_buffer_specs(num_frames: int, width: int, height: int, num_channels: int, frame_dtype: np.dtype):
    if num_channels == 1:
        shape = (num_frames, height, width)
    else:
        shape = (num_frames, height, width, num_channels)
    dummy_batch = np.zeros(shape, dtype=frame_dtype)
    return dummy_batch.nbytes, dummy_batch.shape

class SAM3HarnessServer:
    def __init__(
        self,
        max_num_frames: int,
        max_frame_width: int,
        max_frame_height: int,
        frame_dtype: np.dtype,
        segmentation_dtype: np.dtype = SEG_ID_TYPE,
        address: tuple[str, int] | None = None,
        shared_frame_memory_name: str | None = None,
        shared_segmentation_memory_name: str | None = None,
        segmenter: BaseSAM3Harness | None = None,
        segmenter_kwargs: dict | None = None,
    ):
        if address is None:
            self.address = DEFAULT_SAM3_SERVER_ADDRESS
        else:
            self.address = address
        self.max_num_frames = max_num_frames
        self.max_frame_width = max_frame_width
        self.max_frame_height = max_frame_height
        self.frame_dtype = frame_dtype
        self.segmentation_dtype = segmentation_dtype

        self.frame_buffer_bytes, self.frame_buffer_shape = _get_buffer_specs(
            self.max_num_frames, self.max_frame_width, self.max_frame_height, 3, self.frame_dtype
        )
        self.segmentation_buffer_bytes, self.segmentation_buffer_shape = _get_buffer_specs(
            self.max_num_frames, self.max_frame_width, self.max_frame_height, 1, self.segmentation_dtype
        )

        if shared_frame_memory_name is None:
            self.frame_shm_name = DEFAULT_SAM3_FRAME_SHARED_MEMORY_NAME
        else:
            self.frame_shm_name = shared_frame_memory_name

        if shared_segmentation_memory_name is None:
            self.segmentation_shm_name = DEFAULT_SAM3_SEGMENTATION_SHARED_MEMORY_NAME
        else:
            self.segmentation_shm_name = shared_segmentation_memory_name

        try:
            self.frame_shm = shared_memory.SharedMemory(
                create=True,
                size=self.frame_buffer_bytes,
                name=self.frame_shm_name
            )
            self.frame_array = np.ndarray(self.frame_buffer_shape, dtype=self.frame_dtype, buffer=self.frame_shm.buf)
            print(f"[SAM3 SERVER] Shared memory created: {self.frame_shm_name} (size: {self.frame_buffer_bytes} bytes)")
        except FileExistsError:
            raise RuntimeError(f"Shared memory {self.frame_shm_name} already exists. Please close the server first or use a different name.")
        except Exception as e:
            raise RuntimeError(f"Failed to create shared memory: {e}")

        try:
            self.segmentation_shm = shared_memory.SharedMemory(
                create=True,
                size=self.segmentation_buffer_bytes,
                name=self.segmentation_shm_name
            )
            self.segmentation_array = np.ndarray(self.segmentation_buffer_shape, dtype=self.segmentation_dtype, buffer=self.segmentation_shm.buf)
            print(f"[SAM3 SERVER] Shared memory created: {self.segmentation_shm_name} (size: {self.segmentation_buffer_bytes} bytes)")
        except FileExistsError:
            raise RuntimeError(f"Shared memory {self.segmentation_shm_name} already exists. Please close the server first or use a different name.")
        except Exception as e:
            raise RuntimeError(f"Failed to create shared memory: {e}")

        if segmenter is None:
            if segmenter_kwargs is None:
                segmenter_kwargs = {}
            print(f"[SAM3 SERVER] Initializing SAM3Harness with keyword arguments: {segmenter_kwargs}")
            segmenter = SAM3Harness(
                **segmenter_kwargs,
            )
            print(f"[SAM3 SERVER] Created multiplex predictor")

        self.segmenter = segmenter

    def _start_session(self, num_frames: int, height: int, width: int, frame_numbers: list[int], offload_state_to_cpu: bool | None = None, store_session: bool = True):
        # We need to load the video in from the shared memory
        assert num_frames <= self.max_num_frames, f"Number of frames {num_frames} exceeds maximum {self.max_num_frames}."
        assert height <= self.max_frame_height, f"Height {height} exceeds maximum {self.max_frame_height}."
        assert width <= self.max_frame_width, f"Width {width} exceeds maximum {self.max_frame_width}."
        
        frame_chunk = self.frame_array[:num_frames, :height, :width]
        # We need to convert to PIL images for the segmenter
        frame_chunk_pil = [Image.fromarray(frame) for frame in frame_chunk]

        session_id = self.segmenter.start_session(
            video=frame_chunk_pil,
            frame_numbers = frame_numbers,
            offload_state_to_cpu = offload_state_to_cpu,
            store_session = store_session
        )
        return {"session_id": session_id}

    def _reset_session(self, session_id: str | None = None ):
        self.segmenter.reset_session(session_id)
        return {}

    def _propagate_session(self, session_id: str | None = None):
        out = self.segmenter.propagate_session(session_id)
        
        segmentation = out.segmentation  # [num_frames, height, width]
        seg_num_frames, seg_height, seg_width = segmentation.shape
        # We need to insert these segmentations into the top left corner of the segmentations shared memory
        self.segmentation_array[:seg_num_frames, :seg_height, :seg_width] = segmentation

        return {
            "num_frames": seg_num_frames,
            "height": seg_height,
            "width": seg_width,
            "background_index": out.background_index,
            "confidences": out.confidences,
            "video_frame_indices": out.video_frame_indices,
        }

    def _add_prompt(self, prompt: TextPrompt, frame_index: int = 0, session_id: str | None = None):
        self.segmenter.add_prompt(prompt, frame_index, session_id)
        return {}

    def _close_session(self, session_id: str | None = None):
        self.segmenter.close_session(session_id)
        return {}

    def _handle_request(self, request: dict):
        action = request.get("action")
        params = request.get("params", {})
        print(f"[SAM3 SERVER] Received request: {action} with parameters: {params}")

        # Simple dispatcher mapping
        dispatch = {
            "start_session": self._start_session,
            "reset_session": self._reset_session,
            "add_prompt": self._add_prompt,
            "propagate_session": self._propagate_session,
            "close_session": self._close_session,
        }

        if action in dispatch:
            result = dispatch[action](**params)
            return result
        else:
            raise ValueError(f"Invalid request: {request}")

        
    def run(self):
        try:
            with Listener(self.address) as listener:
                print(f"[SAM3 SERVER] Listening on {self.address}")
                while True:
                    try:
                        with listener.accept() as conn:
                            # We need a lock so that the client knows when to overwrite memory so we avoid race conditions
                            try:
                                initial_request = conn.recv()

                                if initial_request["action"] == "acquire_lock":
                                    conn.send({"status": "success", "message": "Acquired lock", "data": None})
                                    process_request = conn.recv()

                                    res = self._handle_request(process_request)
                                    conn.send({"status": "success", "message": "Processed request", "data": res})
                                else:
                                    conn.send({"status": "error", "message": "Invalid initial request", "data": None})
                            except Exception as e:
                                conn.send({"status": "error", "message": f"Error: {e}", "data": None})
                    except Exception as e:
                        print(f"[SAM3 SERVER] Error in connection: {e}")
        except KeyboardInterrupt:
            print("[SAM3 SERVER] Interrupted. Shutting down...")
        except Exception as e:
            print(f"[SAM3 SERVER] Error: {e}")
        finally:
            self.cleanup()

    def cleanup(self):
        print("[SAM3 SERVER] Cleaning up shared memory...")
        for shm in [self.frame_shm, self.segmentation_shm]:
            try:
                shm.close()
                shm.unlink()
            except Exception as e:
                print(f"Warning: Failed to clean up {shm.name}: {e}")

class SAM3HarnessClient(BaseSAM3Harness):
    def __init__(
        self,
        max_num_frames: int,
        max_frame_width: int,
        max_frame_height: int,
        frame_dtype: np.dtype,
        segmentation_dtype: np.dtype = SEG_ID_TYPE,
        address: tuple[str, int] | None = None,
        shared_frame_memory_name: str | None = None,
        shared_segmentation_memory_name: str | None = None,
    ):
        self.address = address if address is not None else DEFAULT_SAM3_SERVER_ADDRESS
        self.max_num_frames = max_num_frames
        self.max_frame_width = max_frame_width
        self.max_frame_height = max_frame_height
        self.frame_dtype = frame_dtype
        self.segmentation_dtype = segmentation_dtype

        self.frame_shm_name = shared_frame_memory_name or DEFAULT_SAM3_FRAME_SHARED_MEMORY_NAME
        self.segmentation_shm_name = shared_segmentation_memory_name or DEFAULT_SAM3_SEGMENTATION_SHARED_MEMORY_NAME

        # Connect to existing shared memory initialized by the server
        try:
            self.frame_shm = shared_memory.SharedMemory(name=self.frame_shm_name)
            self.segmentation_shm = shared_memory.SharedMemory(name=self.segmentation_shm_name)

            resource_tracker.unregister(self.frame_shm._name, 'shared_memory')
            resource_tracker.unregister(self.segmentation_shm._name, 'shared_memory')
        except FileNotFoundError:
            raise RuntimeError("Shared memory not found. Ensure the SAM3HarnessServer is running before initializing the client.")

        _, self.frame_buffer_shape = _get_buffer_specs(
            self.max_num_frames, self.max_frame_width, self.max_frame_height, 3, self.frame_dtype
        )
        _, self.segmentation_buffer_shape = _get_buffer_specs(
            self.max_num_frames, self.max_frame_width, self.max_frame_height, 1, self.segmentation_dtype
        )

        self.frame_array = np.ndarray(self.frame_buffer_shape, dtype=self.frame_dtype, buffer=self.frame_shm.buf)
        self.segmentation_array = np.ndarray(self.segmentation_buffer_shape, dtype=self.segmentation_dtype, buffer=self.segmentation_shm.buf)

        self.main_session_id: str | None = None

    @contextlib.contextmanager
    def _request_context(self):
        """
        Manages connection and handles the 2-step lock handshake protocol 
        to ensure safe synchronous access to the shared memory blocks.
        """
        with Client(self.address) as conn:
            # 1. Acquire lock
            conn.send({"action": "acquire_lock"})
            resp = conn.recv()
            if resp.get("status") != "success":
                raise RuntimeError(f"Failed to acquire lock: {resp.get('message')}")
            
            # 2. Yield control back to caller to read/write memory and dispatch the actual request
            yield conn

    def start_session(
        self, 
        video: Path | list[Image.Image] | np.ndarray, 
        frame_numbers: list[int] | None = None, 
        offload_state_to_cpu: bool | None = None, 
        store_session: bool = True
    ) -> str:
        if self.main_session_id is not None and store_session:
            raise ValueError("Session already started. Please call close_session first.")

        # Prepare video frames based on type
        if isinstance(video, Path):
            if video.is_dir():
                valid_exts = {".jpg", ".jpeg", ".png", ".bmp"}
                img_paths = sorted([p for p in video.iterdir() if p.suffix.lower() in valid_exts])
                video_frames = [np.array(Image.open(p).convert("RGB")) for p in img_paths]
            else:
                frames_arr = iio.imread(video)
                video_frames = [frames_arr[i] for i in range(frames_arr.shape[0])]
        elif isinstance(video, np.ndarray):
            video_frames = [video[i] for i in range(video.shape[0])]
        else:
            video_frames = [np.array(img.convert("RGB")) for img in video]

        num_frames = len(video_frames)
        if num_frames == 0:
            raise ValueError("Video contains no frames.")

        height, width = video_frames[0].shape[:2]

        if num_frames > self.max_num_frames:
            raise ValueError(f"Number of frames {num_frames} exceeds maximum {self.max_num_frames}.")
        if height > self.max_frame_height:
            raise ValueError(f"Height {height} exceeds maximum {self.max_frame_height}.")
        if width > self.max_frame_width:
            raise ValueError(f"Width {width} exceeds maximum {self.max_frame_width}.")

        # Fill shared memory inside the lock context
        with self._request_context() as conn:
            # Overwrite the shared memory array with new frames
            for i, frame in enumerate(video_frames):
                self.frame_array[i, :height, :width] = frame

            # Send processing instruction to the server
            request = {
                "action": "start_session",
                "params": {
                    "num_frames": num_frames,
                    "height": height,
                    "width": width,
                    "frame_numbers": frame_numbers,
                    "offload_state_to_cpu": offload_state_to_cpu,
                    "store_session": store_session
                }
            }
            conn.send(request)
            resp = conn.recv()

            if resp.get("status") != "success":
                raise RuntimeError(f"Server error: {resp.get('message')}")

            session_id = resp["data"]["session_id"]
            if store_session:
                self.main_session_id = session_id

            return session_id

    def reset_session(self, session_id: str | None = None):
        session_id = session_id or self.main_session_id
        if session_id is None:
            raise ValueError("No session_id provided and no active main session.")

        with self._request_context() as conn:
            conn.send({
                "action": "reset_session",
                "params": {"session_id": session_id}
            })
            resp = conn.recv()
            if resp.get("status") != "success":
                raise RuntimeError(f"Server error: {resp.get('message')}")

    def add_prompt(self, prompt: TextPrompt, frame_index: int = 0, session_id: str | None = None):
        session_id = session_id or self.main_session_id
        if session_id is None:
            raise ValueError("No session_id provided and no active main session.")

        with self._request_context() as conn:
            conn.send({
                "action": "add_prompt",
                "params": {
                    "prompt": prompt, 
                    "frame_index": frame_index, 
                    "session_id": session_id
                }
            })
            resp = conn.recv()
            if resp.get("status") != "success":
                raise RuntimeError(f"Server error: {resp.get('message')}")
            return resp.get("data", {})

    def propagate_session(self, session_id: str | None = None) -> SAM3VideoOutput:
        session_id = session_id or self.main_session_id
        if session_id is None:
            raise ValueError("No session_id provided and no active main session.")

        with self._request_context() as conn:
            conn.send({
                "action": "propagate_session",
                "params": {"session_id": session_id}
            })
            resp = conn.recv()
            
            if resp.get("status") != "success":
                raise RuntimeError(f"Server error: {resp.get('message')}")

            data = resp["data"]
            num_frames = data["num_frames"]
            height = data["height"]
            width = data["width"]

            # Extract segmentation from shared memory while the connection (and lock) is still active.
            # Using `.copy()` is crucial here so subsequent calls don't overwrite the output array
            segmentation_copy = self.segmentation_array[:num_frames, :height, :width].copy()

        return SAM3VideoOutput(
            segmentation=segmentation_copy,
            confidences=data["confidences"],
            video_frame_indices=data["video_frame_indices"]
        )

    def close_session(self, session_id: str | None = None):
        session_id = session_id or self.main_session_id
        if session_id is None:
            raise ValueError("No session_id provided and no active main session.")

        with self._request_context() as conn:
            conn.send({
                "action": "close_session",
                "params": {"session_id": session_id}
            })
            resp = conn.recv()
            if resp.get("status") != "success":
                raise RuntimeError(f"Server error: {resp.get('message')}")

        if session_id == self.main_session_id:
            self.main_session_id = None