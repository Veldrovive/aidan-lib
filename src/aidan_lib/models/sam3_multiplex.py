import os
import tempfile
import itertools
import json
from pathlib import Path
from typing import Iterable, Generator
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
import time

try:
    from sam3.model_builder import build_sam3_multiplex_video_predictor
except ImportError:
    raise ImportError("Meta SAM3 not installed. Make sure you installed it via 'pip install git+https://github.com/facebookresearch/sam3.git'")

SEG_ID_TYPE = torch.int8
MAX_SEG_ID = torch.iinfo(SEG_ID_TYPE).max

TextPrompt = str

VideoSegmentation = Int[torch.Tensor, "n_frames n_segments height width"]
VideoConfidences = list[dict[int, float]]  # Length equals n_frames

@dataclass
class SAM3VideoOutput:
    background_index = MAX_SEG_ID
    segmentation: VideoSegmentation
    confidences: VideoConfidences
    prompt_to_obj_ids: dict[str, list[int]]
    video_frame_indices: list[int]


FrameSegmentation = Int[torch.Tensor, "n_segments height width"]
FrameConfidences = dict[int, float]

@dataclass
class SAM3FrameOutput:
    background_index = MAX_SEG_ID
    segmentation: FrameSegmentation
    confidences: FrameConfidences
    prompt_to_obj_ids: dict[str, list[int]]
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
    ):
        self.device = torch.device(device)
        self.dtype = dtype
        self.inference_device = torch.device(inference_device) if inference_device is not None else self.device
        self.processing_device = torch.device(processing_device)
        self.video_storage_device = torch.device(video_storage_device)
        
        # Initialize Meta's multiplex video predictor
        print(f"Constructing multiplex predictor")
        print(f"WARNING: DISABLING FA3")
        self.predictor = build_sam3_multiplex_video_predictor(use_fa3=False, compile=False, warm_up=False)
        print(f"Created multiplex predictor")

    def _parse_meta_outputs(self, outputs) -> tuple[dict[int, np.ndarray], dict[int, float]]:
        """Extracts masks and scores safely from Meta's output dictionary format."""
        masks = {}
        scores = {}
        
        if isinstance(outputs, dict):
            for obj_id, data in outputs.items():
                if isinstance(data, dict):
                    masks[int(obj_id)] = data.get("mask", data.get("masks", None))
                    scores[int(obj_id)] = float(data.get("score", data.get("scores", 1.0)))
                else:
                    masks[int(obj_id)] = data
                    scores[int(obj_id)] = 1.0
        elif isinstance(outputs, list):
            for i, data in enumerate(outputs):
                obj_id = data.get("obj_id", i)
                masks[int(obj_id)] = data.get("mask", data.get("masks", None))
                scores[int(obj_id)] = float(data.get("score", data.get("scores", 1.0)))

        return masks, scores

    def process_frame_outputs(
        self, 
        masks_dict: dict[int, np.ndarray], 
        scores_dict: dict[int, float], 
        prompt_to_obj_ids: dict[str, list[int]], 
        video_frame_index: int, 
        height: int, 
        width: int
    ) -> SAM3FrameOutput:
        
        segmentation = torch.full(
            (height, width), 
            fill_value=SAM3FrameOutput.background_index, 
            dtype=SEG_ID_TYPE, 
            device='cpu'
        )
        confidences: dict[int, float] = {}

        if not masks_dict:
            return SAM3FrameOutput(
                segmentation=segmentation,
                confidences=confidences,
                prompt_to_obj_ids=prompt_to_obj_ids,
                video_frame_index=video_frame_index
            )

        obj_ids = list(masks_dict.keys())
        mask_arrays = []
        
        for obj_id in obj_ids:
            mask = masks_dict[obj_id]
            if isinstance(mask, torch.Tensor):
                mask = mask.detach().cpu().numpy()
            
            # Map logits to bool
            if mask.dtype in (np.float32, np.float64):
                mask = mask > 0.0
            else:
                mask = mask > 0.5
            mask_arrays.append(mask)

        # Move to inference device to apply Kornia morphology
        frame_masks = torch.tensor(np.stack(mask_arrays), dtype=torch.float32, device=self.inference_device)
        masks_t = frame_masks.unsqueeze(1)

        # 1. Morphological Opening & Closing
        kernel_size = 5
        kernel = torch.ones((kernel_size, kernel_size), device=self.inference_device, dtype=torch.float32)
        
        opened = morph.opening(masks_t, kernel)
        closed = morph.closing(opened, kernel)
        closed = closed.cpu().long()

        for j, obj_id in enumerate(obj_ids):
            score = scores_dict.get(obj_id, 1.0)
            if obj_id >= SAM3FrameOutput.background_index:
                raise ValueError(f"Too many segmentations in this scene (max allowed: {SAM3FrameOutput.background_index - 1})")

            # Extract the 2D morphological mask
            mask_indices = closed[j, 0] > 0
            segmentation[mask_indices] = obj_id
            confidences[obj_id] = score

        return SAM3FrameOutput(
            segmentation=segmentation,
            confidences=confidences,
            prompt_to_obj_ids=prompt_to_obj_ids,
            video_frame_index=video_frame_index
        )

    @torch.no_grad()
    def process_video_stream_from_path(
        self,
        video_path: Path | str,
        text_prompts: TextPrompt | list[TextPrompt],
        frame_skip: int | None = None
    ) -> Generator[SAM3FrameOutput, None, None]:
        
        video_path_str = str(video_path)
        
        # Initialize inference session on video file/directory
        print(f"Starting session")
        start_time = time.perf_counter()
        response = self.predictor.handle_request(
            request=dict(
                type="start_session",
                resource_path=video_path_str,
                offload_state_to_cpu=None
            )
        )
        end_time = time.perf_counter()
        session_id = response["session_id"]
        print(f"Started session {session_id}. Took {end_time - start_time}")
        
        prompts = [text_prompts] if isinstance(text_prompts, str) else text_prompts
        prompt_to_obj_ids = {}
        active_obj_ids = set()
        
        # Add prompts to frame 0 natively
        print(f"Adding prompts to frame 0")
        for p in prompts:
            resp = self.predictor.handle_request(
                request=dict(
                    type="add_prompt",
                    session_id=session_id,
                    frame_index=0,
                    text=p,
                )
            )
            masks, _ = self._parse_meta_outputs(resp.get("outputs", {}))
            
            # The newly appearing object IDs belong to the text prompt we just injected
            new_ids = set(masks.keys()) - active_obj_ids
            prompt_to_obj_ids[p] = sorted(list(new_ids))
            active_obj_ids.update(new_ids)
        print(f"Added prompts to frame 0")

        # Get spatial dimensions for initializing tensor buffers
        if Path(video_path_str).is_dir():
            sample_frame_path = sorted(Path(video_path_str).glob("*.jpg"))[0]
            sample_frame = cv2.imread(str(sample_frame_path))
            assert sample_frame is not None, f"Frame {sample_frame_path} not found"
        else:
            sample_frame = iio.imread(video_path_str, plugin="pyav", index=0)
        height, width = sample_frame.shape[:2]

        # Propagate through the remaining video
        print(f"Starting streaming")
        for response in self.predictor.handle_stream_request(
            request=dict(
                type="propagate_in_video",
                session_id=session_id,
            )
        ):
            print(f"Got stream response", end="")
            frame_idx = response["frame_index"]
            print(f" for frame {frame_idx}")
            
            if frame_skip is not None and frame_idx % frame_skip != 0:
                print(f"Skipping frame {frame_skip}")
                continue

            print(f"Parsing outputs")
            masks, scores = self._parse_meta_outputs(response.get("outputs", {}))
            print(f"Parsed outputs")
            
            yield self.process_frame_outputs(
                masks_dict=masks,
                scores_dict=scores,
                prompt_to_obj_ids=prompt_to_obj_ids,
                video_frame_index=frame_idx,
                height=height,
                width=width
            )

        # Clean up backend session allocations
        self.predictor.handle_request(
            request=dict(
                type="close_session",
                session_id=session_id,
            )
        )

    @torch.no_grad()
    def process_video_from_path(self, video_path: Path | str, prompt: TextPrompt | list[TextPrompt], show_progress: bool = False) -> SAM3VideoOutput:
        stream = self.process_video_stream_from_path(video_path, prompt)
        if show_progress:
            stream = tqdm(stream, desc="Processing Video (SAM 3.1 Object Multiplex)")

        segmentations = []
        confidences = []
        prompt_to_obj_ids = {}
        frame_indices = []

        for frame_output in stream:
            segmentations.append(frame_output.segmentation)
            confidences.append(frame_output.confidences)
            prompt_to_obj_ids = frame_output.prompt_to_obj_ids
            frame_indices.append(frame_output.video_frame_index)

        return SAM3VideoOutput(
            segmentation=torch.stack(segmentations) if segmentations else torch.empty(0),
            confidences=confidences,
            prompt_to_obj_ids=prompt_to_obj_ids,
            video_frame_indices=frame_indices,
        )

    def inject_mask_prompt(self, session_id: str, frame_idx: int, obj_id: int, mask: torch.Tensor):
        """
        Hacky way to add a mask prompt to Sam3MultiplexVideoPredictor.
        
        Args:
            session_id (str): The active session ID
            frame_idx (int): The frame index to apply the mask to
            obj_id (int): The object ID to assign this mask to
            mask (torch.Tensor): A 2D boolean or float tensor of shape [H, W]
        """
        # 1. Retrieve the high-level inference state for the session
        session = self.predictor._get_session(session_id)
        inference_state = session["state"]
        model = self.predictor.model  # Sam3MultiplexTracking
        assert model is not None, "Model not found"
        
        # Ensure mask is a tensor and on the correct device
        if not isinstance(mask, torch.Tensor):
            mask = torch.tensor(mask)
        mask = mask.to(inference_state["device"]).float()
        
        # # 2. In multiplex mode, objects are bucketed. 
        # # Get or create the specific SAM 2 state bucket for this object.
        # sam2_state, _ = model._get_or_create_sam2_state_for_obj(
        #     inference_state, frame_idx, obj_id
        # )
        
        # 3. Call the underlying SAM 2 tracker's add_new_mask method directly
        # This will overwrite any existing points/masks for this object on this frame
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            frame_idx, obj_ids, low_res_masks, video_res_masks = model.tracker.add_new_mask(
                inference_state=inference_state,
                frame_idx=frame_idx,
                obj_id=obj_id,
                mask=mask
            )
            
        # 4. Optional: If you are doing this mid-tracking, you may want to update the multiplex metadata.
        # We update the last use time to prevent session expiration.
        self.predictor._extend_expiration_time(session)
        
        return video_res_masks

if __name__ == "__main__":
    def test():
        from aidan_lib.definitions import DATA_DIR
        from aidan_lib.models.sam3_hdf5_utils import SAM3HDF5Manager
        import time
        from tqdm import tqdm

        print("Starting SAM3 test...")
        MAX_TEST_FRAMES = 100
        
        harness = SAM3Harness(
            inference_device="cuda",
            processing_device="cpu",
            video_storage_device="cpu",
        )
        
        test_video_path = DATA_DIR / "tip_to_tip_short.mp4"
        test_segmentations_path = DATA_DIR / "tip_to_tip_short_segmentations_multiplex.hdf5"
        # print(f"Loading frames from {test_video_path}")
        # start_time = time.perf_counter()
        # video_frames = iio.imread(test_video_path, plugin="pyav")
        # end_time = time.perf_counter()
        # print(f"Time to complete video load: {end_time - start_time}")
        # print(f"Loaded {video_frames.shape}")
        # video_frames = video_frames[:MAX_TEST_FRAMES]
        # print(f"Processing {video_frames.shape}")
            
        # progress = tqdm()
        # for i, frame_output in enumerate(harness.process_video_stream_from_path(test_video_path, "Person")):
        #     progress.update(1)
        #     progress.set_description(f"Frame {i}: Num segments: {len(frame_output.confidences)}")

        manager = SAM3HDF5Manager(test_segmentations_path)
        frame_data_stream = harness.process_video_stream_from_path(test_video_path, "Person")
        manager.save_stream(frame_data_stream, progress_bar=True)
        
    test()