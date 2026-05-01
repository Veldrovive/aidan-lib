from typing import Iterable
try:
    from transformers import Sam3VideoModel, Sam3VideoProcessor
except ImportError:
    raise ImportError("Transformers not installed. Make sure you installed aidan-lib[hf]")
import torch
from pathlib import Path
from typing import Generator
import imageio.v3 as iio
from jaxtyping import Int, Float, Bool
import kornia.morphology as morph
import kornia.contrib as contrib
import numpy as np
from dataclasses import dataclass
import itertools
import h5py
import json
from tqdm import tqdm

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
    model: Sam3VideoModel
    processor: Sam3VideoProcessor
    
    device: torch.device
    dtype: torch.dtype
    inference_device: torch.device
    processing_device: torch.device
    video_storage_device: torch.device

    def __init__(
        self, 
        checkpoint: str = "facebook/sam3", 
        device: str | torch.device = "cuda",
        dtype: torch.dtype = torch.bfloat16,
        inference_device: str | torch.device | None = None,  # Defaults to save as self.device
        processing_device: str | torch.device = "cpu",
        video_storage_device: str | torch.device = "cpu",
    ):
        self.checkpoint = checkpoint
        self.device = torch.device(device)
        self.dtype = dtype
        self.inference_device = torch.device(inference_device) if inference_device is not None else self.device
        self.processing_device = torch.device(processing_device)
        self.video_storage_device = torch.device(video_storage_device)
        
        self.model = Sam3VideoModel.from_pretrained(checkpoint).to(self.device, dtype=self.dtype)
        self.model.eval()
        self.processor = Sam3VideoProcessor.from_pretrained(checkpoint)

    def process_frame_outputs(self, processed_outputs, video_frame_index: int) -> SAM3FrameOutput:
        prompt_to_obj_ids: dict[TextPrompt, list[int]] = processed_outputs["prompt_to_obj_ids"]
        frame_obj_ids: Int[torch.Tensor, "n_segments"] = processed_outputs["object_ids"]
        frame_scores: Float[torch.Tensor, "n_segments"] = processed_outputs["scores"]
        frame_masks: Bool[torch.Tensor, "n_segments height width"] = processed_outputs["masks"]

        n_segments, height, width = frame_masks.shape

        # Initialize the combined frame segmentation map
        # Note: We place this on the CPU as the process_video method expects to append 
        # it to a CPU-bound tensor (`segmentations`) initialized via torch.full
        segmentation = torch.full(
            (height, width), 
            fill_value=SAM3FrameOutput.background_index, 
            dtype=SEG_ID_TYPE, 
            device='cpu'
        )
        confidences: dict[int, float] = {}

        if n_segments == 0:
            return SAM3FrameOutput(
                segmentation=segmentation,
                confidences=confidences,
                prompt_to_obj_ids=prompt_to_obj_ids,
                video_frame_index=video_frame_index
            )

        device = frame_masks.device

        # Prepare for Kornia which expects standard (B, C, H, W) float tensors
        masks_t = frame_masks.to(dtype=torch.float32, device=device).unsqueeze(1)

        # 1. Morphological Opening & Closing
        # Using a batched 5x5 rectangular structural element instead of mapping cv2's ellipse
        kernel_size = 5
        kernel = torch.ones((kernel_size, kernel_size), device=device, dtype=torch.float32)
        
        # Opening: Removes isolated speckles
        opened = morph.opening(masks_t, kernel)
        
        # Closing: Fills in small holes
        closed = morph.closing(opened, kernel)
        closed = closed.cpu().long()

        # 2. Connected Component Filtering
        # Efficient batched identification of contiguous regions
        # labels = contrib.connected_components(closed, num_iterations=2000)

        # Iterate through the segments to isolate the largest blob and assign IDs
        for j in range(n_segments):
            score = float(frame_scores[j].item())
            seg_idx = int(frame_obj_ids[j].item())

            if seg_idx >= SAM3FrameOutput.background_index:
                raise ValueError("Too many segmentations in this scene")

            # # Extract the 2D label map for this segment and cast to int for bincounting
            # label_map = labels[j, 0].to(torch.int64)

            # # Find the largest component excluding the background (label 0)
            # counts = torch.bincount(label_map.flatten())
            
            # if len(counts) > 1:
            #     counts[0] = 0  # Ignore background
            #     largest_label = int(torch.argmax(counts).item())
            #     final_mask = (label_map == largest_label)
            # else:
            #     final_mask = (label_map > 0)

            # Move mask back to CPU and write to final tensor
            # segmentation[final_mask.cpu()] = seg_idx
            segmentation[closed[j, 0]] = seg_idx
            confidences[seg_idx] = score

        return SAM3FrameOutput(
            segmentation=segmentation,
            confidences=confidences,
            prompt_to_obj_ids=prompt_to_obj_ids,
            video_frame_index=video_frame_index
        )

    @torch.no_grad()
    def process_video(self, video_frames: Float[np.ndarray, "frames height width channels"], prompt: TextPrompt | list[TextPrompt], show_progress: bool = False) -> SAM3VideoOutput:
        num_frames, height, width, _ = video_frames.shape

        if show_progress:
            print(f"Initializing SAM3 video session for {num_frames} frames...")
        inference_session = self.processor.init_video_session(
            video=video_frames,
            inference_device=self.inference_device,
            processing_device=self.processing_device,
            video_storage_device=self.video_storage_device,
            dtype=self.dtype,
        )

        if show_progress:
            print(f"Adding text prompt: {prompt}")
        inference_session = self.processor.add_text_prompt(
            inference_session=inference_session,
            text=prompt,
        )

        # MAX_SEG_ID corresponds to the background
        segmentations = torch.full((num_frames, height, width), fill_value=MAX_SEG_ID, dtype=SEG_ID_TYPE)
        confidences: list[dict[int, float]] = [{} for _ in range(num_frames)]
        prompt_to_obj_ids_set: dict[TextPrompt, set[int]] = {}

        for model_outputs in self.model.propagate_in_video_iterator(
            inference_session=inference_session,
            show_progress_bar=show_progress,
        ):
            frame_idx: int = model_outputs.frame_idx
            processed_outputs = self.processor.postprocess_outputs(inference_session, model_outputs)
            frame_output = self.process_frame_outputs(processed_outputs, video_frame_index=frame_idx)

            # Repackage the outputs
            segmentations[frame_idx] = frame_output.segmentation
            confidences[frame_idx] = frame_output.confidences
            for prompt, obj_ids in frame_output.prompt_to_obj_ids.items():
                if prompt not in prompt_to_obj_ids_set:
                    prompt_to_obj_ids_set[prompt] = set()
                prompt_to_obj_ids_set[prompt].update(obj_ids)
        
        prompt_to_obj_ids: dict[TextPrompt, list[int]] = {prompt: sorted(list(obj_ids)) for prompt, obj_ids in prompt_to_obj_ids_set.items()}
        return SAM3VideoOutput(
            segmentation=segmentations,
            confidences=confidences,
            prompt_to_obj_ids=prompt_to_obj_ids,
            video_frame_indices=list(range(num_frames)),
        )

    @torch.no_grad()
    def process_video_from_path(self, video_path: Path | str, prompt: TextPrompt | list[TextPrompt], show_progress: bool = False) -> SAM3VideoOutput:
        video_frames = iio.imread(video_path, plugin="pyav")
        return self.process_video(video_frames, prompt, show_progress)


    @torch.no_grad()
    def process_video_stream(
        self,
        video_frames: Iterable[Float[np.ndarray, "height width channels"]],
        text_prompts: TextPrompt | list[TextPrompt],
        frame_indices: Iterable[int] | None = None,
    ) -> Generator[SAM3FrameOutput, None, None]:
        streaming_inference_session = self.processor.init_video_session(
            inference_device=self.inference_device,
            processing_device=self.processing_device,
            video_storage_device=self.video_storage_device,
            dtype=self.dtype,
        )

        streaming_inference_session = self.processor.add_text_prompt(
            inference_session=streaming_inference_session,
            text=text_prompts,
        )

        if frame_indices is None:
            frame_indices = itertools.count(0)

        for frame, frame_idx in zip(video_frames, frame_indices):
            # First, process the frame using the processor
            inputs = self.processor(images=frame, device=self.device, return_tensors="pt")

            # Process frame using streaming inference - pass the processed pixel_values
            model_outputs = self.model(
                inference_session=streaming_inference_session,
                frame=inputs.pixel_values[0],  # Provide processed frame - this enables streaming mode
                reverse=False,
            )

            # Post-process outputs with original_sizes for proper resolution handling
            processed_outputs = self.processor.postprocess_outputs(
                streaming_inference_session,
                model_outputs,
                original_sizes=inputs.original_sizes,  # Required for streaming inference
            )
            frame_output = self.process_frame_outputs(processed_outputs, video_frame_index=frame_idx)
            yield frame_output

    def process_video_stream_from_path(
        self,
        video_path: Path | str,
        text_prompts: TextPrompt | list[TextPrompt],
        frame_skip: int | None = None
    ) -> Generator[SAM3FrameOutput, None, None]:
        video_iterator = iio.imiter(video_path, plugin="pyav")
        if frame_skip is None:
            frame_iterator = video_iterator
            frame_indices = itertools.count(0)
        else:
            frame_iterator = itertools.islice(video_iterator, 0, None, frame_skip)
            frame_indices = itertools.count(0, frame_skip)
            
        frame_output_iterator = self.process_video_stream(frame_iterator, text_prompts, frame_indices=frame_indices)
        return frame_output_iterator

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
        test_segmentations_path = DATA_DIR / "tip_to_tip_short_segmentations_test.hdf5"
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