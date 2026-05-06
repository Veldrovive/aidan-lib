from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from PIL import Image
from typing import Union, List, Dict, Tuple, Any
import numpy as np
import torch
import os

try:
    import sam3
    from sam3 import build_sam3_image_model
    from sam3.train.data.sam3_image_dataset import InferenceMetadata, FindQueryLoaded, Image as SAMImage, Datapoint
    from sam3.train.transforms.basic_for_api import ComposeAPI, RandomResizeAPI, ToTensorAPI, NormalizeAPI
    from sam3.eval.postprocessors import PostProcessImage
    from sam3.train.data.collator import collate_fn_api as collate
    from sam3.model.utils.misc import copy_data_to_device
except ImportError:
    raise ImportError("Meta SAM3 not installed. Make sure you installed it via 'pip install git+https://github.com/facebookresearch/sam3.git'")


@dataclass
class VisualPrompt:
    """Represents a visual prompt consisting of bounding boxes and positive/negative labels."""
    boxes: List[List[float]]  # Expected in [X1, Y1, X2, Y2] format
    labels: List[bool]        # True for positive inclusion, False for exclusion
    text: str = "visual"      # Optional textual hint for the visual boxes


@dataclass
class ImageQuery:
    """Represents a single image and all the queries to be run against it."""
    image: Union[Path, str, Image.Image]
    text_prompts: List[str] = field(default_factory=list)
    visual_prompts: List[VisualPrompt] = field(default_factory=list)


@dataclass
class QueryResult:
    """The segmentation result for a specific prompt on an image."""
    masks: np.ndarray    # Binary masks of shape [N, H, W]
    scores: np.ndarray   # Confidence scores of shape [N]
    boxes: np.ndarray    # Bounding boxes of shape [N, 4]


@dataclass
class SAM3ImageOutput:
    """The combined output for a single image, mapped back to the original prompts."""
    text_results: Dict[str, QueryResult]
    visual_results: List[QueryResult]  # Ordered identically to the input visual_prompts


class BaseSAM3ImageHarness(ABC):
    @abstractmethod
    def predict(self, queries: List[ImageQuery]) -> List[SAM3ImageOutput]:
        pass

    def __call__(self, queries: Union[ImageQuery, List[ImageQuery]]) -> Union[SAM3ImageOutput, List[SAM3ImageOutput]]:
        is_single = isinstance(queries, ImageQuery)
        if is_single:
            queries = [queries]
        
        results = self.predict(queries)
        return results[0] if is_single else results


class SAM3BatchedImageHarness(BaseSAM3ImageHarness):
    def __init__(
        self,
        bpe_path: Union[str, Path, None] = None,
        device: Union[str, torch.device] = "cuda",
        dtype: torch.dtype = torch.bfloat16,
        detection_threshold: float = 0.5,
        image_size: int = 1008,
    ):
        self.device = torch.device(device)
        self.dtype = dtype

        if bpe_path is None:
            sam3_root = os.path.join(os.path.dirname(sam3.__file__), "..")
            bpe_path = os.path.join(sam3_root, "assets", "bpe_simple_vocab_16e6.txt.gz")

        print("Constructing SAM3 Image Model...")
        self.model = build_sam3_image_model(bpe_path=str(bpe_path))
        self.model.to(device=self.device, dtype=self.dtype)
        self.model.eval()

        self.transform = ComposeAPI(
            transforms=[
                RandomResizeAPI(sizes=image_size, max_size=image_size, square=True, consistent_transform=False),
                ToTensorAPI(),
                NormalizeAPI(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ]
        )

        self.postprocessor = PostProcessImage(
            max_dets_per_img=-1,
            iou_type="segm",
            use_original_sizes_box=True,
            use_original_sizes_mask=True,
            convert_mask_to_rle=False,
            detection_threshold=detection_threshold,
            to_cpu=True,
        )

    def _load_image(self, img_input: Union[Path, str, Image.Image]) -> Image.Image:
        if isinstance(img_input, (str, Path)):
            return Image.open(img_input).convert("RGB")
        return img_input

    def _add_text_prompt(self, datapoint: Datapoint, text_query: str, query_id: int):
        w, h = datapoint.images[0].size
        datapoint.find_queries.append(
            FindQueryLoaded(
                query_text=text_query,
                image_id=0,
                object_ids_output=[], 
                is_exhaustive=True, 
                query_processing_order=0,
                inference_metadata=InferenceMetadata(
                    coco_image_id=query_id,
                    original_image_id=query_id,
                    original_category_id=1,
                    original_size=[w, h],
                    object_id=0,
                    frame_index=0,
                )
            )
        )

    def _add_visual_prompt(self, datapoint: Datapoint, vis_prompt: VisualPrompt, query_id: int):
        w, h = datapoint.images[0].size
        
        if len(vis_prompt.boxes) != len(vis_prompt.labels):
            raise ValueError(f"Boxes and labels must have the same length. Got {len(vis_prompt.boxes)} boxes and {len(vis_prompt.labels)} labels.")
            
        labels_tensor = torch.tensor(vis_prompt.labels, dtype=torch.bool).view(-1)
        boxes_tensor = torch.tensor(vis_prompt.boxes, dtype=torch.float).view(-1, 4)

        datapoint.find_queries.append(
            FindQueryLoaded(
                query_text=vis_prompt.text,
                image_id=0,
                object_ids_output=[],
                is_exhaustive=True,
                query_processing_order=0,
                input_bbox=boxes_tensor,
                input_bbox_label=labels_tensor,
                inference_metadata=InferenceMetadata(
                    coco_image_id=query_id,
                    original_image_id=query_id,
                    original_category_id=1,
                    original_size=[w, h],
                    object_id=0,
                    frame_index=0,
                )
            )
        )

    def predict(self, queries: List[ImageQuery]) -> List[SAM3ImageOutput]:
        if not queries:
            return []

        datapoints: List[Datapoint] = []
        query_id_counter = 1
        query_id_mapping: Dict[int, Tuple[int, str, Any]] = {}

        # 1. Build Datapoints
        for img_idx, query in enumerate(queries):
            dp = Datapoint(find_queries=[], images=[])
            pil_img = self._load_image(query.image)
            w, h = pil_img.size
            dp.images = [SAMImage(data=pil_img, objects=[], size=[h, w])]

            for text_prompt in query.text_prompts:
                self._add_text_prompt(dp, text_prompt, query_id_counter)
                query_id_mapping[query_id_counter] = (img_idx, "text", text_prompt)
                query_id_counter += 1

            for vis_idx, vis_prompt in enumerate(query.visual_prompts):
                self._add_visual_prompt(dp, vis_prompt, query_id_counter)
                query_id_mapping[query_id_counter] = (img_idx, "visual", vis_idx)
                query_id_counter += 1

            dp = self.transform(dp)
            datapoints.append(dp)

        # 2. Collate & Move to Device
        batch = collate(datapoints, dict_key="dummy")["dummy"]
        batch = copy_data_to_device(batch, self.device, non_blocking=True)

        # 3. Forward Pass
        with torch.autocast(self.device.type, dtype=self.dtype), torch.inference_mode():
            output = self.model(batch)

        # 4. Post-process
        processed_results = self.postprocessor.process_results(output, batch.find_metadatas)

        # 5. Route results back to original query structures
        outputs = [
            SAM3ImageOutput(
                text_results={}, 
                visual_results=[None] * len(q.visual_prompts)
            ) for q in queries
        ]

        for q_id, (img_idx, p_type, p_key) in query_id_mapping.items():
            if q_id not in processed_results:
                continue
            
            raw_res = processed_results[q_id]
            
            # Safely extract and format tensors to numpy arrays
            masks = raw_res.get("masks", torch.empty(0))
            scores = raw_res.get("scores", torch.empty(0))
            boxes = raw_res.get("boxes", torch.empty(0))

            if isinstance(masks, torch.Tensor):
                masks = masks.numpy()
                if masks.ndim == 4 and masks.shape[1] == 1:
                    masks = masks.squeeze(1)  # Remove channel dim for binary masks [N, H, W]
            
            if isinstance(scores, torch.Tensor):
                scores = scores.numpy()
                
            if isinstance(boxes, torch.Tensor):
                boxes = boxes.numpy()

            q_result = QueryResult(masks=masks, scores=scores, boxes=boxes)

            if p_type == "text":
                outputs[img_idx].text_results[p_key] = q_result
            elif p_type == "visual":
                outputs[img_idx].visual_results[p_key] = q_result

        return outputs