from dataclasses import dataclass
from sam3.train.data.sam3_image_dataset import InferenceMetadata, FindQueryLoaded, Image as SAMImage, Datapoint
from sam3.train.data.collator import collate_fn_api as collate
from sam3.model.utils.misc import copy_data_to_device
from sam3 import build_sam3_image_model
from sam3.eval.postprocessors import PostProcessImage
from sam3.train.transforms.basic_for_api import ComposeAPI, RandomResizeAPI, ToTensorAPI, NormalizeAPI

from PIL import Image
from pathlib import Path
from jaxtyping import Float, Int, Bool

import torch

@dataclass
class SAM3ImageResult:
    scores: Float[torch.Tensor, "num_segments"]
    labels: Int[torch.Tensor, "num_segments"]
    boxes: Float[torch.Tensor, "num_segments 4"]
    masks: Bool[torch.Tensor, "num_segments H W"]

class SAM3BatchedImageHarness:
    def __init__(self, device: str | torch.device = "cuda", dtype: torch.dtype = torch.bfloat16, model_compile=False):
        print(f"Building SAM3 image model with {dtype} precision")
        self.model = build_sam3_image_model(bpe_path=None, compile=model_compile)
        print(f"Built model, now moving to device {device} with {dtype} precision")
        self.model.to(device)
        self.model.eval()
        print(f"Finished loading model")
        self.dtype = dtype
        self.device = torch.device(device)

        self.postprocessor = PostProcessImage(
            max_dets_per_img=-1,       # if this number is positive, the processor will return topk. For this demo we instead limit by confidence, see below
            iou_type="segm",           # we want masks
            use_original_sizes_box=True,   # our boxes should be resized to the image size
            use_original_sizes_mask=True,   # our masks should be resized to the image size
            convert_mask_to_rle=False, # the postprocessor supports efficient conversion to RLE format. In this demo we prefer the binary format for easy plotting
            detection_threshold=0.5,   # Only return confident detections
            to_cpu=False,
        )

        self.transform = ComposeAPI(
            transforms=[
                RandomResizeAPI(sizes=1008, max_size=1008, square=True, consistent_transform=False),
                ToTensorAPI(),
                NormalizeAPI(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ]
        )

        self.image_id_counter = 0

    def _create_empty_datapoint(self):
        """ A datapoint is a single image on which we can apply several queries at once. """
        return Datapoint(find_queries=[], images=[])

    def _set_image(self, datapoint, pil_image):
        """ Add the image to be processed to the datapoint """
        w,h = pil_image.size
        datapoint.images = [SAMImage(data=pil_image, objects=[], size=[h,w])]

    def _add_text_prompt(self, datapoint, text_query):
        """ Add a text query to the datapoint """

        # in this function, we require that the image is already set.
        # that's because we'll get its size to figure out what dimension to resize masks and boxes
        # In practice you're free to set any size you want, just edit the rest of the function
        assert len(datapoint.images) == 1, "please set the image first"

        image_id = self.image_id_counter
        self.image_id_counter += 1

        w, h = datapoint.images[0].size
        datapoint.find_queries.append(
            FindQueryLoaded(
                query_text=text_query,
                image_id=0,
                object_ids_output=[], # unused for inference
                is_exhaustive=True, # unused for inference
                query_processing_order=0,
                inference_metadata=InferenceMetadata(
                    coco_image_id=image_id,
                    original_image_id=image_id,
                    original_category_id=1,
                    original_size=[w, h],
                    object_id=0,
                    frame_index=0,
                )
            )
        )
        return image_id

    @torch.no_grad()
    def __call__(self, images: list[Image.Image] | list[Path], prompt: str, move_to_cpu: bool = True):
        datapoints = []
        datapoint_ids = []
        for image in images:
            if isinstance(image, Path):
                image = Image.open(image)
            image = image.convert("RGB")
            
            datapoint = self._create_empty_datapoint()
            self._set_image(datapoint, image)
            image_id = self._add_text_prompt(datapoint, prompt)
            datapoint = self.transform(datapoint)

            datapoints.append(datapoint)
            datapoint_ids.append(image_id)

        batch = collate(datapoints, dict_key="dummy")["dummy"]
        batch = copy_data_to_device(batch, self.device, non_blocking=True)
        
        with torch.inference_mode(), torch.autocast(self.device.type, self.dtype, enabled=self.device != torch.device("cpu")):
            output = self.model(batch)
            processed_results = self.postprocessor.process_results(output, batch.find_metadatas)

        results = []
        for datapoint_id in datapoint_ids:
            processed_result = processed_results[datapoint_id]
            move_to_device = lambda x: x.to("cpu") if move_to_cpu else x
            result = SAM3ImageResult(
                scores=move_to_device(processed_result["scores"]),
                labels=move_to_device(processed_result["labels"]),
                boxes=move_to_device(processed_result["boxes"]),
                masks=move_to_device(processed_result["masks"]).squeeze(1),
            )
            results.append(result)
        
        return results