import torch
from PIL import Image
from typing import Tuple, Literal
import math
import numpy as np
from dataclasses import dataclass
import torch.nn.functional as F
import torchvision.transforms.functional as TF

try:
    from transformers import AutoImageProcessor, AutoModel
except ImportError:
    raise ImportError("Transformers not installed. Make sure you installed aidan-lib[hf]")


DINOv3BboxEmbeddingMap = dict[Tuple[int, int, int, int], torch.Tensor]

@dataclass
class DINOv3Segmentation:
    person_id: int
    dino_embeddings: torch.Tensor
    dino_overlaps: torch.Tensor
    dino_bboxes: torch.Tensor

DINOv3Checkpoint = Literal[
    "facebook/dinov3-vits16-pretrain-lvd1689m",
    "facebook/dinov3-vits16plus-pretrain-lvd1689m",
    "facebook/dinov3-vitb16-pretrain-lvd1689m",
    "facebook/dinov3-vitl16-pretrain-lvd1689m",
    "facebook/dinov3-vith16plus-pretrain-lvd1689m",
    "facebook/dinov3-vit7b16-pretrain-lvd1689m",
]

DINOv3EmbeddingDimMap: dict[DINOv3Checkpoint, int] = {
    "facebook/dinov3-vits16-pretrain-lvd1689m": 384,
    "facebook/dinov3-vits16plus-pretrain-lvd1689m": 384,
    "facebook/dinov3-vitb16-pretrain-lvd1689m": 768,
    "facebook/dinov3-vitl16-pretrain-lvd1689m": 1024,
    "facebook/dinov3-vith16plus-pretrain-lvd1689m": 1280,
    "facebook/dinov3-vit7b16-pretrain-lvd1689m": 4096,
}

DINOv3PatchSizeMap: dict[DINOv3Checkpoint, int] = {
    "facebook/dinov3-vits16-pretrain-lvd1689m": 16,
    "facebook/dinov3-vits16plus-pretrain-lvd1689m": 16,
    "facebook/dinov3-vitb16-pretrain-lvd1689m": 16,
    "facebook/dinov3-vitl16-pretrain-lvd1689m": 16,
    "facebook/dinov3-vith16plus-pretrain-lvd1689m": 16,
    "facebook/dinov3-vit7b16-pretrain-lvd1689m": 16,
}

class DINOv3Harness:
    max_side_len: int
    checkpoint: DINOv3Checkpoint
    embedding_dim: int
    patch_size: int
    model: torch.nn.Module
    device: torch.device
    

    def __init__(self, checkpoint: DINOv3Checkpoint = "facebook/dinov3-vits16-pretrain-lvd1689m", device="cuda", max_side_len=1024):
        self.device = torch.device(device)
        self.max_side_len = max_side_len
        self.checkpoint = checkpoint
        self.embedding_dim = DINOv3EmbeddingDimMap[checkpoint]
        self.patch_size = DINOv3PatchSizeMap[checkpoint]

        # Bypass the slow processor, only load the model
        self.model = AutoModel.from_pretrained(checkpoint).to(self.device)
        self.model.eval()

        if hasattr(self.model.config, "patch_size"):
            self.patch_size = self.model.config.patch_size
            assert (self.max_side_len / self.patch_size).is_integer(), "max_side_len must be multiple of patch_size"
        else:
            raise ValueError("Patch size not found in model config")

        # Standard ImageNet normalization used by DINO
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(self.device)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(self.device)

    def preprocess_images(self, imgs: list[Image.Image] | list[torch.Tensor]) -> tuple[torch.Tensor, list[tuple[int, int]], list[tuple[int, int]]]:
        """
        Batched resizing, padding, and normalization entirely on GPU.
        Returns:
            batched_padded_images: (B, 3, max_side_len, max_side_len)
            grid_sizes: list of (valid_grid_h, valid_grid_w)
            original_sizes: list of (orig_w, orig_h)
        """
        processed_tensors = []
        grid_sizes = []
        original_sizes = []

        for img in imgs:
            # 1. Ensure we have a GPU float tensor (C, H, W) scaled 0-1
            if isinstance(img, Image.Image):
                orig_w, orig_h = img.size
                img_t = TF.to_tensor(img).to(self.device)
            else:
                orig_h, orig_w = img.shape[1], img.shape[2]
                img_t = img.to(self.device)
                if img_t.dtype != torch.float32 and img_t.dtype != torch.float16:
                    img_t = img_t.float() / 255.0

            original_sizes.append((orig_w, orig_h))

            # 2. Calculate scales
            scale = self.max_side_len / max(orig_w, orig_h)
            new_w, new_h = orig_w * scale, orig_h * scale

            resize_w = int(round(new_w / self.patch_size)) * self.patch_size
            resize_h = int(round(new_h / self.patch_size)) * self.patch_size

            grid_sizes.append((resize_h // self.patch_size, resize_w // self.patch_size))

            # 3. Resize
            # Add batch dim, resize, remove batch dim
            img_resized = F.interpolate(
                img_t.unsqueeze(0), 
                size=(resize_h, resize_w), 
                mode='bicubic', 
                antialias=True
            ).squeeze(0)

            # 4. Pad to max_side_len (pad takes left, right, top, bottom)
            pad_w = self.max_side_len - resize_w
            pad_h = self.max_side_len - resize_h
            img_padded = F.pad(img_resized, (0, pad_w, 0, pad_h), value=0.0)
            
            processed_tensors.append(img_padded)

        # Stack into a single batch
        batched_tensors = torch.stack(processed_tensors)
        
        # 5. Normalize the entire batch at once
        batched_tensors = (batched_tensors - self.mean) / self.std

        return batched_tensors, grid_sizes, original_sizes

    def extract_patch_features(self, imgs: list[Image.Image] | list[torch.Tensor]) -> tuple[list[torch.Tensor], torch.Tensor, list[tuple[int, int]], list[tuple[int, int]]]:
        batched_images, grid_sizes, original_sizes = self.preprocess_images(imgs)

        with torch.no_grad():
            outputs = self.model(pixel_values=batched_images)
            
            last_hidden_state = outputs.last_hidden_state
            cls_tokens = last_hidden_state[:, 0, :]
            patch_tokens = last_hidden_state[:, 5:, :]

            grid_dim = self.max_side_len // self.patch_size
            b, n_patches, dim = patch_tokens.shape

            if n_patches != grid_dim * grid_dim:
                grid_dim = int(math.sqrt(n_patches))

            full_grid_embeddings = patch_tokens.reshape(b, grid_dim, grid_dim, dim)
        
        features_list = []
        for i, (valid_h, valid_w) in enumerate(grid_sizes):
            features_list.append(full_grid_embeddings[i, :valid_h, :valid_w, :].clone())
        
        return features_list, cls_tokens, grid_sizes, original_sizes

    def match_segmentations_to_dino(
        self, 
        images: list[torch.Tensor] | list[torch.Tensor],  
        segs: list[np.ndarray] | list[torch.Tensor]
    ) -> list[list[DINOv3Segmentation]]:
        """
        Uses GPU-accelerated Average Pooling to instantly map masks to patch embeddings.
        """
        features_list, _, grid_sizes, original_sizes = self.extract_patch_features(images)

        dino_segmentations = []

        for img_idx in range(len(images)):
            image_dino_segmentations = []
            segmentation = segs[img_idx]
            
            # Ensure segmentation is a tensor on the correct device
            if isinstance(segmentation, np.ndarray):
                segmentation = torch.from_numpy(segmentation)
            segmentation = segmentation.to(self.device)

            embeddings = features_list[img_idx]  # (Grid_H, Grid_W, Dim)
            valid_grid_h, valid_grid_w = grid_sizes[img_idx]
            orig_w, orig_h = original_sizes[img_idx]

            valid_pixel_w = valid_grid_w * self.patch_size
            valid_pixel_h = valid_grid_h * self.patch_size
            scale_x = valid_pixel_w / orig_w
            scale_y = valid_pixel_h / orig_h

            unique_person_ids = torch.unique(segmentation)

            for person_id in unique_person_ids:
                if person_id == -1:
                    continue

                # Create binary mask (1 for person, 0 for background)
                person_mask = (segmentation == person_id).float().unsqueeze(0).unsqueeze(0) # (1, 1, H, W)

                # Resize mask to the valid image space (nearest neighbor to keep 0s and 1s)
                mask_resized = F.interpolate(person_mask, size=(valid_pixel_h, valid_pixel_w), mode='nearest')

                overlaps = F.avg_pool2d(mask_resized, kernel_size=self.patch_size, stride=self.patch_size).squeeze() # (Grid_H, Grid_W)

                # Find patches where overlap > 0
                valid_y, valid_x = torch.where(overlaps > 0.0)

                if len(valid_y) == 0:
                    continue

                # Vectorized bounding box calculation
                px1 = torch.round((valid_x * self.patch_size) / scale_x).int()
                py1 = torch.round((valid_y * self.patch_size) / scale_y).int()
                px2 = torch.round(((valid_x + 1) * self.patch_size) / scale_x).int()
                py2 = torch.round(((valid_y + 1) * self.patch_size) / scale_y).int()

                dino_bboxes = torch.stack([px1, py1, px2, py2], dim=1)
                dino_overlaps = overlaps[valid_y, valid_x]
                
                # Extract embeddings for the valid patches
                dino_embeddings = embeddings[valid_y, valid_x]

                image_dino_segmentations.append(
                    DINOv3Segmentation(person_id.item(), dino_embeddings, dino_overlaps, dino_bboxes)
                )
            
            dino_segmentations.append(image_dino_segmentations)
                
        return dino_segmentations

    def match_bool_segmentations_to_dino(
        self, 
        images: list[torch.Tensor] | list[torch.Tensor],  
        segs: list[np.ndarray] | list[torch.Tensor]
    ) -> list[list[DINOv3Segmentation]]:
        
        features_list, _, grid_sizes, original_sizes = self.extract_patch_features(images)
        dino_segmentations = []

        for img_idx in range(len(images)):
            segmentation = segs[img_idx]
            
            if isinstance(segmentation, np.ndarray):
                segmentation = torch.from_numpy(segmentation)
            
            # Move to device and ensure it's a float for interpolation
            segmentation = segmentation.to(self.device, non_blocking=True).float()

            valid_grid_h, valid_grid_w = grid_sizes[img_idx]
            orig_w, orig_h = original_sizes[img_idx]

            person_mask = segmentation.unsqueeze(0).unsqueeze(0) # (1, 1, H, W)

            overlaps = F.interpolate(
                person_mask, 
                size=(valid_grid_h, valid_grid_w), 
                mode='area'
            ).squeeze() # (grid_h, grid_w)

            valid_y, valid_x = torch.where(overlaps > 0.0)

            if len(valid_y) == 0:
                dino_segmentations.append([])
                continue

            # Vectorized bounding box calculation
            valid_pixel_w = valid_grid_w * self.patch_size
            valid_pixel_h = valid_grid_h * self.patch_size
            scale_x = valid_pixel_w / orig_w
            scale_y = valid_pixel_h / orig_h

            px1 = torch.round((valid_x * self.patch_size) / scale_x).int()
            py1 = torch.round((valid_y * self.patch_size) / scale_y).int()
            px2 = torch.round(((valid_x + 1) * self.patch_size) / scale_x).int()
            py2 = torch.round(((valid_y + 1) * self.patch_size) / scale_y).int()

            dino_bboxes = torch.stack([px1, py1, px2, py2], dim=1)
            dino_overlaps = overlaps[valid_y, valid_x]
            dino_embeddings = features_list[img_idx][valid_y, valid_x]

            dino_segmentations.append([
                DINOv3Segmentation(1, dino_embeddings, dino_overlaps, dino_bboxes)
            ])
                
        return dino_segmentations

    def embed_pooled(self, imgs: list[Image.Image] | list[torch.Tensor] | torch.Tensor) -> torch.Tensor:
        if isinstance(imgs, torch.Tensor):
            batched_images = imgs.to(self.device)
            if batched_images.dim() == 3:
                batched_images = batched_images.unsqueeze(0)
            if batched_images.dtype not in (torch.float32, torch.float16):
                batched_images = batched_images.float() / 255.0
            batched_images = (batched_images - self.mean) / self.std
        else:
            batched_images, _, _ = self.preprocess_images(imgs)

        with torch.no_grad():
            outputs = self.model(pixel_values=batched_images)
            cls_tokens = outputs.last_hidden_state[:, 0, :]
            
        return cls_tokens

if __name__ == "__main__":
    import time
    
    print("Setting up benchmark...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Create mock batch of 8 high-res images and masks
    batch_size = 64
    mock_images = [torch.rand(3, 720, 1280) for _ in range(batch_size)]
    
    # Create mock masks with 3 distinct "people" (ids 0, 1, 2) and background (-1)
    mock_masks = []
    for _ in range(batch_size):
        mask = torch.full((720, 1280), -1, dtype=torch.long)
        mask[100:300, 200:400] = 0  # Person 0
        mask[400:600, 800:1000] = 1 # Person 1
        mask[200:500, 500:700] = 2  # Person 2
        mock_masks.append(mask)

    PIL_mock_images = [TF.to_pil_image(img) for img in mock_images]

    print("Initializing Models (This takes a moment)...")
    harness = DINOv3Harness(device=device)

    print("\n--- Running Optimized Harness ---")
    start = time.perf_counter()
    new_results = harness.match_segmentations_to_dino(mock_images, mock_masks)
    torch.cuda.synchronize() if device == "cuda" else None
    new_time = time.perf_counter() - start
    
    print(f"Optimized Pipeline Time: {new_time:.4f} seconds")
    print(f"Total people detected across {batch_size} images: {sum(len(img) for img in new_results)}")