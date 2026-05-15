import torch
import torch_tensorrt
import torchvision.transforms.v2.functional as Fv2
import torch.nn.functional as F
from typing import Literal, Callable
import torch._dynamo as dynamo

from jaxtyping import Float, Bool

try:
    from transformers import AutoModel
except ImportError:
    raise ImportError("Transformers not installed. Make sure you installed aidan-lib[hf]")

import math
import numpy as np

from aidan_lib.definitions import DINOV3_DIR, DINOV3_VITS16_URL, DINOV3_VITS16_PLUS_URL, DINOV3_VITB16_URL, DINOV3_VITL16_URL, DINOV3_VITH16PLUS_URL, DINOV3_VIT7B16_URL
from aidan_lib.models.dino_lib import DINOv3Segmentation

TorchImage = Float[torch.Tensor, "3 H W"]
TorchMask = Bool[torch.Tensor, "H W"]

DINOv3Checkpoint = Literal[
    "facebook/dinov3-vits16-pretrain-lvd1689m",
    "facebook/dinov3-vits16plus-pretrain-lvd1689m",
    "facebook/dinov3-vitb16-pretrain-lvd1689m",
    "facebook/dinov3-vitl16-pretrain-lvd1689m",
    "facebook/dinov3-vith16plus-pretrain-lvd1689m",
    "facebook/dinov3-vit7b16-pretrain-lvd1689m",
]

DINOv3URLMap: dict[DINOv3Checkpoint, str | None] = {
    "facebook/dinov3-vits16-pretrain-lvd1689m": DINOV3_VITS16_URL,
    "facebook/dinov3-vits16plus-pretrain-lvd1689m": DINOV3_VITS16_PLUS_URL,
    "facebook/dinov3-vitb16-pretrain-lvd1689m": DINOV3_VITB16_URL,
    "facebook/dinov3-vitl16-pretrain-lvd1689m": DINOV3_VITL16_URL,
    "facebook/dinov3-vith16plus-pretrain-lvd1689m": DINOV3_VITH16PLUS_URL,
    "facebook/dinov3-vit7b16-pretrain-lvd1689m": DINOV3_VIT7B16_URL,
}

DINOv3ModelNameMap: dict[DINOv3Checkpoint, str] = {
    "facebook/dinov3-vits16-pretrain-lvd1689m": "dinov3_vits16",
    "facebook/dinov3-vits16plus-pretrain-lvd1689m": "dinov3_vits16plus",
    "facebook/dinov3-vitb16-pretrain-lvd1689m": "dinov3_vitb16",
    "facebook/dinov3-vitl16-pretrain-lvd1689m": "dinov3_vitl16",
    "facebook/dinov3-vith16plus-pretrain-lvd1689m": "dinov3_vith16plus",
    "facebook/dinov3-vit7b16-pretrain-lvd1689m": "dinov3_vit7b16",
}

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

CompileBackend = Literal["inductor", "tensorrt", "cudagraphs", "aotautograd", "nvcc"]

def preprocess_imgs_w_masks(imgs: list[TorchImage], masks: list[TorchMask], max_side_len: int = 1024, patch_size: int = 16, normalize: bool = True):
    batch_size = len(imgs)
    total_image_patch_w = total_image_patch_h = max_side_len // patch_size
    device = imgs[0].device

    imgs_resized = torch.empty((batch_size, 3, max_side_len, max_side_len), dtype=torch.float32, device=device)
    masks_resized = torch.empty((batch_size, 1, max_side_len, max_side_len), dtype=torch.float32, device=device)

    grid_sizes = torch.empty((batch_size, 2), dtype=torch.long, device=device)
    original_sizes = torch.empty((batch_size, 2), dtype=torch.long, device=device)
    scales = torch.empty((batch_size, 2), dtype=torch.float32, device=device)  # (x scale, y scale)

    valids = torch.full((batch_size, total_image_patch_h, total_image_patch_w), False, dtype=torch.bool, device=device)

    for idx, (img, mask) in enumerate(zip(imgs, masks)):
        # Check and ensure proper type and range
        assert img.min() >= 0.0
        assert img.max() <= 1.0
        assert mask.dtype == torch.bool

        # Cast to float for interpolation down the line
        mask = mask.float().unsqueeze(0) 

        orig_h, orig_w = img.shape[1], img.shape[2]
        scale = max_side_len / max(orig_w, orig_h)
        new_w, new_h = orig_w * scale, orig_h * scale

        resize_w = round(new_w / patch_size) * patch_size
        resize_h = round(new_h / patch_size) * patch_size

        scale_w = resize_w / orig_w
        scale_h = resize_h / orig_h

        # Size argument is [H, W]
        img_resized = Fv2.resize(img, size=[resize_h, resize_w], interpolation=Fv2.InterpolationMode.BICUBIC, antialias=True)
        mask_resized = Fv2.resize(mask, size=[resize_h, resize_w], interpolation=Fv2.InterpolationMode.NEAREST, antialias=False)

        pad_w = max_side_len - resize_w
        pad_h = max_side_len - resize_h
        
        # Padding sequence is [left, top, right, bottom]
        img_padded = Fv2.pad(img_resized, padding=[0, 0, pad_w, pad_h], padding_mode="constant", fill=0.0)
        mask_padded = Fv2.pad(mask_resized, padding=[0, 0, pad_w, pad_h], padding_mode="constant", fill=0.0)

        imgs_resized[idx] = img_padded
        masks_resized[idx] = mask_padded
        grid_sizes[idx] = torch.tensor([resize_h // patch_size, resize_w // patch_size], device=device)
        original_sizes[idx] = torch.tensor([orig_w, orig_h], device=device)
        scales[idx] = torch.tensor([scale_w, scale_h], device=device)
        valids[idx, :resize_h // patch_size, :resize_w // patch_size] = True

    # First, get the mask overlap with the patches
    overlaps = F.interpolate(
        masks_resized,
        size=(total_image_patch_h, total_image_patch_w),
        mode='area'
    )

    overlaps_flat = overlaps.view(batch_size, -1)
    assert overlaps_flat.shape[1] == total_image_patch_h * total_image_patch_w

    # Create 1D coordinate arrays
    y_coords = torch.arange(0, max_side_len, patch_size, dtype=torch.float32, device=device)
    x_coords = torch.arange(0, max_side_len, patch_size, dtype=torch.float32, device=device)
    
    # Generate a 2D grid of coordinates
    grid_y, grid_x = torch.meshgrid(y_coords, x_coords, indexing='ij')
    
    # Flatten the 2D grid into row-major order (matches overlaps_flat)
    py1_unscaled = grid_y.reshape(-1)
    px1_unscaled = grid_x.reshape(-1)
    
    py2_unscaled = py1_unscaled + patch_size
    px2_unscaled = px1_unscaled + patch_size

    # We scale by the scaling factors to get the original image coordinates.
    # unsqueeze(0) makes the arrays (1, num_patches) so they broadcast with scales (batch, 1) -> (batch, num_patches)
    px1 = px1_unscaled.unsqueeze(0) / scales[:, 0:1]
    px2 = px2_unscaled.unsqueeze(0) / scales[:, 0:1]
    py1 = py1_unscaled.unsqueeze(0) / scales[:, 1:2]
    py2 = py2_unscaled.unsqueeze(0) / scales[:, 1:2]

    assert px1.shape[1] == total_image_patch_h * total_image_patch_w
    assert px2.shape[1] == total_image_patch_h * total_image_patch_w
    assert py1.shape[1] == total_image_patch_h * total_image_patch_w
    assert py2.shape[1] == total_image_patch_h * total_image_patch_w

    if normalize:
        imgs_resized = Fv2.normalize(imgs_resized, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    
    return imgs_resized, masks_resized, overlaps_flat, px1, px2, py1, py2, valids

def preprocess_imgs(imgs: list[TorchImage], max_side_len: int = 1024, patch_size: int = 16, normalize: bool = True):
    batch_size = len(imgs)
    total_image_patch_w = total_image_patch_h = max_side_len // patch_size
    device = imgs[0].device

    imgs_resized = torch.empty((batch_size, 3, max_side_len, max_side_len), dtype=torch.float32, device=device)

    grid_sizes = torch.empty((batch_size, 2), dtype=torch.long, device=device)
    original_sizes = torch.empty((batch_size, 2), dtype=torch.long, device=device)
    scales = torch.empty((batch_size, 2), dtype=torch.float32, device=device)  # (x scale, y scale)

    valids = torch.full((batch_size, total_image_patch_h, total_image_patch_w), False, dtype=torch.bool, device=device)
    for idx, img in enumerate(imgs):
        # Check and ensure proper type and range
        assert img.min() >= 0.0
        assert img.max() <= 1.0

        orig_h, orig_w = img.shape[1], img.shape[2]
        scale = max_side_len / max(orig_w, orig_h)
        new_w, new_h = orig_w * scale, orig_h * scale

        resize_w = round(new_w / patch_size) * patch_size
        resize_h = round(new_h / patch_size) * patch_size

        scale_w = resize_w / orig_w
        scale_h = resize_h / orig_h

        # Size argument is [H, W]
        img_resized = Fv2.resize(img, size=[resize_h, resize_w], interpolation=Fv2.InterpolationMode.BICUBIC, antialias=True)

        pad_w = max_side_len - resize_w
        pad_h = max_side_len - resize_h
        
        # Padding sequence is [left, top, right, bottom]
        img_padded = Fv2.pad(img_resized, padding=[0, 0, pad_w, pad_h], padding_mode="constant", fill=0.0)

        imgs_resized[idx] = img_padded
        grid_sizes[idx] = torch.tensor([resize_h // patch_size, resize_w // patch_size], device=device)
        original_sizes[idx] = torch.tensor([orig_w, orig_h], device=device)
        scales[idx] = torch.tensor([scale_w, scale_h], device=device)
        valids[idx, :resize_h // patch_size, :resize_w // patch_size] = True

    if normalize:
        imgs_resized = Fv2.normalize(imgs_resized, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    return imgs_resized, grid_sizes, original_sizes, scales, valids

def get_dino_from_repo(checkpoint: DINOv3Checkpoint = "facebook/dinov3-vits16-pretrain-lvd1689m", device="cuda", dtype: torch.dtype = torch.bfloat16):
    model_url = DINOv3URLMap[checkpoint]
    if model_url is None:
        raise ValueError(f"Checkpoint {checkpoint} not found in DINOv3URLMap")
    model_name = DINOv3ModelNameMap[checkpoint]
    if model_name is None:
        raise ValueError(f"Model {checkpoint} not found in DINOv3ModelNameMap")
    
    dino = torch.hub.load(DINOV3_DIR, model_name, source='local', weights=model_url)
    dino = dino.to(device, dtype=dtype).eval()  # pyrefly: ignore
    return dino

def get_compiled_dino_from_repo(
    checkpoint: DINOv3Checkpoint = "facebook/dinov3-vits16-pretrain-lvd1689m", 
    device: str = "cuda", 
    dtype: torch.dtype = torch.bfloat16,
    max_side_len: int = 1024, 
    warmup_batch_size: int = 64, 
    warmup: bool = True,
    backend: CompileBackend = "inductor",
    dynamic: bool = False
) -> Callable:
    print("Loading dino model")
    dino = get_dino_from_repo(checkpoint, device, dtype=dtype)

    print(f"Compiling dino model (backend: {backend}, dynamic: {dynamic})")
    # 1. Pass the dynamic flag to torch.compile
    compiled_dino = torch.compile(
        dino.forward_features, 
        backend=backend, 
        dynamic=dynamic
    )

    if warmup:
        print(f"Performing warmup pass with batch size {warmup_batch_size}...")
        dummy_input = torch.randn(warmup_batch_size, 3, max_side_len, max_side_len, dtype=torch.float32, device=device)
        
        # 2. If dynamic is True, explicitly mark the batch dimension (dim 0) as dynamic
        if dynamic:
            dynamo.mark_dynamic(dummy_input, 0)

        with torch.no_grad():
            with torch.autocast(device_type=torch.device(device).type, dtype=dtype):
                _ = compiled_dino(dummy_input)

    return compiled_dino

class DINOv3CompiledHarness:
    max_side_len: int
    checkpoint: DINOv3Checkpoint
    embedding_dim: int
    patch_size: int
    model: Callable
    device: torch.device

    def __init__(
        self,
        checkpoint: DINOv3Checkpoint = "facebook/dinov3-vits16-pretrain-lvd1689m",
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
        max_side_len: int = 1024,
        warmup_batch_size: int = 64,
        warmup: bool = True,
        backend: CompileBackend = "inductor",
        dynamic: bool = False
    ):
        self.device = torch.device(device)
        self.dtype = dtype
        self.max_side_len = max_side_len
        self.checkpoint = checkpoint
        self.embedding_dim = DINOv3EmbeddingDimMap[checkpoint]
        self.patch_size = DINOv3PatchSizeMap[checkpoint]

        assert (self.max_side_len / self.patch_size).is_integer(), "max_side_len must be multiple of patch_size"

        # Load the compiled model
        self.model = get_compiled_dino_from_repo(
            checkpoint=checkpoint,
            device=device,
            dtype=dtype,
            max_side_len=max_side_len,
            warmup_batch_size=warmup_batch_size,
            warmup=warmup,
            backend=backend,
            dynamic=dynamic
        )

        # Standard ImageNet normalization used by DINO
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(self.device)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(self.device)

    def extract_patch_features(self, imgs: list[torch.Tensor]) -> tuple[list[torch.Tensor], torch.Tensor, list[tuple[int, int]], list[tuple[int, int]]]:
        imgs_resized, grid_sizes, original_sizes, scales, valids = preprocess_imgs(imgs, max_side_len=self.max_side_len, patch_size=self.patch_size, normalize=True)

        with torch.no_grad():
            with torch.autocast(device_type=self.device.type, dtype=self.dtype):
                outputs = self.model(imgs_resized)
            cls_tokens = outputs["x_norm_clstoken"]
            patch_tokens = outputs["x_norm_patchtokens"]

        grid_dim = self.max_side_len // self.patch_size
        b, n_patches, dim = patch_tokens.shape

        if n_patches != grid_dim * grid_dim:
            grid_dim = int(math.sqrt(n_patches))

        full_grid_embeddings = patch_tokens.reshape(b, grid_dim, grid_dim, dim)
        
        features_list = []
        grid_sizes_list = []
        original_sizes_list = []
        
        for i in range(b):
            valid_h, valid_w = int(grid_sizes[i, 0].item()), int(grid_sizes[i, 1].item())
            features_list.append(full_grid_embeddings[i, :valid_h, :valid_w, :].clone())
            grid_sizes_list.append((valid_h, valid_w))
            original_sizes_list.append((int(original_sizes[i, 0].item()), int(original_sizes[i, 1].item())))
        
        return features_list, cls_tokens, grid_sizes_list, original_sizes_list

    def match_bool_segmentations_to_dino(
        self, 
        images: list[torch.Tensor],  
        segs: list[torch.Tensor | np.ndarray]
    ) -> list[list[DINOv3Segmentation]]:
        # Ensure segs are all boolean tensors on the right device
        processed_segs = []
        for seg in segs:
            if isinstance(seg, np.ndarray):
                seg = torch.from_numpy(seg)
            seg = seg.to(self.device).bool()
            processed_segs.append(seg)
            
        imgs_resized, masks_resized, overlaps_flat, px1, px2, py1, py2, valids = preprocess_imgs_w_masks(
            images, processed_segs, max_side_len=self.max_side_len, patch_size=self.patch_size, normalize=True
        )

        with torch.no_grad():
            with torch.autocast(device_type=self.device.type, dtype=self.dtype):
                outputs = self.model(imgs_resized)
            patch_tokens = outputs["x_norm_patchtokens"]
        
        dino_segmentations = []
        b = len(images)
        valids_flat = valids.view(b, -1)

        for img_idx in range(b):
            valid_indices = torch.where((overlaps_flat[img_idx] > 0.0) & valids_flat[img_idx])[0]

            if len(valid_indices) == 0:
                dino_segmentations.append([])
                continue
            
            img_px1 = torch.round(px1[img_idx, valid_indices]).int()
            img_py1 = torch.round(py1[img_idx, valid_indices]).int()
            img_px2 = torch.round(px2[img_idx, valid_indices]).int()
            img_py2 = torch.round(py2[img_idx, valid_indices]).int()

            dino_bboxes = torch.stack([img_px1, img_py1, img_px2, img_py2], dim=1)
            dino_overlaps = overlaps_flat[img_idx, valid_indices]
            dino_embeddings = patch_tokens[img_idx, valid_indices]

            dino_segmentations.append([
                DINOv3Segmentation(1, dino_embeddings, dino_overlaps, dino_bboxes)
            ])

        return dino_segmentations

    def embed_pooled(self, imgs: list[torch.Tensor] | torch.Tensor) -> torch.Tensor:
        if isinstance(imgs, torch.Tensor):
            batched_images = imgs.to(self.device)
            if batched_images.dim() == 3:
                batched_images = batched_images.unsqueeze(0)
            if batched_images.dtype not in (torch.float32, torch.float16, torch.bfloat16):
                batched_images = batched_images.float() / 255.0
            batched_images = (batched_images - self.mean) / self.std
        else:
            batched_images, _, _, _, _ = preprocess_imgs(imgs, max_side_len=self.max_side_len, patch_size=self.patch_size, normalize=True)

        with torch.no_grad():
            with torch.autocast(device_type=self.device.type, dtype=self.dtype):
                outputs = self.model(batched_images)
            cls_tokens = outputs["x_norm_clstoken"]
            
        return cls_tokens