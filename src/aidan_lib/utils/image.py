import torch
import torchvision.transforms.v2.functional as Fv2
import torch.nn.functional as F
from jaxtyping import Float, Bool
from typing import Callable, Optional

TorchImage = Float[torch.Tensor, "3 H W"]
TorchMask = Bool[torch.Tensor, "H W"]

def resize_and_pad_images(
    imgs: list[TorchImage],
    masks: Optional[list[TorchMask]] = None,
    max_side_len: int = 1024,
    patch_size: int = 16,
    normalize: bool = True
) -> tuple[
    torch.Tensor,  # imgs_resized
    torch.Tensor,  # valids
    Callable[[torch.Tensor, int], torch.Tensor],  # pixel_mapper
    torch.Tensor,  # grid_sizes
    torch.Tensor,  # original_sizes
    Optional[torch.Tensor],  # masks_resized
    Optional[torch.Tensor],  # overlaps_flat
]:
    batch_size = len(imgs)
    total_image_patch_w = total_image_patch_h = max_side_len // patch_size
    device = imgs[0].device

    imgs_resized = torch.empty((batch_size, 3, max_side_len, max_side_len), dtype=torch.float32, device=device)
    if masks is not None:
        masks_resized = torch.empty((batch_size, 1, max_side_len, max_side_len), dtype=torch.float32, device=device)
    else:
        masks_resized = None

    grid_sizes = torch.empty((batch_size, 2), dtype=torch.long, device=device)
    original_sizes = torch.empty((batch_size, 2), dtype=torch.long, device=device)
    scales = torch.empty((batch_size, 2), dtype=torch.float32, device=device)  # (x scale, y scale)

    valids = torch.full((batch_size, total_image_patch_h, total_image_patch_w), False, dtype=torch.bool, device=device)

    for idx, img in enumerate(imgs):
        assert img.min() >= 0.0
        assert img.max() <= 1.0

        if masks is not None:
            mask = masks[idx]
            assert mask.dtype == torch.bool
            mask = mask.float().unsqueeze(0) 

        orig_h, orig_w = img.shape[1], img.shape[2]
        scale = max_side_len / max(orig_w, orig_h)
        new_w, new_h = orig_w * scale, orig_h * scale

        resize_w = round(new_w / patch_size) * patch_size
        resize_h = round(new_h / patch_size) * patch_size

        scale_w = resize_w / orig_w
        scale_h = resize_h / orig_h

        img_resized = Fv2.resize(img, size=[resize_h, resize_w], interpolation=Fv2.InterpolationMode.BICUBIC, antialias=True)
        
        pad_w = max_side_len - resize_w
        pad_h = max_side_len - resize_h
        
        img_padded = Fv2.pad(img_resized, padding=[0, 0, pad_w, pad_h], padding_mode="constant", fill=0.0)
        imgs_resized[idx] = img_padded
        
        if masks is not None:
            mask_resized = Fv2.resize(mask, size=[resize_h, resize_w], interpolation=Fv2.InterpolationMode.NEAREST, antialias=False)
            mask_padded = Fv2.pad(mask_resized, padding=[0, 0, pad_w, pad_h], padding_mode="constant", fill=0.0)
            masks_resized[idx] = mask_padded

        grid_sizes[idx] = torch.tensor([resize_h // patch_size, resize_w // patch_size], device=device)
        original_sizes[idx] = torch.tensor([orig_w, orig_h], device=device)
        scales[idx] = torch.tensor([scale_w, scale_h], device=device)
        valids[idx, :resize_h // patch_size, :resize_w // patch_size] = True

    overlaps_flat = None
    if masks is not None:
        overlaps = F.interpolate(
            masks_resized,
            size=(total_image_patch_h, total_image_patch_w),
            mode='area'
        )
        overlaps_flat = overlaps.view(batch_size, -1)
        assert overlaps_flat.shape[1] == total_image_patch_h * total_image_patch_w

    if normalize:
        imgs_resized = Fv2.normalize(imgs_resized, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    def pixel_mapper(new_coords: torch.Tensor, batch_idx: int) -> torch.Tensor:
        # new_coords: shape (..., 2) where the last dim is (x, y)
        scale_w = scales[batch_idx, 0]
        scale_h = scales[batch_idx, 1]
        
        old_coords = new_coords.clone().float()
        old_coords[..., 0] /= scale_w
        old_coords[..., 1] /= scale_h
        return old_coords

    return (imgs_resized, valids, pixel_mapper, grid_sizes, original_sizes, masks_resized, overlaps_flat)
