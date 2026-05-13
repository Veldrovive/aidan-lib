import torch
import torch_tensorrt
import torchvision.transforms.v2.functional as Fv2
import torch.nn.functional as F

from jaxtyping import Float, Bool

TorchImage = Float[torch.Tensor, "3 H W"]
TorchMask = Bool[torch.Tensor, "H W"]

def preprocess_imgs(imgs: list[TorchImage], masks: list[TorchMask], max_side_len: int = 1024, patch_size: int = 16):
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
    
    return imgs_resized, masks_resized, overlaps_flat, px1, px2, py1, py2, valids

