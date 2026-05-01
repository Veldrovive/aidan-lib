import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont
from typing import Union, List, Tuple, Optional
from .colors import get_colors

def visualize_segmentations(
    image: Union[Image.Image, np.ndarray],
    masks: List[np.ndarray] | np.ndarray,
    colors: Optional[List[Union[Tuple[int, int, int], str]]] = None,
    labels: Optional[List[str]] = None,
    alpha: float = 0.5
) -> Union[Image.Image, np.ndarray]:
    """
    Visualizes segmentations on an image with text labels placed at the 
    furthest point from every edge inside the mask (pole of inaccessibility).
    
    Args:
        image: Input image (PIL Image or numpy array).
        masks: List of binary segmentation masks (numpy arrays).
        colors: List of colors for each mask (RGB tuples or hex strings).
        labels: List of text labels for each mask.
        alpha: Transparency of the segmentation masks.
        
    Returns:
        Image of the same type as input with segmentations and labels drawn.
    """
    is_numpy = isinstance(image, np.ndarray)
    
    # Convert input to numpy float32 for blending
    if is_numpy:
        img_array = np.array(image, dtype=np.float32)
    else:
        img_array = np.array(image.convert("RGB"), dtype=np.float32)
        
    text_positions = []
    
    colors = get_colors(len(masks), colors)
    if labels is None:
        actual_labels = [""] * len(masks)
    else:
        actual_labels = labels + [""] * max(0, len(masks) - len(labels))
        
    for mask, color, label in zip(masks, colors, actual_labels):
        bool_mask = mask.astype(bool)
        
        # Convert color to RGB tuple if it isn't
        if isinstance(color, str):
            from PIL import ImageColor
            color_rgb = ImageColor.getrgb(color)
        else:
            color_rgb = tuple(color)
            
        color_arr = np.array(color_rgb, dtype=np.float32)
        
        # Blend this specific mask
        img_array[bool_mask] = img_array[bool_mask] * (1 - alpha) + color_arr * alpha
        
        # Find pole of inaccessibility (furthest point from edge)
        # Using * 255 to be explicitly binary for OpenCV
        uint8_mask = (bool_mask.astype(np.uint8) * 255)
        dist = cv2.distanceTransform(uint8_mask, cv2.DIST_L2, 5)
        _, max_val, _, max_loc = cv2.minMaxLoc(dist)
        
        # Only add text if there's actually a mask present
        if max_val > 0:
            text_positions.append((max_loc, label))
        
    blended = np.clip(img_array, 0, 255).astype(np.uint8)
    
    # Convert to PIL for drawing text
    pil_img = Image.fromarray(blended)
    draw = ImageDraw.Draw(pil_img)
    
    # Try to load a nice font, fallback to default
    try:
        # Load a default sans-serif font if available
        font = ImageFont.truetype("DejaVuSans-Bold.ttf", 16)
    except IOError:
        try:
            font = ImageFont.truetype("arial.ttf", 16)
        except IOError:
            font = ImageFont.load_default()
            
    # Need to check if font has getbbox or textbbox, handle old PIL versions
    has_textbbox = hasattr(draw, 'textbbox')
        
    for max_loc, label in text_positions:
        if label:
            if has_textbbox:
                bbox = draw.textbbox((0, 0), label, font=font)
                text_width = bbox[2] - bbox[0]
                text_height = bbox[3] - bbox[1]
            else:
                # Older PIL fallback
                text_width, text_height = draw.textsize(label, font=font)
            
            x = max_loc[0] - text_width // 2
            y = max_loc[1] - text_height // 2
            
            # Draw text with outline for better visibility
            outline_color = (0, 0, 0)
            text_color = (255, 255, 255)
            
            # Outline
            for dx, dy in [(-1,-1), (-1,1), (1,-1), (1,1), (-1,0), (1,0), (0,-1), (0,1)]:
                draw.text((x + dx, y + dy), label, font=font, fill=outline_color)
            
            # Text
            draw.text((x, y), label, font=font, fill=text_color)
            
    if is_numpy:
        return np.array(pil_img)
    return pil_img

def visualize_segmentations_np(
    image: np.ndarray,
    masks: List[np.ndarray],
    colors: Optional[List[Union[Tuple[int, int, int], str]]] = None,
    labels: Optional[List[str]] = None,
    alpha: float = 0.5
) -> np.ndarray:
    viz = visualize_segmentations(image, masks, colors, labels, alpha)
    assert isinstance(viz, np.ndarray)
    return viz

def visualize_segmentations_pil(
    image: Image.Image,
    masks: List[np.ndarray],
    colors: Optional[List[Union[Tuple[int, int, int], str]]] = None,
    labels: Optional[List[str]] = None,
    alpha: float = 0.5
) -> Image.Image:
    viz = visualize_segmentations(image, masks, colors, labels, alpha)
    assert isinstance(viz, Image.Image)
    return viz