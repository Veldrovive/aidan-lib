import torch
import numpy as np
from aidan_lib.models.dino_lib_compiled import DINOv3CompiledHarness

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Initializing model...")
# Only test the init and basic structures without taking 5 minutes to compile
harness = DINOv3CompiledHarness(warmup=False, backend="eager")
print("Initialization successful.")

# test dummy data
batch_size = 2
mock_images = [torch.rand(3, 720, 1280, device=device) for _ in range(batch_size)]
mock_masks = []
for _ in range(batch_size):
    mask = torch.full((720, 1280), -1, dtype=torch.long, device=device)
    mask[100:300, 200:400] = 0
    mask[400:600, 800:1000] = 1
    mock_masks.append(mask)

print("Running match_segmentations_to_dino...")
segs1 = harness.match_segmentations_to_dino(mock_images, mock_masks)
print(f"Detected: {sum(len(img) for img in segs1)}")

mock_bool_masks = [(mask != -1) for mask in mock_masks]
print("Running match_bool_segmentations_to_dino...")
segs2 = harness.match_bool_segmentations_to_dino(mock_images, mock_bool_masks)
print(f"Detected: {sum(len(img) for img in segs2)}")
