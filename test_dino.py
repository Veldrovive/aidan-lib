import torch
from aidan_lib.models.dino_lib_compiled import get_dino_from_repo

device = "cuda" if torch.cuda.is_available() else "cpu"
model = get_dino_from_repo("facebook/dinov3-vits16-pretrain-lvd1689m", device)
dummy_input = torch.randn(1, 3, 224, 224, dtype=torch.bfloat16, device=device)
output = model.forward_features(dummy_input)
print(type(output))
if isinstance(output, dict):
    for k, v in output.items():
        print(f"{k}: {v.shape}")
else:
    print(output.shape)
