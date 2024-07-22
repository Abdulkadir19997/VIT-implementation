import torch
from model_initialization.vit_model import ViT

model = ViT(
    img_size=224,
    patch_size=16,
    embed_dim=768,
    hidden_dim=3072,
    num_heads=12,
    num_layers=12
)
print(model)
# Total parameters and trainable parameters.
total_params = sum(p.numel() for p in model.parameters())
print(f"{total_params:,} total parameters.")
total_trainable_params = sum(
    p.numel() for p in model.parameters() if p.requires_grad)
print(f"{total_trainable_params:,} training parameters.")

rnd_int = torch.randn(1, 3, 224, 224)
output = model(rnd_int)
print(f"Output shape from model: {output.shape}")