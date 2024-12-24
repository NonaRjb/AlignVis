import sys

sys.path.append("/proj/rep-learning-robotics/users/x_nonra/eeg_asif_img")

import torch.nn as nn

from src.models.image_architectures import VIT, DINO, DEIT, CLIP_IMG

class ImageEncoder(nn.Module):
    def __init__(
        self,
        backbone: str = "ViT",
        embed_dim: int = None,
        add_ln_layer: bool = False,
        model_name: str = "openai/clip-vit-base-patch32",
        **kwargs
    ):
        super().__init__()
        self.backbone = backbone
        if backbone == "ViT":
            self.image_backbone = VIT()
        elif backbone == "DINO":
            self.image_backbone = DINO()
        elif backbone == "DeiT":
            self.image_backbone = DEIT()
        elif backbone == "CLIP_IMG":
            self.image_backbone = CLIP_IMG(model_name=model_name)
        else:
            raise NotImplementedError

        # If add_ln_layer is true, add the FC layer
        if add_ln_layer:
            assert embed_dim is not None, "Embed_dim must be specified when adding FC layer"
            self.fc = nn.Linear(self.image_backbone.embedding_size, embed_dim)
            self.embed_dim = embed_dim
            
        else:
            self.fc = None
            self.embed_dim = self.image_backbone.embedding_size
           
        for param in self.image_backbone.parameters():
            param.requires_grad = False

        print("image embedding size = ", self.embed_dim)

    def forward(self, x):
        # Forward pass through the backbone encoder
        x = self.image_backbone(x)
        # If the FC layer exists, pass through it
        if self.fc is not None:
            x = self.fc(x)
        return x