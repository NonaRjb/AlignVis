import torch
from transformers import ViTImageProcessor, ViTForImageClassification, ViTModel, ConvNextFeatureExtractor, ConvNextModel, DeiTImageProcessor, DeiTModel
from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection, CLIPVisionModel
from transformers import AutoProcessor, CLIPModel


class VIT(torch.nn.Module):
    """
    Supervised image encoder based on the original ViT model from Google.
    """
    def __init__(self, model_name: str = "google/vit-base-patch16-224-in21k", device: str = None, alr_preprocessed: bool = True):
        """
        Args:
            model_name: name of the pretrained model to use
            device: device to use for inference
            load_jit: load model just in time (in the encode method)
        """
        super().__init__()
        if not device:
            self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        self.alr_preprocessed = alr_preprocessed
        self.model_name = model_name
        self.feature_extractor = ViTImageProcessor.from_pretrained(model_name)
        self.model = ViTModel.from_pretrained(model_name).to(self.device)
        self.embedding_size = self.model(torch.zeros(1, 3, 224, 224).to(self.device)).last_hidden_state[:,0,:].shape[1]  # 768
        
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        Args:
            images: images to encode
            alr_preprocessed: whether the images are already preprocessed to tensors
        Returns:
            image embeddings
        """
        with torch.no_grad():
            if not self.alr_preprocessed:
                x = self.feature_extractor(images=images, return_tensors="pt").to(self.device)
                x = self.model(**x)
            else:
                x = self.model(images)
            x = x.last_hidden_state[:,0,:]
        return x  # taking the embedding of the CLS token as image representation


class DINO(torch.nn.Module):
    """
    Unsupervised image encoder using the DINO model from Meta Research.
    """
    
    def __init__(self, model_name: str = 'facebook/dino-vits8', device: str = None, alr_preprocessed: bool = True):
        """
        Args:
            model_name: name of the model to use: dino_vit{"s"mall or "b"ase}{"8" or "16" patch size}
            device: device to use.
            load_jit: load model just in time (in the encode method)
        """
        super().__init__()
        if not device:
            self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device 
        self.alr_preprocessed = alr_preprocessed

        self.model_name = model_name
        self.feature_extractor = ViTImageProcessor.from_pretrained(model_name)
        self.model = ViTModel.from_pretrained(model_name, add_pooling_layer=False).to(self.device)
        self.embedding_size = self.model(torch.zeros(1, 3, 224, 224).to(self.device)).last_hidden_state[:,0,:].shape[1]  # 384

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        Args:
            images: images to encode.
            alr_preprocessed: if True, the images are already preprocessed.
        Returns:
            encoded images.
        """

        with torch.no_grad():
            if not self.alr_preprocessed:
                x = self.feature_extractor(images=images, return_tensors="pt").to(self.device)
                x = self.model(**x)
            else:
                x = self.model(images)
            x = x.last_hidden_state[:,0,:]
        return x  # taking the embedding of the CLS token as image representation


class DEIT(torch.nn.Module):
    """
    Supervised image encoder based on the DEIT model from Meta (Visual Transformer).
    """
    def __init__(self, model_name: str = 'facebook/deit-tiny-distilled-patch16-224', device: str = None, alr_preprocessed: bool = True):
        """
        Args:
            model_name: name of the pretrained model to use
            device: device to use for inference
        """
        super().__init__()
        if not device:
            self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        self.alr_preprocessed = alr_preprocessed

        self.feature_extractor = DeiTImageProcessor.from_pretrained(model_name)
        self.model = DeiTModel.from_pretrained(model_name, add_pooling_layer=False).to(self.device)
        self.embedding_size = self.model(torch.zeros(1, 3, 224, 224).to(self.device)).last_hidden_state[:,0,:].shape[1]  # 768

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        Args:
            images: images to encode
            alr_preprocessed: whether the images are already preprocessed to tensors
        Returns:
            image embeddings
        """
        with torch.no_grad():
            if not self.alr_preprocessed:
                x = self.feature_extractor(images=images, return_tensors="pt").to(self.device)
                x = self.model(**x)
            else:
                x = self.model(images)
            x = x.last_hidden_state[:,0,:]
        return x

class CLIP_IMG(torch.nn.Module):
    """
    Self-Supervised image encoder based on the original CLIP model from OpenAI.
    """
    def __init__(self, model_name: str = "openai/clip-vit-base-patch32", device: str = None, alr_preprocessed: bool = True):
        """
        Args:
            model_name: name of the pretrained model to use
            device: device to use for inference
            load_jit: load model just in time (in the encode method)
        """
        super().__init__()
        if not device:
            self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        self.alr_preprocessed = alr_preprocessed
        self.model_name = model_name
        # self.feature_extractor = CLIPImageProcessor.from_pretrained(model_name)
        # self.model = CLIPVisionModel.from_pretrained(model_name).to(self.device)
        # self.embedding_size = self.model(torch.zeros(1, 3, 224, 224).to(self.device)).last_hidden_state[:,0,:].shape[1]  # 768

        self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(self.device)
        self.feature_extractor = AutoProcessor.from_pretrained("openai/clip-vit-base-patch32")
        self.embedding_size = self.model.get_image_features(torch.zeros(1, 3, 224, 224).to(self.device)).shape[1]  # 512
        
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        Args:
            images: images to encode
            alr_preprocessed: whether the images are already preprocessed to tensors
        Returns:
            image embeddings
        """
        with torch.no_grad():
            if not self.alr_preprocessed:
                x = self.feature_extractor(images=images, return_tensors="pt").to(self.device)
                # x = self.model(**x)
                x = self.model.get_image_features(**x)
            else:
                # x = self.model(images)
                x = self.model.get_image_features(images)
            # x = x.last_hidden_state[:,0,:]
        return x  # taking the embedding of the CLS token as image representation