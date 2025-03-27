import torch
import numpy as np

from PIL import Image
from torchvision import transforms

def preprocess_image(path: str) -> torch.Tensor:
    r"""same as torchvision.models.VGG16_Weights.IMAGENET1K_V1.transforms"""
    preprocess = transforms.Compose([
        transforms.Resize(256),          
        transforms.CenterCrop(224),      
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    result = preprocess(Image.open(path))
    return result 

def denormalize(tensor):
    mean = torch.tensor([0.485, 0.456, 0.406])
    std = torch.tensor([0.229, 0.224, 0.225])
    mean_ = mean.view(-1, 1, 1)
    std_ = std.view(-1, 1, 1)
    return tensor * std_ + mean_

def tensor_to_image(tensor):
    tensor = denormalize(tensor)

    # remove batch dimension
    if len(tensor.shape) == 4:
        tensor = tensor.squeeze(0)
    tensor = tensor.detach().cpu()

    # permute (C, H, W) -> (H, W, C)
    tensor = tensor.permute(1, 2, 0)

    numpy_image = tensor.numpy()
    numpy_image = np.clip(numpy_image, 0, 1)    
    numpy_image = (numpy_image * 255).astype(np.uint8)
    return numpy_image

def save_processed_image(tensor, path):
    numpy_image = tensor_to_image(tensor.to("cpu"))
    pil_image = Image.fromarray(numpy_image)
    pil_image.save(path)
