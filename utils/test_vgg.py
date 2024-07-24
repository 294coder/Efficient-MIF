import torch
import torchvision as tv
from torchvision.models.vgg import VGG16_Weights

vgg16 = tv.models.vgg16(weights=VGG16_Weights.IMAGENET1K_V1)
for m in vgg16.modules():
    if isinstance(m, torch.nn.MaxPool2d):
        print('max_pool')

# pass