import torch
import torchvision.transforms.functional as F
from torchvision import transforms


class CropScale(torch.nn.Module):

    def __init__(self, model_name, scale_factor):
        super().__init__()
        size = 224 if "clip" in model_name else 256
        self.resize = transforms.Resize(int(scale_factor * size))
        self.center_crop = transforms.CenterCrop(224)

    def forward(self, img):
        return self.center_crop(self.resize(img))


class CropShift(torch.nn.Module):

    def __init__(self, model_name, shift):
        super().__init__()
        self.target_size = 224
        self.resize_dim = 256 if "clip" not in model_name else 224
        self.resize = transforms.Resize((224, 224))
        self.shift = shift

    def forward(self, img):
        w, h = img.size
        if w > h:
            crop_y = (self.resize_dim - 224) / 2
            crop_x = ((w / h * self.resize_dim) - 224) / 2

            top = (crop_y * h) / self.resize_dim
            left = (crop_x * h) / (self.resize_dim) + self.shift

            crop_h = self.target_size * h / self.resize_dim
            crop_w = self.target_size * h / (self.resize_dim)
        else:
            crop_y = (h / w * self.resize_dim - 224) / 2
            crop_x = (self.resize_dim - 224) / 2

            top = (crop_y * w) / (self.resize_dim) + self.shift
            left = (crop_x * w) / self.resize_dim

            crop_h = self.target_size * w / (self.resize_dim)
            crop_w = self.target_size * w / self.resize_dim

        crop = F.crop(img, int(top), int(left), int(crop_h), int(crop_w))
        return self.resize(crop)
