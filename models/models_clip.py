import torch
import torch.nn as nn
from open_clip import (IMAGENET_CLASSNAMES, OPENAI_IMAGENET_TEMPLATES,
                       build_zero_shot_classifier, get_tokenizer)
from open_clip.transformer import VisionTransformer


class CLIPModel(nn.Module):

    def __init__(self, model, model_name):
        super().__init__()
        self.model = model
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        tokenizer = get_tokenizer(model_name)
        self.classifier = build_zero_shot_classifier(
            self.model,
            tokenizer=tokenizer,
            classnames=IMAGENET_CLASSNAMES,
            templates=OPENAI_IMAGENET_TEMPLATES,
            num_classes_per_batch=10,
            device=self.device,
            use_tqdm=True,
        )

    def forward(self, images, return_features=False):
        images = images.to(self.device)
        output = self.model(image=images)
        image_features = (output["image_features"]
                          if isinstance(output, dict) else output[0])
        if return_features:
            return 100.0 * image_features @ self.classifier, image_features
        return 100.0 * image_features @ self.classifier

    def forward_features(self, x, **kwargs):
        if isinstance(self.model.visual, VisionTransformer):
            self.model.visual.output_tokens = True
            pooled, x = self.model.visual(x)
        else:
            x = self.model.visual.trunk.stem(x)
            x = self.model.visual.trunk.stages(x)
        return x
