import json
import os
import pkgutil
from datetime import datetime

import numpy as np
import open_clip
import torch
import torch.distributed as dist
from torch.utils.data import Dataset
from torchvision.datasets import ImageFolder

from models import models_clip, models_convnextv1, models_deit


class RealLabelsImagenet:

    def __init__(self, filenames, real_json=None, topk=(1, )):
        if real_json is not None:
            with open(real_json) as real_labels:
                real_labels = json.load(real_labels)
        else:
            real_labels = json.loads(
                pkgutil.get_data(
                    __name__,
                    os.path.join('_info',
                                 'imagenet_real_labels.json')).decode('utf-8'))
        real_labels = {
            f'ILSVRC2012_val_{i + 1:08d}.JPEG': labels
            for i, labels in enumerate(real_labels)
        }
        self.real_labels = real_labels
        self.filenames = filenames
        assert len(self.filenames) == len(self.real_labels)
        self.topk = topk
        self.is_correct = {k: [] for k in topk}
        self.sample_idx = 0

    def add_result(self, output):
        maxk = max(self.topk)
        _, pred_batch = output.topk(maxk, 1, True, True)
        pred_batch = pred_batch.cpu().numpy()
        for pred in pred_batch:
            filename = self.filenames[self.sample_idx]
            filename = os.path.basename(filename)
            if self.real_labels[filename]:
                for k in self.topk:
                    self.is_correct[k].append(
                        any([
                            p in self.real_labels[filename] for p in pred[:k]
                        ]))
            self.sample_idx += 1

    def get_accuracy(self, k=None):
        if k is None:
            return {
                k: float(np.mean(self.is_correct[k])) * 100
                for k in self.topk
            }
        else:
            return float(np.mean(self.is_correct[k])) * 100


class HuggingFaceImageDataset(Dataset):

    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform

    def __getitem__(self, idx):
        image = self.dataset[idx]['image']
        label = torch.tensor(self.dataset[idx]['label'])
        return image, label

    def __len__(self):
        return len(self.dataset)


class ImageFolderWithPaths(ImageFolder):

    def __getitem__(self, index):
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)
        filename = os.path.basename(path)
        return sample, target, filename


def resolve_name(args):
    date = datetime.now().strftime("%d-%m-%Y_%H:%M:%S")
    if args.experiment == "robustness":
        return f"{date}_robustness_{args.model}"
    elif args.experiment == "imagenetx":
        return f"{date}_imagenetx_{args.model}"
    elif args.experiment == "imagenethard":
        return f"{date}_imagenethard_{args.model}"
    elif args.experiment == "pug_imagenet":
        return f"{date}_pug_imagenet_{args.model}"
    elif args.experiment == "scale":
        return f"{date}_scale_{args.model}"
    elif args.experiment == "resolution":
        return f"{date}_resolution_{args.model}"
    elif args.experiment == "imagenet_real":
        return f"{date}_real_{args.model}"
    elif "shift" in args.experiment:
        return f"{date}_{args.experiment}_{args.model}"


def load_model_transform(model_name, pretrained_dir, img_size=224):
    print(f"Loading {model_name}")
    checkpoint_path = None
    transform_val = None
    if model_name == "deit3_21k":
        model = models_deit.deit_base_patch16_LS(img_size=img_size)
        checkpoint_path = os.path.join(pretrained_dir,
                                       "deit_3_base_224_21k.pth")
    elif model_name == "convnext_base_21k":
        model = models_convnextv1.convnext_base()
        checkpoint_path = os.path.join(pretrained_dir,
                                       "convnext_base_22k_1k_224.pth")
    elif model_name == "vit_clip":
        model, _, transform_val = open_clip.create_model_and_transforms(
            'ViT-B-16', pretrained='laion400m_e31', force_image_size=img_size)
        model = models_clip.CLIPModel(model=model, model_name='ViT-B-16')
        checkpoint_path = None
    elif model_name == "convnext_clip":
        model, _, transform_val = open_clip.create_model_and_transforms(
            'convnext_base',
            pretrained='laion400m_s13b_b51k',
            force_image_size=img_size)
        model = models_clip.CLIPModel(model=model, model_name='convnext_base')
        checkpoint_path = None

    if checkpoint_path is not None:
        checkpoint = torch.load(checkpoint_path)
        state_dict = checkpoint['model']
        if img_size != 224 and model_name == 'deit3_21k':
            state_dict = interpolate_pos_embed(model, state_dict)
        msg = model.load_state_dict(state_dict, strict=False)
        print(msg)
        assert set(checkpoint['model'].keys()) == set(
            model.state_dict().keys())
        assert len(msg.missing_keys) == 0 and len(
            msg.unexpected_keys
        ) == 0, "Some keys in the state dict do not match"

    return model, transform_val


def interpolate_pos_embed(model, checkpoint_model):
    if 'pos_embed' in checkpoint_model:
        pos_embed_checkpoint = checkpoint_model['pos_embed']
        embedding_size = pos_embed_checkpoint.shape[-1]
        num_patches = model.patch_embed.num_patches
        num_extra_tokens = model.pos_embed.shape[-2] - num_patches
        # height (== width) for the checkpoint position embedding
        orig_size = int(
            (pos_embed_checkpoint.shape[-2] - num_extra_tokens)**0.5)
        # height (== width) for the new position embedding
        new_size = int(num_patches**0.5)
        # class_token and dist_token are kept unchanged
        if orig_size != new_size:
            print("Position interpolate from %dx%d to %dx%d" %
                  (orig_size, orig_size, new_size, new_size))
            extra_tokens = pos_embed_checkpoint[:, :num_extra_tokens]
            # only the position tokens are interpolated
            pos_tokens = pos_embed_checkpoint[:, num_extra_tokens:]
            pos_tokens = pos_tokens.reshape(-1, orig_size, orig_size,
                                            embedding_size).permute(
                                                0, 3, 1, 2)
            pos_tokens = torch.nn.functional.interpolate(
                pos_tokens,
                size=(new_size, new_size),
                mode='bicubic',
                antialias=True,
                align_corners=False)  # antialias set to True
            pos_tokens = pos_tokens.permute(0, 2, 3, 1).flatten(1, 2)
            new_pos_embed = torch.cat((extra_tokens, pos_tokens), dim=1)
            checkpoint_model['pos_embed'] = new_pos_embed
    return checkpoint_model


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def is_main_process():
    return get_rank() == 0
