import torch
import torch.nn.functional as F
from torchvision import transforms
from tqdm import tqdm

from utils.transforms import CropScale, CropShift


@torch.no_grad()
def run_invariance(data_loader, model, device, args):
    if args.experiment == "scale" or "shift" in args.experiment:
        if args.experiment == "scale":
            crop_transform = CropScale(model_name=args.model,
                                       scale_factor=args.scale_factor)
        elif "shift" in args.experiment:
            crop_transform = CropShift(model_name=args.model,
                                       shift=args.shift_x)
        if transform_val is None:
            transform_val = transforms.Compose([
                crop_transform,
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225]),
            ])
        else:
            normalize = transform_val.transforms[-1]
            assert isinstance(normalize, transforms.Normalize)
            transform_val = transforms.Compose([
                crop_transform,
                transform_val.transforms[-3],
                transform_val.transforms[-2],
                transform_val.transforms[-1],
            ])
    elif args.experiment == "resolution":
        if transform_val is None:
            transform_val = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225]),
            ])
            scaling = 256 / 224
            transform_val.transforms[0].size = int(args.image_size * scaling)
            transform_val.transforms[1].size = args.image_size

    model.eval()

    total_correct_class_probs = 0.0
    total_correct = 0
    total_samples = 0

    for batch in tqdm(data_loader):
        images, target = batch
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        logits = model(images)

        probs = F.softmax(logits, dim=1)
        correct_class_probs = probs[torch.arange(images.size(0)), target]

        _, pred = torch.max(logits, 1)
        total_correct += (pred == target).sum().item()

        total_correct_class_probs += correct_class_probs.sum().item()
        total_samples += images.size(0)

    avg_correct_class_prob = total_correct_class_probs / total_samples
    top1_accuracy = total_correct / total_samples

    return {
        "correct_class_prob": avg_correct_class_prob,
        "top1_accuracy": top1_accuracy,
    }
