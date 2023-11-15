import os

import torch
import tqdm
from datasets import load_dataset
from easyrobust.benchmarks import (evaluate_imagenet_a, evaluate_imagenet_r,
                                   evaluate_imagenet_sketch,
                                   evaluate_imagenet_v2, evaluate_imagenet_val)
from torchvision import transforms
from torchvision.datasets import ImageFolder

from utils.misc import RealLabelsImagenet


@torch.no_grad()
def run_robustness(model, root, test_transform, args):
    model.eval()

    acc_val = evaluate_imagenet_val(
        model=model,
        data_dir=os.path.join(root, "benchmarks/data/imagenet-val"),
        test_transform=test_transform,
    )

    acc_a = evaluate_imagenet_a(
        model=model,
        data_dir=os.path.join(root, "benchmarks/data/imagenet-a"),
        test_transform=test_transform,
    )

    acc_r = evaluate_imagenet_r(
        model=model,
        data_dir=os.path.join(root, "benchmarks/data/imagenet-r"),
        test_transform=test_transform,
    )

    acc_sketch = evaluate_imagenet_sketch(
        model=model,
        data_dir=os.path.join(root, "benchmarks/data/imagenet-sketch"),
        test_transform=test_transform,
    )

    acc_v2 = evaluate_imagenet_v2(
        model=model,
        data_dir=os.path.join(root, "benchmarks/data/imagenetv2"),
        test_transform=test_transform,
    )

    # acc_c, _ = evaluate_imagenet_c(
    #     model=model,
    #     data_dir=os.path.join(root, "benchmarks/data/imagenet-c"),
    #     test_transform=test_transform,
    # )

    acc_hard = evaluate_imagenet_hard(model,
                                      test_transform,
                                      device=args.device,
                                      args=args)

    return {
        "acc_a": acc_a,
        "acc_r": acc_r,
        "acc_sketch": acc_sketch,
        "acc_v2": acc_v2,
        "acc_val": acc_val,
        "acc_hard": acc_hard,
        # "acc_c": acc_c,
    }


@torch.no_grad()
def evaluate_imagenet_hard(model, transform, device, args):
    dataset_val = load_dataset("taesiri/imagenet-hard",
                               split="validation",
                               cache_dir=args.data_path)

    if transform is None:
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.Lambda(lambda img: img.convert("RGB")),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

    def apply_transforms(examples):
        examples["pixel_values"] = examples["image"]
        examples["image"] = [transform(image) for image in examples["image"]]
        return examples

    dataset_val.set_transform(apply_transforms)

    def collate_fn(batch):
        labels = [item["label"] for item in batch]
        labels = [label + [-1] * (10 - len(label)) for label in labels]

        return torch.stack([item["image"]
                            for item in batch]), torch.tensor(labels)

    data_loader = torch.utils.data.DataLoader(dataset_val,
                                              batch_size=args.batch_size,
                                              num_workers=args.num_workers,
                                              pin_memory=True,
                                              shuffle=False,
                                              collate_fn=collate_fn)

    correct_ones = 0

    for batch in tqdm(data_loader):
        images, target = batch
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        model_output = model(images)

        preds = model_output.data.max(1)[1]
        correct_ones += (preds[:, None] == target).any(1).sum().item()

    accuracy = 100 * correct_ones / len(data_loader.dataset)
    return accuracy


@torch.no_grad()
def get_accuracy_imagenet_real(model, transform, device, real_labels_imagenet,
                               args):
    dataset_val = ImageFolder(os.path.join(args.data_path),
                              transform=transform)
    paths, labels = zip(*dataset_val.samples)
    filenames = [path.split("/")[-1] for path in paths]
    real_labels_imagenet = RealLabelsImagenet(filenames,
                                              real_json="./real.json")
    data_loader = torch.utils.data.DataLoader(
        dataset_val,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
        shuffle=False,
    )

    model.eval()

    for batch in tqdm(data_loader):
        images, _ = batch
        images = images.to(device, non_blocking=True)

        logits = model(images)

        real_labels_imagenet.add_result(logits)

    accuracy_metrics = real_labels_imagenet.get_accuracy(k=1)

    return {"acc_real": accuracy_metrics}
