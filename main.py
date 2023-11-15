import argparse
import json
import os
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas
import torch
import torch.backends.cudnn as cudnn
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from imagenetv2_pytorch import ImageNetV2Dataset
from torchvision import transforms

import wandb
from inference.imagenet_x import run_imagenetx
from inference.invariance import run_invariance
from inference.pug_imagenet import run_pug_imagenet
from inference.robustness import run_robustness
from utils.misc import (ImageFolderWithPaths, get_world_size,
                        load_model_transform, resolve_name)


def get_args_parser():
    parser = argparse.ArgumentParser("Beyond ImageNet accuracy")
    parser.add_argument(
        "--batch_size",
        default=512,
        type=int,
        help="Batch size per GPU (effective batch size is batch_size * # gpus",
    )
    parser.add_argument("--model",
                        type=str,
                        metavar="MODEL",
                        help="name of model")
    parser.add_argument("--experiment",
                        default="scale",
                        type=str,
                        help="Name of model to train")
    parser.add_argument("--scale_factor", type=float, help="scale factor")
    parser.add_argument("--shift_x", type=int, default=0, help="Shift X")
    parser.add_argument("--shift_y", type=int, default=0, help="Shift Y")
    parser.add_argument("--data_path",
                        type=str,
                        default="",
                        help="dataset path")
    parser.add_argument("--pretrained_dir",
                        type=str,
                        default="pretrained",
                        help="pretrained directory")
    parser.add_argument(
        "--nb_classes",
        default=1000,
        type=int,
        help="number of the classification types",
    )
    parser.add_argument("--image_size",
                        default=224,
                        type=int,
                        help="image size")
    parser.add_argument(
        "--output_dir",
        default="./outputs",
        help="path where to save, empty for no saving",
    )
    parser.add_argument("--device",
                        default="cuda",
                        help="device to use for training / testing")
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--num_workers", default=10, type=int)
    parser.add_argument(
        "--pin_mem",
        action="store_true",
        help=
        "Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.",
    )
    parser.set_defaults(pin_mem=True)

    parser.add_argument(
        "--num_runs",
        default=1,
        type=int,
        help="number of how many repeated runs of experiment",
    )
    parser.add_argument("--n_bins",
                        default=15,
                        type=int,
                        help="number of bins in ECE calculation")
    parser.add_argument("--run_name", type=str, default="")
    parser.add_argument("--dataset", type=str, default="")
    parser.add_argument("--debug", action="store_true")
    return parser


def main(args):
    print("job dir: {}".format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(", ", ",\n"))

    device = torch.device(args.device)

    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    cudnn.benchmark = True
    model_name = None
    transform_val = None
    data_loader_val = None

    model, transform_val = load_model_transform(args.model,
                                                args.pretrained_dir,
                                                args.image_size)

    if transform_val is None:
        transform_val = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])

    print(transform_val)

    if args.experiment == "imagenetx" or args.experiment == "pug_imagenet":
        dataset_val = ImageFolderWithPaths(root=args.data_path,
                                           transform=transform_val)

    else:
        if "imagenetv2" in args.data_path:
            dataset_val = ImageNetV2Dataset("matched-frequency",
                                            transform=transform_val,
                                            location=args.data_path)
        elif "imagenet-r" in args.data_path:
            dataset_val = datasets.ImageFolder(os.path.join(args.data_path),
                                               transform=transform_val)
        elif "imagenet" in args.data_path:
            dataset_val = datasets.ImageFolder(os.path.join(args.data_path),
                                               transform=transform_val)

    if args.experiment != "robustness":
        data_loader_val = torch.utils.data.DataLoader(
            dataset_val,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=True,
            shuffle=False,
        )

    model.to(device)
    model.eval()

    model_without_ddp = model
    n_parameters = sum(p.numel() for p in model.parameters()
                       if p.requires_grad)

    print("Model = %s" % str(model_without_ddp))
    print(args.model)
    print("number of params (M): %.2f" % (n_parameters / 1.0e6))

    eff_batch_size = args.batch_size * get_world_size()

    print("effective batch size: %d" % eff_batch_size)
    if data_loader_val is not None:
        print(data_loader_val.dataset)

    if (args.experiment == "scale" or args.experiment == "shift_xy"
            or args.experiment == "resolution"):
        return run_invariance(data_loader_val, model, device)
    elif args.experiment == "robustness":
        return run_robustness(model, args.data_path, transform_val, args)
    elif args.experiment == "imagenetx":
        return run_imagenetx(data_loader_val, model, device, model_name)
    elif args.experiment == "pug_imagenet":
        return run_pug_imagenet(model, args.data_path, transform_val)
    return


if __name__ == "__main__":
    args = get_args_parser()
    args = args.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    seed_vals = [0, 50, 1000]
    metrics_dict = defaultdict(list)

    name = resolve_name(args)
    run = wandb.init(
        name=name,
        mode="disabled" if args.debug else "online",
        project="beyond-imagenet-accuracy",
    )
    args.run_name = name
    if args.experiment in ["scale", "shift_x", "shift_xy", "resolution"]:
        correct_prob_list = []
        accuracy_list = []
        if args.experiment == "scale":
            transform_vals = [1, 1.25, 1.5, 2, 3]
            for x in transform_vals:
                args.scale_factor = x
                d = main(args)
                correct_prob_list.append(d["correct_class_prob"])
                accuracy_list.append(d["top1_accuracy"])
                run.log({
                    "transform_val": x,
                    "correct_class_prob": d["correct_class_prob"],
                    "accuracy": d["top1_accuracy"],
                })

        elif args.experiment == "shift_xy":
            transform_vals = [0, 5, 30, 75, 100]
            for x in transform_vals:
                args.shift_x = x
                args.shift_y = x
                d = main(args)
                correct_prob_list.append(d["correct_class_prob"])
                accuracy_list.append(d["top1_accuracy"])
                run.log({
                    "transform_val": x,
                    "correct_class_prob": d["correct_class_prob"],
                    "accuracy": d["top1_accuracy"],
                })
        elif args.experiment == "resolution":
            transform_vals = [112, 224, 336, 512, 640]
            for x in transform_vals:
                args.image_size = x
                print(args)
                d = main(args)
                correct_prob_list.append(d["correct_class_prob"])
                accuracy_list.append(d["top1_accuracy"])
                run.log({
                    "transform_val": x,
                    "correct_class_prob": d["correct_class_prob"],
                    "accuracy": d["top1_accuracy"],
                })
        data = {
            "transform_val": transform_vals,
            "correct_class_prob": correct_prob_list,
            "accuracy": accuracy_list,
        }

        with open(f"./scale_shift_reso/{name}.json", "w") as json_file:
            json.dump(data, json_file)
        run.finish()
        exit(0)

    for i in range(args.num_runs):
        args.seed = seed_vals[i]
        results = main(args)
        if isinstance(results, pandas.Series) or isinstance(
                results, pandas.DataFrame):
            results = results.loc["preds"].to_dict()
        for key, value in results.items():
            if run is not None:
                run.log({f"{key}_run_{i}": value})
            metrics_dict[key].append(value)

    mean_std_metrics = {}
    for key, values in metrics_dict.items():
        mean_value = np.mean(values)
        std_value = np.std(values)
        mean_std_metrics[f"{key}_mean"] = mean_value
        mean_std_metrics[f"{key}_std"] = std_value
    print(mean_std_metrics)
    if run is not None:
        run.log(mean_std_metrics)
        run.finish()
