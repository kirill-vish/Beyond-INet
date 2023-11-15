import json
import os

import torch
import torchvision.datasets as datasets
import tqdm
from torch.utils.data import DataLoader
from torchvision import transforms


@torch.no_grad()
def run_pug_imagenet(model, root_folder, transform_val=None):
    with open(os.path.join(root_folder, "class_to_imagenet_idx.json")) as f:
        labels = json.load(f)
    labels = dict(sorted(labels.items()))
    inversed_dict = {}
    counter = 0
    for k, v in labels.items():
        for val in v:
            inversed_dict[int(val)] = counter
        counter = counter + 1

    if transform_val is None:
        tr_normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                            std=[0.229, 0.224, 0.225])
        transform_val = transforms.Compose([
            transforms.CenterCrop(256),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            tr_normalize,
        ])

    dataset_names = [
        "Worlds",
        "Camera_Pitch",
        "Camera_Yaw",
        "Camera_Roll",
        "Object_Pitch",
        "Object_Yaw",
        "Object_Roll",
        "Object_Scale",
        "Object_Texture",
        "Scene_Light",
    ]

    results = {}

    for dataset_name in dataset_names:
        dataset_path = os.path.join(root_folder, dataset_name)
        dataset = datasets.ImageFolder(dataset_path, transform=transform_val)
        dataloader = DataLoader(dataset,
                                batch_size=256,
                                shuffle=False,
                                num_workers=10,
                                drop_last=False)

        print(f"Running inference on {dataset_name}.")

        nb_corrects = 0.0
        for images, labels in tqdm(dataloader):
            images = images.cuda()
            labels = labels.cuda()
            with torch.no_grad(), torch.cuda.amp.autocast():
                output = model(images).softmax(dim=-1)
                pred = torch.argmax(output, dim=1)
                for p in range(pred.size(0)):
                    if pred[p].item() in inversed_dict.keys():
                        pred[p] = inversed_dict[pred[p].item()]
                    else:
                        pred[p] = 999
                nb_corrects += sum((pred == labels).float())

        accuracy = (nb_corrects / len(dataset)) * 100.0
        results[dataset_name] = accuracy.item()

    return results
