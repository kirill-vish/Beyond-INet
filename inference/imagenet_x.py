import csv
import os
from datetime import datetime

import torch
import torch.nn.functional as F
import tqdm
from imagenet_x import error_ratio, get_factor_accuracies


@torch.no_grad()
def run_imagenetx(data_loader, model, device, model_name):
    model.eval()

    total_samples = 0
    max_probs_list = []
    max_indices_list = []
    file_names_list = []

    date = datetime.now().strftime("%d-%m-%Y_%H:%M:%S")

    folder = os.path.join(os.environ["HOME"],
                          f"outputs/imagenetx/{date}_{model_name}")
    if not os.path.exists(folder):
        os.makedirs(folder)
    output_csv_path = os.path.join(folder, "preds.csv")

    with open(output_csv_path, "w", newline="") as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(
            ["file_name", "predicted_class", "predicted_probability"])

        for batch in tqdm(data_loader):
            images, target, file_names = batch

            images = images.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)

            logits = model(images)

            probs = F.softmax(logits, dim=1)
            max_probs, max_indices = torch.max(probs, dim=1)

            max_probs_list.append(max_probs.detach().cpu().numpy())
            max_indices_list.append(max_indices.detach().cpu().numpy())
            file_names_list.append(file_names)

            for file_name, pred_class, pred_prob in zip(
                    file_names, max_indices, max_probs):
                csv_writer.writerow(
                    [file_name, pred_class.item(),
                     pred_prob.item()])

            total_samples += images.shape[0]

    print("General inference finished successfully!")
    factor_accs = get_factor_accuracies(
        os.path.join(os.environ["HOME"],
                     f"outputs/imagenetx/{date}_{model_name}"))
    return error_ratio(factor_accs)
