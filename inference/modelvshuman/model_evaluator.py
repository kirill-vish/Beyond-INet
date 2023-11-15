import copy
import datetime
import logging
import os

import matplotlib as mpl
import torch
from torch.nn.functional import softmax
from tqdm import tqdm

from utils.misc import load_model_transform

from .evaluation import evaluate as e
from .utils import load_dataset, load_model

logger = logging.getLogger(__name__)
MAX_NUM_MODELS_IN_CACHE = 3

mpl.rcParams['font.size'] = 22


def device():
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class ModelEvaluator:

    def _pytorch_evaluator(self, model_name, model, dataset, *args, **kwargs):
        """
        Evaluate Model on the given dataset and return the accuracy.
        Args:
            model_name:
            model:
            dataset:
            *args:
            **kwargs:
        """
        logging_info = f"Evaluating model {model_name} on dataset {dataset.name} using Pytorch Evaluator"
        logger.info(logging_info)
        print(logging_info)
        for metric in dataset.metrics:
            metric.reset()
        with torch.no_grad():
            result_writer = e.ResultPrinter(model_name=model_name,
                                            dataset=dataset)

            for images, target, paths in tqdm(dataset.loader):
                images = images.to(device())
                if "forward_batch" in dir(model):
                    logits = model.forward_batch(images)
                    softmax_output = model.softmax(logits)
                else:
                    logits = model(images)
                    softmax_output = softmax(logits,
                                             dim=1).detach().cpu().numpy()

                if isinstance(target, torch.Tensor):
                    batch_targets = model.to_numpy(target)
                else:
                    batch_targets = target
                predictions = dataset.decision_mapping(softmax_output)
                for metric in dataset.metrics:
                    metric.update(predictions, batch_targets, paths)
                if kwargs["print_predictions"]:
                    result_writer.print_batch_to_csv(
                        object_response=predictions,
                        batch_targets=batch_targets,
                        paths=paths)

    def _get_datasets(self, dataset_names, *args, **kwargs):
        dataset_list = []
        for dataset in dataset_names:
            dataset = load_dataset(dataset, *args, **kwargs)
            dataset_list.append(dataset)
        return dataset_list

    def _get_evaluator(self, framework):
        if framework == 'pytorch':
            return self._pytorch_evaluator
        else:
            raise NameError("Unsupported evaluator")

    def _remove_model_from_cache(self, framework, model_name):

        def _format_name(name):
            return name.lower().replace("-", "_")

        try:
            if framework == "pytorch":
                cachedir = "/root/.cache/torch/checkpoints/"
                downloaded_models = os.listdir(cachedir)
                for dm in downloaded_models:
                    if _format_name(dm).startswith(_format_name(model_name)):
                        os.remove(os.path.join(cachedir, dm))
        except:
            pass

    def __call__(self, models, dataset_names, *args, **kwargs):
        """
        Wrapper call to _evaluate function.

        Args:
            models:
            dataset_names:
            *args:
            **kwargs:

        Returns:

        """
        logging.info("Model evaluation.")
        _datasets = self._get_datasets(dataset_names, *args, **kwargs)
        for model_name in models:
            datasets = _datasets
            if model_name in [
                    "deit3_21k", "convnext_base_21k", "convnextv2_base",
                    "vit_clip", "convnext_clip"
            ]:
                model, transform_val = load_model_transform(
                    model_name, kwargs['pretrained_dir'])
                if transform_val is not None:
                    datasets[0].loader.dataset.transform = transform_val
                model.cuda()
                framework = "pytorch"
            else:
                print("Using standard model!")
                model, framework = load_model(model_name, *args)
                print(model, framework)
                print(model.model)
            evaluator = self._get_evaluator(framework)
            logger.info(f"Loaded model: {model_name}")
            for dataset in datasets:
                print(dataset.loader.dataset.transform)
                # start time
                time_a = datetime.datetime.now()
                evaluator(model_name, model, dataset, *args, **kwargs)
                for metric in dataset.metrics:
                    logger.info(str(metric))
                    print(metric)

                # end time
                time_b = datetime.datetime.now()
                c = time_b - time_a

                if kwargs["print_predictions"]:
                    # print performances to csv
                    for metric in dataset.metrics:
                        e.print_performance_to_csv(model_name=model_name,
                                                   dataset_name=dataset.name,
                                                   performance=metric.value,
                                                   metric_name=metric.name)
            if len(models) >= MAX_NUM_MODELS_IN_CACHE:
                self._remove_model_from_cache(framework, model_name)

        logger.info("Finished evaluation.")
