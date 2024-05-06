import json
from itertools import chain
from pathlib import Path
import random

import numpy as np
import torch


def get_dataset(augment=True):
    dataset_dict = {}
    for split_name in ["train", "test"]:
        annotations_path = Path(__file__).parent.parent / "data" / "processed" / f"{split_name}.json"
        with open(annotations_path, "r") as f:
            annotations = json.load(f)

        inputs = []
        labels = []

        for annotation in annotations:
            source = annotation["source"]
            target = annotation["target"]

            aspect_ratio = source["aspect_ratio"]
            points = list(chain.from_iterable(source["pose"]))
            inputs.append([aspect_ratio] + points)

            labels.append([
                target["center"][0],
                target["center"][1],
                target["size"],
                target["rotation"] / 360
            ])

            if augment and split_name == "train":
                for i in range(10):
                    x_shift = random.gauss(mu=0.0, sigma=.2)
                    y_shift = random.gauss(mu=0.0, sigma=.2)
                    points = np.array(annotation["source"]["pose"]) + np.array([x_shift, y_shift, 0])
                    inputs.append([aspect_ratio] + list(chain.from_iterable(points)))
                    labels.append([
                        target["center"][0] + x_shift,
                        target["center"][1] + y_shift,
                        target["size"],
                        target["rotation"] / 360
                    ])

        dataset_dict[f"{split_name}_input"] = torch.tensor(inputs, dtype=torch.float32)
        dataset_dict[f"{split_name}_label"] = torch.tensor(labels, dtype=torch.float32)

    return dataset_dict
