import json
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt

from mediapipe_crop_estimate.evaluation_utils import intersection_over_union
from mediapipe_crop_estimate.mediapipe_utils import holistic_body_landmarks_to_rect, Rect

annotations_path = Path(__file__).parent.parent / "data" / "processed" / "test.json"


def original_method(aspect_ratio: float, points):
    height = 1  # just a number
    width = height * aspect_ratio  # Important for angle calculation

    points *= np.array([width, height, 1])
    wrist, index, pinky = points[[2, 4, 5]]
    rect = holistic_body_landmarks_to_rect(wrist, index, pinky)

    size = rect.width / width
    return Rect(x_center=rect.x_center / width,
                y_center=rect.y_center / height,
                width=size,
                height=size,
                rotation=rect.rotation)


@lru_cache(maxsize=None)
def load_mlp(model_name: str):
    model = torch.jit.load(Path(__file__).parent / "mlp" / f"{model_name}.pt")
    model.eval()
    return model


def mlp_method(aspect_ratio: float, points: np.ndarray):
    center_mlp = load_mlp("center")
    size_mlp = load_mlp("size")
    rotation_mlp = load_mlp("rotation")
    with torch.no_grad():
        input_vector = torch.tensor([aspect_ratio] + points.flatten().tolist(), dtype=torch.float32)
        center = center_mlp(input_vector).numpy()
        size = float(size_mlp(input_vector).numpy())
        rotation = float(rotation_mlp(input_vector).numpy()) * 360

    return Rect(x_center=float(center[0]), y_center=float(center[1]),
                width=size, height=size, rotation=rotation)


methods = {
    "original": original_method,
    "mlp": mlp_method,
}

with open(annotations_path, "r") as f:
    annotations = json.load(f)

# plot a histogram
fig = plt.figure()

methods_ious = {}
for method_name, method in methods.items():
    center_error = 0
    size_error = 0
    rotation_error = 0
    iou_total = 0

    min_iou = 1

    method_ious = []

    for annotation in annotations:
        # if annotation["file"] not in ["data/hand_labels/manual_train/036362775_01_l.jpg",
        #                               "data/hand_labels/manual_train/ex1_3.flv_000006_r.jpg"]:
        #     continue

        gold_center = annotation["target"]["center"]
        gold_rect = Rect(x_center=gold_center[0], y_center=gold_center[1],
                         width=annotation["target"]["size"], height=annotation["target"]["size"],
                         rotation=annotation["target"]["rotation"])

        aspect_ratio = annotation["source"]["aspect_ratio"]
        points = np.array(annotation["source"]["pose"])
        rect = method(aspect_ratio, points)

        iou = intersection_over_union(gold_rect, rect)
        if iou < min_iou:
            min_iou = iou

        method_ious.append(iou)
        iou_total += iou
        center_error += np.linalg.norm(np.array([rect.x_center * aspect_ratio, rect.y_center]) -
                                       np.array([gold_rect.x_center * aspect_ratio, gold_rect.y_center]))
        size_error += abs(gold_rect.width - rect.width) / gold_rect.width
        # rotation error is circular
        rotation_error += min(abs(gold_rect.rotation - rect.rotation), abs(gold_rect.rotation - rect.rotation + 360))

    print(f"Method: {method_name}")
    print(f"IOU: {iou_total / len(annotations):.2f}")
    print(f"Center error: {center_error / len(annotations) * 100:.2f}%")
    print(f"Size error: {size_error / len(annotations) * 100:.2f}%")
    print(f"Rotation error: {rotation_error / len(annotations):.2f}")
    print(f"Min IOU: {min_iou:.2f}")
    print()

    plt.hist(method_ious, bins=20, alpha=0.5, label=method_name, density=True)
    methods_ious[method_name] = method_ious

plt.legend(loc='upper left')
plt.gca().axes.get_yaxis().set_visible(False)
plt.tight_layout()
plt.savefig("histogram.pdf")

# count how many times each method wins
wins = {method_name: 0 for method_name in methods}
num_annotations = len(methods_ious["original"])
for i in range(num_annotations):
    ious = {method_name: method_ious[i] for method_name, method_ious in methods_ious.items()}
    best_method = max(ious, key=lambda k: ious[k])
    wins[best_method] += 1
print("Wins", {k: v / num_annotations for k, v in wins.items()})