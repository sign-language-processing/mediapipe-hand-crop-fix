from pathlib import Path

import cv2
from pose_format.utils.holistic import load_holistic
from tqdm import tqdm

hand_labels_dir = Path(__file__).parent.parent / "data" / "hand_labels"
panopticdb_dir = Path(__file__).parent.parent / "data" / "hand143_panopticdb"

dataset_dirs = [
    hand_labels_dir / "manual_test",
    hand_labels_dir / "manual_train",
    panopticdb_dir / "imgs"
]

for dataset_dir in dataset_dirs:
    jpg_files = list(dataset_dir.glob("*.jpg"))
    for file in tqdm(jpg_files):
        pose_file = file.with_suffix(".pose")
        if pose_file.exists():
            continue
        image = cv2.cvtColor(cv2.imread(str(file)), cv2.COLOR_BGR2RGB)
        pose = load_holistic([image], fps=30, width=image.shape[1], height=image.shape[0])
        with open(pose_file, "wb") as f:
            pose.write(f)
