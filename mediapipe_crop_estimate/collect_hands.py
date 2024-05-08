import json
from pathlib import Path

import cv2
import numpy as np
from pose_format import Pose
from tqdm import tqdm

from mediapipe_crop_estimate.evaluation_utils import intersection_over_union
from mediapipe_crop_estimate.mediapipe_utils import landmarks_to_rect, holistic_body_landmarks_to_rect

if __name__ == "__main__":
    data_dir = Path(__file__).parent.parent / "data"

    temp_dir = data_dir / "temp"
    temp_dir.mkdir(exist_ok=True)

    processed_dir = data_dir / "processed"
    processed_dir.mkdir(exist_ok=True)

    splits = {
        "test": data_dir / "hand_labels" / "manual_test",
        "train": data_dir / "hand_labels" / "manual_train",
    }

    min_iou = 1
    min_iou_file = ""
    max_iou = 0
    max_iou_file = ""

    for split_name, directory in splits.items():
        total_files = 0
        missing_poses = 0
        pose_out_of_rect = 0

        data = []

        # find every json and matching pose file
        json_files = list(directory.glob("*.json"))
        for file in tqdm(json_files):
            total_files += 1

            pose_file = file.with_suffix(".pose")
            if not pose_file.exists():
                print(f"Pose file {pose_file} does not exist")
                continue

            with open(file, "r") as f:
                content = json.load(f)

            with open(pose_file, "rb") as f:
                pose = Pose.read(f.read())

            pose_w = pose.header.dimensions.width
            pose_h = pose.header.dimensions.height
            pose_max = max(pose_w, pose_h)
            aspect_ratio = pose_w / pose_h

            handedness = "LEFT" if content["is_left"] == 1 else "RIGHT"

            pose_points = [f"{handedness}_SHOULDER", f"{handedness}_ELBOW", f"{handedness}_WRIST",
                           f"{handedness}_THUMB", f"{handedness}_INDEX", f"{handedness}_PINKY"]
            pose_body_indexes = [pose.header._get_point_index("POSE_LANDMARKS", point) for point in pose_points]
            pose_points = pose.body.data[0, 0, pose_body_indexes].filled(0)

            if pose.body.confidence[0, 0, pose_body_indexes].sum() == 0:
                missing_poses += 1
                continue

            hand_points = np.array(content["hand_pts"])[:, :2]
            rect = landmarks_to_rect(hand_points)

            # Estimate of number of points in rect
            points_in_rect = [point for point in pose_points if rect.contains(point[0], point[1])]
            if len(points_in_rect) < 3:
                pose_out_of_rect += 1
                continue

            rect.postprocess(scale=2, shift_y=-0.1)

            if handedness == "LEFT":
                pose_points = (pose_points - np.array([pose_w, 0, 0])) * np.array([-1, 1, 1])
                rect.reflect_horizontal(pose_w)

            data.append({
                "file": str(file.with_suffix(".jpg").relative_to(Path(__file__).parent.parent)),
                "source": {
                    "aspect_ratio": aspect_ratio,
                    "pose": (pose_points / np.array([pose_w, pose_h, 1])).tolist()
                },
                "target": {
                    "center": (rect.x_center / pose_w, rect.y_center / pose_h),
                    "size": rect.width / pose_max,
                    "rotation": rect.rotation
                }
            })

            # Save the image in a temp directory to make sure the rect is correct
            temp_file = str(temp_dir / file.with_suffix(".jpg").name)

            image = cv2.imread(str(file.with_suffix(".jpg")))
            if handedness == "LEFT":
                image = cv2.flip(image, 1)
            rect.draw(image)

            for point in pose_points:
                cv2.circle(image, (int(point[0]), int(point[1])), 3, (0, 0, 255), -1)


            # draw lines between shoulder, elbow, wrist, thumb, index, pinky
            def draw_line(p1, p2):
                cv2.line(image, (int(p1[0]), int(p1[1])), (int(p2[0]), int(p2[1])), (0, 255, 0), 2)


            draw_line(pose_points[0], pose_points[1])
            draw_line(pose_points[1], pose_points[2])
            draw_line(pose_points[2], pose_points[3])
            draw_line(pose_points[2], pose_points[4])
            draw_line(pose_points[2], pose_points[5])
            draw_line(pose_points[4], pose_points[5])

            estimated_rect = holistic_body_landmarks_to_rect(pose_points[2], pose_points[4], pose_points[5])
            estimated_rect.draw(image, color=(0, 0, 255))

            cv2.imwrite(temp_file, image)

            iou = intersection_over_union(rect, estimated_rect)
            if iou < min_iou:
                min_iou = iou
                min_iou_file = file
            if iou > max_iou:
                max_iou = iou
                max_iou_file = file

        print(f"Total files: {total_files}")
        print(f"Missing poses: {missing_poses}")
        print(f"Poses out of rect: {pose_out_of_rect}")
        with open(processed_dir / f"{split_name}.json", "w") as f:
            json.dump(data, f, indent=2)

    print(f"Min IoU: {min_iou} for file {min_iou_file}")
    print(f"Max IoU: {max_iou} for file {max_iou_file}")

    # panopticdb_dir = Path(__file__).parent.parent / "data" / "hand143_panopticdb"
