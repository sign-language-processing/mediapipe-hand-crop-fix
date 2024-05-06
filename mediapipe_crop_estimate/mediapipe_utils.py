import cv2
import numpy as np


def normalize_radians(angle):
    return angle - 2 * np.pi * np.floor((angle + np.pi) / (2 * np.pi))


def compute_rotation(landmarks):
    wrist = landmarks[0]
    index = landmarks[5]
    middle = landmarks[9]
    ring = landmarks[13]

    fingers_center = (index + ring) / 2
    fingers_center = (fingers_center + middle) / 2

    rotation = normalize_radians(np.pi / 2 - np.arctan2(-(fingers_center[1] - wrist[1]), fingers_center[0] - wrist[0]))
    return rotation


class Rect:
    def __init__(self, x_center, y_center, width, height, rotation):
        self.x_center = x_center
        self.y_center = y_center
        self.width = width
        self.height = height
        self.rotation = rotation

    def postprocess(self, scale, shift_y):
        size = max(self.width, self.height)  # Square long

        # shift y by 0.1 * size, taking rotation into account
        xShift = shift_y * size * np.sin(self.rotation)
        yShift = shift_y * size * np.cos(self.rotation)
        self.x_center += xShift
        self.y_center += yShift

        # scale size
        size *= scale
        self.width = self.height = size

    def reflect_horizontal(self, width):
        self.x_center = width - self.x_center
        self.rotation = (-self.rotation) % 360

    def draw(self, image, color=(0, 255, 0)):
        box = cv2.boxPoints(((self.x_center, self.y_center), (self.width, self.height), self.rotation))
        cv2.drawContours(image, [np.int0(box)], 0, color, 2)
        # draw a line at the bottom of the box
        cv2.line(image, (int(box[0][0]), int(box[0][1])), (int(box[-1][0]), int(box[-1][1])), (255, 0, 0), 2)

    def contains(self, x: float, y: float):
        return self.x_center - self.width / 2 < x < self.x_center + self.width / 2 and \
            self.y_center - self.height / 2 < y < self.y_center + self.height / 2

    def get_corners(self):
        return cv2.boxPoints(((self.x_center, self.y_center), (self.width, self.height), self.rotation))

    def __str__(self):
        return f"x_center: {self.x_center}, y_center: {self.y_center}, width: {self.width}, height: {self.height}, rotation: {self.rotation}"


def landmarks_to_rect(landmarks):
    rotation = compute_rotation(landmarks)
    reverse_angle = normalize_radians(-rotation)

    min_coords = landmarks.min(axis=0)
    max_coords = landmarks.max(axis=0)
    center_coords = (max_coords + min_coords) / 2

    rotated_min = np.array([float('inf'), float('inf')])
    rotated_max = np.array([-float('inf'), -float('inf')])
    for x, y in (landmarks - center_coords):
        rotated_coords = np.array([
            x * np.cos(reverse_angle) - y * np.sin(reverse_angle),
            x * np.sin(reverse_angle) + y * np.cos(reverse_angle)
        ])
        rotated_min = np.minimum(rotated_min, rotated_coords)
        rotated_max = np.maximum(rotated_max, rotated_coords)

    rotated_center = (rotated_max + rotated_min) / 2
    final_center = (
        rotated_center[0] * np.cos(rotation) - rotated_center[1] * np.sin(rotation) + center_coords[0],
        rotated_center[0] * np.sin(rotation) + rotated_center[1] * np.cos(rotation) + center_coords[1]
    )
    width = (rotated_max[0] - rotated_min[0])
    height = (rotated_max[1] - rotated_min[1])

    return Rect(x_center=final_center[0],
                y_center=final_center[1],
                width=width, height=height,
                rotation=np.degrees(rotation))


def holistic_body_landmarks_to_rect(wrist, index, pinky):
    wrist = wrist[:2]
    index = index[:2]
    pinky = pinky[:2]

    # Estimate middle finger position
    center = (2 * index + pinky) / 3
    # Estimate hand size
    size = 2 * np.linalg.norm(center - wrist)
    # Estimate hand 2D rotation
    rotation = 90 + np.degrees(np.arctan2(center[1] - wrist[1], center[0] - wrist[0]))
    # Shift center
    rect = Rect(center[0], center[1], size, size, rotation)
    rect.postprocess(2.7, -0.1)
    return rect
