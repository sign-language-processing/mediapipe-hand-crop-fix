from shapely.geometry import Polygon

from mediapipe_crop_estimate.mediapipe_utils import Rect


def intersection_over_union(rect1: Rect, rect2: Rect):  # IoU including rotation
    box1 = rect1.get_corners()
    box2 = rect2.get_corners()

    poly1 = Polygon(box1)
    poly2 = Polygon(box2)

    # Intersection polygon
    inter_poly = poly1.intersection(poly2)

    if inter_poly.is_empty:
        return 0.0  # No intersection

    # Areas of the intersection and the union
    inter_area = inter_poly.area
    area1 = poly1.area
    area2 = poly2.area
    union_area = area1 + area2 - inter_area

    # Compute IoU
    return inter_area / union_area
