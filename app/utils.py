from typing import List, Tuple

import numpy as np


def rgba_overlay_from_mask(mask: np.ndarray, color=(0, 255, 0, 96)) -> np.ndarray:
    """Create an RGBA overlay from a binary mask. color is (B, G, R, A)."""
    if mask.dtype != np.uint8:
        mask = (mask > 0).astype(np.uint8)
    h, w = mask.shape[:2]
    overlay = np.zeros((h, w, 4), dtype=np.uint8)
    overlay[mask > 0] = color
    return overlay


def largest_polygon_from_mask(mask: np.ndarray, epsilon_frac: float = 0.01) -> List[Tuple[float, float]]:
    """Extract the largest polygon from the mask using contours and approxPolyDP.

    Returns a list of (x, y) float coordinates.
    """
    import cv2

    if mask.dtype != np.uint8:
        mask_u8 = (mask > 0).astype(np.uint8) * 255
    else:
        mask_u8 = (mask > 0).astype(np.uint8) * 255

    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return []
    largest = max(contours, key=cv2.contourArea)
    peri = cv2.arcLength(largest, True)
    epsilon = max(1.0, epsilon_frac * peri)
    approx = cv2.approxPolyDP(largest, epsilon, True)
    polygon = [(float(pt[0][0]), float(pt[0][1])) for pt in approx]
    return polygon


def labelme_like_shape(label: str, points: List[Tuple[float, float]]) -> dict:
    return {
        "label": label,
        "points": points,
        "group_id": None,
        "shape_type": "polygon",
        "flags": {}
    }