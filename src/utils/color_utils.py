import cv2
import numpy as np


def get_dominant_color(bgr_crop):
    """
    Get the dominant color name from a BGR image crop.

    Approach:
      1. Focus on the center 50% of the crop (where the object is)
      2. Convert to HSV for reliable color classification
      3. Classify using HSV ranges (hue for chromatic, value for achromatic)

    Args:
        bgr_crop: BGR numpy array.

    Returns:
        Color name string (e.g. "red", "black", "blue").
    """
    if bgr_crop.size == 0:
        return "unknown"

    h, w = bgr_crop.shape[:2]
    if h < 4 or w < 4:
        return "unknown"

    # Focus on center 50% of the crop to avoid background
    cy1, cy2 = h // 4, 3 * h // 4
    cx1, cx2 = w // 4, 3 * w // 4
    center = bgr_crop[cy1:cy2, cx1:cx2]

    # Resize to reduce noise
    center = cv2.resize(center, (24, 24), interpolation=cv2.INTER_AREA)

    # Convert to HSV
    hsv = cv2.cvtColor(center, cv2.COLOR_BGR2HSV)

    # Get median H, S, V (median is more robust than mean to outliers)
    h_vals = hsv[:, :, 0].flatten()
    s_vals = hsv[:, :, 1].flatten()
    v_vals = hsv[:, :, 2].flatten()

    med_h = float(np.median(h_vals))  # 0-179 in OpenCV
    med_s = float(np.median(s_vals))  # 0-255
    med_v = float(np.median(v_vals))  # 0-255

    return _hsv_to_color_name(med_h, med_s, med_v)


def _hsv_to_color_name(h, s, v):
    """
    Map HSV values to a color name.

    OpenCV HSV ranges: H [0-179], S [0-255], V [0-255].

    Args:
        h: Hue (0-179)
        s: Saturation (0-255)
        v: Value/brightness (0-255)

    Returns:
        Color name string.
    """
    # --- Achromatic colors (low saturation) ---
    if s < 40:
        if v < 60:
            return "black"
        elif v < 160:
            return "gray"
        else:
            return "white"

    # --- Very dark = black regardless of hue ---
    if v < 40:
        return "black"

    # --- Dark with some saturation ---
    if v < 80:
        if s > 60:
            # Could be a very dark color — check hue
            if 100 <= h <= 130:
                return "blue"
            elif 35 <= h <= 85:
                return "green"
            elif h <= 10 or h >= 165:
                return "red"
            else:
                return "black"
        return "black"

    # --- Chromatic colors (S >= 40, V >= 80) ---
    # Red wraps around 0/180
    if h <= 8 or h >= 165:
        return "red"
    elif 9 <= h <= 20:
        return "orange"
    elif 21 <= h <= 35:
        return "yellow"
    elif 36 <= h <= 85:
        return "green"
    elif 86 <= h <= 100:
        return "cyan"
    elif 101 <= h <= 130:
        return "blue"
    elif 131 <= h <= 145:
        return "purple"
    elif 146 <= h <= 164:
        return "pink"

    return "unknown"
