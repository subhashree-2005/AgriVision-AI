"""
severity_estimation.py
-----------------------
Simple, transparent severity estimation: estimates what percentage
of the leaf area shows lesion/disease coloring, using HSV color
thresholding. This is NOT a trained model — it's a rule-based
methodology, which is fine for a research prototype as long as you
document that you tuned/validated the thresholds (see note below).

IMPORTANT (for your defense/paper): these thresholds are a starting
point. Run this on 15-20 sample images across different disease
classes, print severity_percent for each, and adjust lesion_lower /
lesion_upper if healthy leaves show high lesion% or diseased leaves
show low lesion%. That tuning process is genuine methodology you can
describe in your report.
"""

import cv2
import numpy as np


# Rough HSV ranges: leaf-green vs brown/yellow lesion coloring.
# Tune these against your own sample images before trusting the output.
LEAF_GREEN_LOWER = np.array([25, 40, 40])
LEAF_GREEN_UPPER = np.array([95, 255, 255])

LESION_LOWER = np.array([5, 40, 40])
LESION_UPPER = np.array([30, 255, 255])


def estimate_severity(image_path):
    """
    Returns a dict:
        {
            "leaf_area_px": int,
            "lesion_area_px": int,
            "severity_percent": float,
            "severity_label": "Mild" | "Moderate" | "Severe" | "Healthy"
        }
    """
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    img_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)

    leaf_mask = cv2.inRange(img_hsv, LEAF_GREEN_LOWER, LEAF_GREEN_UPPER)
    lesion_mask = cv2.inRange(img_hsv, LESION_LOWER, LESION_UPPER)

    # Only count lesion pixels that fall within the leaf region,
    # so background clutter doesn't inflate the severity score.
    combined_leaf_mask = cv2.bitwise_or(leaf_mask, lesion_mask)
    leaf_area_px = int(np.count_nonzero(combined_leaf_mask))
    lesion_area_px = int(np.count_nonzero(lesion_mask))

    if leaf_area_px == 0:
        severity_percent = 0.0
    else:
        severity_percent = round((lesion_area_px / leaf_area_px) * 100, 2)

    if severity_percent < 5:
        label = "Healthy / Very Low"
    elif severity_percent < 20:
        label = "Mild"
    elif severity_percent < 50:
        label = "Moderate"
    else:
        label = "Severe"

    return {
        "leaf_area_px": leaf_area_px,
        "lesion_area_px": lesion_area_px,
        "severity_percent": severity_percent,
        "severity_label": label,
    }


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python severity/severity_estimation.py path/to/image.jpg")
        sys.exit(1)
    result = estimate_severity(sys.argv[1])
    print(result)
