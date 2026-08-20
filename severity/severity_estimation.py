"""
severity_estimation.py
-----------------------
Simple, transparent severity estimation: estimates what percentage
of the leaf area shows lesion/disease coloring, using HSV color
thresholding. This is NOT a trained model -- it's a rule-based
methodology, which is fine for a research prototype as long as you
document that you tuned/validated the thresholds (see note below).

FIX (dark-spot detection): the original lesion mask only looked for
brown/yellow hues with Value (brightness) >= 40. Many real disease
spots -- e.g. Apple Scab -- are dark brown/near-black, often with
brightness BELOW that threshold, so they were silently excluded from
the lesion count entirely. This caused severity to read near-zero
even on visibly, heavily-spotted leaves.

The fix adds a second mask that specifically catches dark, low-
saturation regions (near-black/dark-brown spots) regardless of hue,
and combines it with the original brown/yellow lesion mask.

IMPORTANT (for your defense/paper): these thresholds are a starting
point. Run this on 15-20 sample images across different disease
classes, print severity_percent for each, and adjust the ranges below
if healthy leaves show high lesion% or diseased leaves show low
lesion%. That tuning process is genuine methodology you can describe
in your report.
"""

import cv2
import numpy as np


# Rough HSV ranges: leaf-green vs brown/yellow lesion coloring.
# Tune these against your own sample images before trusting the output.
LEAF_GREEN_LOWER = np.array([25, 40, 40])
LEAF_GREEN_UPPER = np.array([95, 255, 255])

# Brown/yellow lesion coloring (e.g. early blight, rust, mild scab)
LESION_LOWER = np.array([5, 40, 40])
LESION_UPPER = np.array([30, 255, 255])

# Dark / near-black lesion spots (e.g. Apple Scab, late-stage necrosis).
# Hue is unreliable at low brightness, so we match on LOW Value instead,
# with a generous Saturation range to still exclude pure black background.
DARK_SPOT_VALUE_MAX = 90   # pixels darker than this are treated as lesion
DARK_SPOT_SAT_MIN = 15     # excludes pure black / shadow background


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
    h, s, v = cv2.split(img_hsv)

    leaf_mask = cv2.inRange(img_hsv, LEAF_GREEN_LOWER, LEAF_GREEN_UPPER)
    brown_lesion_mask = cv2.inRange(img_hsv, LESION_LOWER, LESION_UPPER)

    # Dark/near-black spot mask: low Value, but not so low-saturation
    # that it's just black background/shadow.
    dark_spot_mask = cv2.inRange(
        img_hsv,
        np.array([0, DARK_SPOT_SAT_MIN, 0]),
        np.array([179, 255, DARK_SPOT_VALUE_MAX]),
    )

    # Combined lesion mask = brown/yellow spots OR dark/black spots
    lesion_mask = cv2.bitwise_or(brown_lesion_mask, dark_spot_mask)

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