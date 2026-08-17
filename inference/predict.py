"""
predict.py
----------
Standalone prediction module for AgriVision-AI.

Usage (from project root):
    python inference/predict.py path/to/photo.jpg

This is the ONE function ("predict_image") that everything else —
Grad-CAM, severity, the Flask website — should call. Do not
duplicate model-loading or preprocessing logic anywhere else.
"""

import os
import sys
import json
import numpy as np
import tensorflow as tf

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from inference.preprocessing import load_and_preprocess_image  # noqa: E402

MODEL_PATH = os.path.join(BASE_DIR, "saved_models", "cnn.keras")
CLASS_INDEX_PATH = os.path.join(BASE_DIR, "saved_models", "class_indices.json")

_model = None
_class_names = None


def _load_model_and_classes():
    """Loads the model and class list once, then caches them."""
    global _model, _class_names
    if _model is None:
        print("Loading model:", MODEL_PATH)
        _model = tf.keras.models.load_model(MODEL_PATH)

    if _class_names is None:
        with open(CLASS_INDEX_PATH, "r") as f:
            class_indices = json.load(f)
        # class_indices.json maps "ClassName": index -> invert it so
        # we can go from predicted index back to the class name
        index_to_class = {v: k for k, v in class_indices.items()}
        _class_names = [index_to_class[i] for i in range(len(index_to_class))]

    return _model, _class_names


def predict_image(image_path, top_k=3):
    """
    Runs the full prediction pipeline on a single image.

    Returns a dict:
        {
            "image": <filename>,
            "predicted_class": "Tomato___Early_blight",
            "confidence": 94.32,
            "top_k": [
                {"class": "Tomato___Early_blight", "confidence": 94.32},
                {"class": "Tomato___Late_blight", "confidence": 3.21},
                ...
            ],
            "raw_probs": np.ndarray  # full 38-length probability vector
        }
    """
    model, class_names = _load_model_and_classes()

    image_tensor, _pil_img = load_and_preprocess_image(image_path)
    probs = model.predict(image_tensor, verbose=0)[0]  # shape (38,)

    top_indices = np.argsort(probs)[::-1][:top_k]

    result = {
        "image": os.path.basename(image_path),
        "predicted_class": class_names[top_indices[0]],
        "confidence": round(float(probs[top_indices[0]]) * 100, 2),
        "top_k": [
            {"class": class_names[i], "confidence": round(float(probs[i]) * 100, 2)}
            for i in top_indices
        ],
        "raw_probs": probs,
    }
    return result


def print_result(result):
    print("=" * 40)
    print("AGRIVISION-AI PREDICTION")
    print("=" * 40)
    print(f"\nImage       : {result['image']}")
    print(f"\nPrediction  : {result['predicted_class']}")
    print(f"Confidence  : {result['confidence']}%")
    print("\nTop 3:")
    for i, entry in enumerate(result["top_k"], start=1):
        print(f"{i}. {entry['class']:<35} {entry['confidence']}%")
    print("=" * 40)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python inference/predict.py path/to/image.jpg")
        sys.exit(1)

    img_path = sys.argv[1]
    if not os.path.exists(img_path):
        print(f"Error: file not found -> {img_path}")
        sys.exit(1)

    result = predict_image(img_path)
    print_result(result)
