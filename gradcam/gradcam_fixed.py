"""
gradcam_fixed.py
-----------------
Fixed Grad-CAM implementation for cnn.keras.

Root cause of the "gradient is None" bug: cnn.keras was saved as a
Sequential model. When you build a new Model() using an intermediate
layer's .output directly from a reloaded Sequential model, Keras
sometimes fails to preserve full graph connectivity, so
GradientTape can't trace a path from the input to that layer.

Fix: rebuild an equivalent Functional-API model by manually piping
each layer's output into the next, then attach Grad-CAM to THAT model.

Usage:
    python gradcam/gradcam_fixed.py path/to/photo.jpg
"""

import os
import sys
import json
import numpy as np
import tensorflow as tf
import cv2

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from inference.preprocessing import load_and_preprocess_image  # noqa: E402

MODEL_PATH = os.path.join(BASE_DIR, "saved_models", "cnn.keras")
CLASS_INDEX_PATH = os.path.join(BASE_DIR, "saved_models", "class_indices.json")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Name of the last convolutional layer in cnn.keras.
# Verify this matches model.summary() on YOUR saved model before running.
LAST_CONV_LAYER_NAME = "conv2d_2"


def load_functional_model(model_path):
    """
    Loads the saved Sequential model and rebuilds it as a Functional
    model so GradientTape can trace gradients to intermediate layers.
    """
    sequential_model = tf.keras.models.load_model(model_path)

    inputs = tf.keras.Input(shape=sequential_model.input_shape[1:])
    x = inputs
    layer_outputs = {}
    for layer in sequential_model.layers:
        x = layer(x)
        layer_outputs[layer.name] = x

    functional_model = tf.keras.Model(inputs=inputs, outputs=x)
    return functional_model, layer_outputs


def make_gradcam_heatmap(img_array, functional_model, layer_outputs,
                          last_conv_layer_name, pred_index=None):
    conv_output = layer_outputs[last_conv_layer_name]
    grad_model = tf.keras.Model(
        inputs=functional_model.inputs,
        outputs=[conv_output, functional_model.output],
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        tape.watch(conv_outputs)
        if pred_index is None:
            pred_index = tf.argmax(predictions[0])
        class_channel = predictions[:, pred_index]

    grads = tape.gradient(class_channel, conv_outputs)

    if grads is None:
        raise RuntimeError(
            "Gradient is still None. Double-check LAST_CONV_LAYER_NAME "
            "matches an actual Conv2D layer name from model.summary()."
        )

    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-8)
    return heatmap.numpy(), int(pred_index)


def overlay_heatmap(pil_img, heatmap, alpha=0.4):
    img = np.array(pil_img)
    heatmap_resized = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap_uint8 = np.uint8(255 * heatmap_resized)
    heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)
    overlaid = np.uint8(heatmap_color * alpha + img * (1 - alpha))
    return overlaid


def run_gradcam(image_path, save=True):
    with open(CLASS_INDEX_PATH, "r") as f:
        class_indices = json.load(f)
    index_to_class = {v: k for k, v in class_indices.items()}

    functional_model, layer_outputs = load_functional_model(MODEL_PATH)
    img_array, pil_img = load_and_preprocess_image(image_path)

    heatmap, pred_index = make_gradcam_heatmap(
        img_array, functional_model, layer_outputs, LAST_CONV_LAYER_NAME
    )
    predicted_class = index_to_class[pred_index]
    overlaid = overlay_heatmap(pil_img, heatmap)

    print(f"Predicted class : {predicted_class} (index {pred_index})")

    if save:
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        out_path = os.path.join(OUTPUT_DIR, f"gradcam_{base_name}.jpg")
        cv2.imwrite(out_path, cv2.cvtColor(overlaid, cv2.COLOR_RGB2BGR))
        print(f"Saved Grad-CAM overlay -> {out_path}")

    return overlaid, predicted_class


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python gradcam/gradcam_fixed.py path/to/image.jpg")
        sys.exit(1)
    run_gradcam(sys.argv[1])
