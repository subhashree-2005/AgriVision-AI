"""
preprocessing.py
-----------------
Shared image preprocessing used by predict.py, gradcam_fixed.py,
and the website backend. Keeping this in ONE place means the
website will always preprocess images exactly the same way the
model was trained/evaluated on.
"""

import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array

IMG_HEIGHT = 224
IMG_WIDTH = 224


def load_and_preprocess_image(image_path):
    """
    Loads an image from disk and returns a (1, 224, 224, 3) tensor
    ready to be passed into the CNN, plus the original PIL image
    (useful later for Grad-CAM overlay and for displaying on the website).
    """
    pil_img = load_img(image_path, target_size=(IMG_HEIGHT, IMG_WIDTH))
    array = img_to_array(pil_img)
    array = array / 255.0
    array = np.expand_dims(array, axis=0)
    return array, pil_img
