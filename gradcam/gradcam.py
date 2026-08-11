import os
import json
import cv2
import numpy as np
import tensorflow as tf

# ==========================================================
# PROJECT PATHS
# ==========================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

MODEL_PATH = os.path.join(
    BASE_DIR,
    "saved_models",
    "cnn.keras"
)

IMAGE_PATH = os.path.join(
    BASE_DIR,
    "sample_images",
    "early_blight.jpg"
)

CLASS_PATH = os.path.join(
    BASE_DIR,
    "saved_models",
    "class_indices.json"
)

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "outputs"
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)

# ==========================================================
# LOAD MODEL
# ==========================================================

print("\n========================================")
print("Loading CNN Model...")
print("========================================")

model = tf.keras.models.load_model(
    MODEL_PATH
)

print("Model Loaded Successfully!")

# Build model once
dummy = np.zeros(
    (1, 224, 224, 3),
    dtype=np.float32
)

_ = model.predict(
    dummy,
    verbose=0
)

print("Model Initialized Successfully!")

# ==========================================================
# LOAD CLASS NAMES
# ==========================================================

with open(CLASS_PATH, "r") as f:
    class_indices = json.load(f)

class_names = {
    v: k
    for k, v in class_indices.items()
}

print("\nTotal Classes :", len(class_names))

# ==========================================================
# LOAD IMAGE
# ==========================================================

print("\nLoading Image...")

img = tf.keras.preprocessing.image.load_img(
    IMAGE_PATH,
    target_size=(224, 224)
)

img_array = tf.keras.preprocessing.image.img_to_array(
    img
)

img_array = img_array.astype(
    np.float32
)

img_array /= 255.0

img_array = np.expand_dims(
    img_array,
    axis=0
)

print("Image Loaded Successfully!")
print("Image Shape :", img_array.shape)

# ==========================================================
# FIND LAST CONVOLUTION LAYER
# ==========================================================

print("\nSearching Last Convolution Layer...")

last_conv_layer = None

for layer in reversed(model.layers):

    if isinstance(layer, tf.keras.layers.Conv2D):

        last_conv_layer = layer

        break

if last_conv_layer is None:

    raise ValueError(
        "No Conv2D layer found in model."
    )

print("Last Conv Layer :", last_conv_layer.name)

# ==========================================================
# BUILD GRAD-CAM MODEL
# ==========================================================

print("\n========================================")
print("Building Grad-CAM Model...")
print("========================================")

grad_model = tf.keras.models.Model(

    inputs=model.inputs,

    outputs=[
        last_conv_layer.output,
        model.outputs
    ]

)

print("Grad-CAM Model Created Successfully!")

print("\nInput Shape")
print(grad_model.input_shape)

print("\nNumber of Outputs")
print(len(grad_model.outputs))

print("\nOutput 1 Shape (Feature Maps)")
print(grad_model.outputs[0].shape)

print("\nOutput 2")
print(grad_model.outputs[1])

# ==========================================================
# FORWARD PASS
# ==========================================================

print("\n========================================")
print("Running Forward Pass...")
print("========================================")

feature_maps, predictions = grad_model(
    img_array,
    training=False
)

# If predictions comes as a list, convert to tensor
if isinstance(predictions, list):
    predictions = predictions[0]

print("\nForward Pass Completed Successfully!")

print("\nFeature Map Shape")
print(feature_maps.shape)

print("\nPrediction Shape")
print(predictions.shape)

# Predicted class index
predicted_index = int(
    tf.argmax(predictions[0]).numpy()
)

confidence = float(
    predictions[0][predicted_index].numpy() * 100
)

print("\nPredicted Class Index :", predicted_index)

print("Predicted Class Name  :",
      class_names[predicted_index])

print("Confidence : {:.2f}%".format(confidence))

print("\nTop 5 Predictions")

top5 = tf.argsort(
    predictions[0],
    direction="DESCENDING"
)[:5]

for idx in top5.numpy():

    print(
        class_names[int(idx)],
        " : ",
        "{:.2f}%".format(
            predictions[0][idx].numpy() * 100
        )
    )

# ==========================================================
# COMPUTE GRADIENTS
# ==========================================================

print("\n========================================")
print("Computing Gradients...")
print("========================================")

with tf.GradientTape() as tape:

    # Forward pass again inside GradientTape
    feature_maps, predictions = grad_model(
        img_array,
        training=False
    )

    if isinstance(predictions, list):
        predictions = predictions[0]

    class_channel = predictions[:, predicted_index]

# Tell TensorFlow we need gradients w.r.t. feature maps
gradients = tape.gradient(
    class_channel,
    feature_maps
)

print("\nGradient Type")
print(type(gradients))

if gradients is None:

    print("\nERROR")
    print("Gradient is None")

else:

    print("\nGradient Shape")
    print(gradients.shape)

    print("\nFeature Map Shape")
    print(feature_maps.shape)

