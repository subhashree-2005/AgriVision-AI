import os
import sys
import json
import numpy as np
import tensorflow as tf

from tensorflow.keras.preprocessing.image import ImageDataGenerator

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

import matplotlib.pyplot as plt

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

MODEL_PATH = os.path.join(
    BASE_DIR,
    "saved_models",
    "cnn.keras"
)

CLASS_INDEX_PATH = os.path.join(
    BASE_DIR,
    "saved_models",
    "class_indices.json"
)

VAL_DIR = os.path.join(
    BASE_DIR,
    "datasets",
    "processed",
    "val"
)

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "evaluation"
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)

IMG_HEIGHT = 224
IMG_WIDTH = 224
BATCH_SIZE = 32
# ==========================================================
# LOAD TRAINED MODEL
# ==========================================================

print("\n==========================================")
print("Loading Trained CNN Model...")
print("==========================================")

model = tf.keras.models.load_model(MODEL_PATH)

print("Model Loaded Successfully!")

# ==========================================================
# LOAD CLASS LABELS
# ==========================================================

with open(CLASS_INDEX_PATH, "r") as file:
    class_indices = json.load(file)

class_names = list(class_indices.keys())

print("\nTotal Classes :", len(class_names))

# ==========================================================
# LOAD VALIDATION DATASET
# ==========================================================

print("\n==========================================")
print("Loading Validation Dataset...")
print("==========================================")

datagen = ImageDataGenerator(
    rescale=1.0 / 255.0
)

validation_generator = datagen.flow_from_directory(
    VAL_DIR,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    shuffle=False
)

print("\nValidation Images :", validation_generator.samples)

print("Validation Classes :", validation_generator.num_classes)

print("\nDataset Loaded Successfully!")
print("\nEvaluating Model...")

loss, accuracy = model.evaluate(
    validation_generator,
    verbose=1
)

print("\n===================================")
print("MODEL EVALUATION")
print("===================================")

print(f"Validation Loss     : {loss:.4f}")
print(f"Validation Accuracy : {accuracy*100:.2f}%")

print("\nGenerating Predictions...")

predictions = model.predict(
    validation_generator,
    verbose=1
)

predicted_classes = np.argmax(
    predictions,
    axis=1
)

true_classes = validation_generator.classes

print("Predictions Generated Successfully!")
print("\nClassification Report")

report = classification_report(
    true_classes,
    predicted_classes,
    target_names=list(validation_generator.class_indices.keys())
)

print(report)
print("\nCalculating Performance Metrics...")

accuracy = accuracy_score(
    true_classes,
    predicted_classes
)

precision = precision_score(
    true_classes,
    predicted_classes,
    average="weighted"
)

recall = recall_score(
    true_classes,
    predicted_classes,
    average="weighted"
)

f1 = f1_score(
    true_classes,
    predicted_classes,
    average="weighted"
)

print("\n==========================================")
print("FINAL PERFORMANCE")
print("==========================================")

print(f"Accuracy  : {accuracy*100:.2f}%")
print(f"Precision : {precision*100:.2f}%")
print(f"Recall    : {recall*100:.2f}%")
print(f"F1 Score  : {f1*100:.2f}%")

print("\nGenerating Confusion Matrix...")

cm = confusion_matrix(
    true_classes,
    predicted_classes
)

print(cm)

np.savetxt(
    os.path.join(
        OUTPUT_DIR,
        "confusion_matrix.csv"
    ),
    cm,
    delimiter=",",
    fmt="%d"
)
with open(
    os.path.join(
        OUTPUT_DIR,
        "classification_report.txt"
    ),
    "w"
) as f:

    f.write(report)
with open(
    os.path.join(
        OUTPUT_DIR,
        "metrics.txt"
    ),
    "w"
) as f:

    f.write(f"Accuracy : {accuracy*100:.2f}%\n")
    f.write(f"Precision : {precision*100:.2f}%\n")
    f.write(f"Recall : {recall*100:.2f}%\n")
    f.write(f"F1 Score : {f1*100:.2f}%\n")
print("\n==========================================")
print("Evaluation Completed Successfully!")
print("==========================================")

print("Saved Files:")

print("evaluation/classification_report.txt")

print("evaluation/confusion_matrix.csv")

print("evaluation/metrics.txt")