import os
import cv2
from tqdm import tqdm

# ==========================
# INPUT DATASET
# ==========================
INPUT_DIR = "datasets/PlantVillage"

# OUTPUT DATASET
OUTPUT_DIR = "datasets/processed"

# Image Size
IMG_SIZE = 224

# ==========================

os.makedirs(OUTPUT_DIR, exist_ok=True)

for split in ["train", "val"]:

    split_path = os.path.join(INPUT_DIR, split)

    if not os.path.exists(split_path):
        continue

    for class_name in os.listdir(split_path):

        class_input = os.path.join(split_path, class_name)
        class_output = os.path.join(OUTPUT_DIR, split, class_name)

        os.makedirs(class_output, exist_ok=True)

        images = os.listdir(class_input)

        print(f"\nProcessing {class_name}")

        for image_name in tqdm(images):

            input_image = os.path.join(class_input, image_name)

            output_image = os.path.join(class_output, image_name)

            img = cv2.imread(input_image)

            if img is None:
                continue

            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))

            cv2.imwrite(output_image, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

print("\n================================")
print("Image Preprocessing Completed")
print("================================")