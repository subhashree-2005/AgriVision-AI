import os
import cv2
import hashlib
import pandas as pd
from tqdm import tqdm

# ===================================================
# CHANGE THESE PATHS IF NEEDED
# ===================================================

DATASETS = {
    "PlantVillage": r"datasets/PlantVillage",
    "PlantDoc": r"datasets/PlantDoc",
    "PlantLab2Real": r"datasets/PlantLab2Real"
}

# ===================================================

image_extensions = (".jpg", ".jpeg", ".png")

corrupted = []
duplicates = []
resolutions = []

hashes = {}

for dataset_name, dataset_path in DATASETS.items():

    print(f"\nScanning {dataset_name}")

    for root, dirs, files in os.walk(dataset_path):

        for file in tqdm(files):

            if not file.lower().endswith(image_extensions):
                continue

            path = os.path.join(root, file)

            try:

                image = cv2.imread(path)

                if image is None:
                    corrupted.append(path)
                    continue

                h, w = image.shape[:2]

                resolutions.append({
                    "Dataset": dataset_name,
                    "Path": path,
                    "Width": w,
                    "Height": h
                })

                with open(path, "rb") as f:
                    filehash = hashlib.md5(f.read()).hexdigest()

                if filehash in hashes:
                    duplicates.append(path)
                else:
                    hashes[filehash] = path

            except:
                corrupted.append(path)

# =====================================

os.makedirs("outputs/reports", exist_ok=True)

pd.DataFrame(resolutions).to_csv(
    "outputs/reports/image_resolutions.csv",
    index=False
)

pd.DataFrame({"Corrupted Images": corrupted}).to_csv(
    "outputs/reports/corrupted_images.csv",
    index=False
)

pd.DataFrame({"Duplicate Images": duplicates}).to_csv(
    "outputs/reports/duplicate_images.csv",
    index=False
)

print("\n===============================")
print("DATASET CLEANING COMPLETED")
print("===============================")

print(f"Corrupted Images : {len(corrupted)}")
print(f"Duplicate Images : {len(duplicates)}")
print(f"Total Images Checked : {len(resolutions)}")

print("\nReports Saved Successfully")