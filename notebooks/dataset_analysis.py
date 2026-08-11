import os
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image

# ===========================
# CHANGE THESE PATHS
# ===========================

DATASETS = {
    "PlantVillage": r"datasets/PlantVillage",
    "PlantDoc": r"datasets/PlantDoc",
    "PlantLab2Real": r"datasets/PlantLab2Real"
}

# ===========================

image_extensions = (".jpg", ".jpeg", ".png")

summary = []

for dataset_name, dataset_path in DATASETS.items():

    print(f"\nScanning {dataset_name}...")

    total_images = 0

    for root, dirs, files in os.walk(dataset_path):

        image_count = 0

        for file in files:

            if file.lower().endswith(image_extensions):

                image_count += 1

        if image_count > 0:

            class_name = os.path.basename(root)

            summary.append({
                "Dataset": dataset_name,
                "Class": class_name,
                "Images": image_count
            })

            total_images += image_count

    print(f"Total Images : {total_images}")

df = pd.DataFrame(summary)

print(df)

# ===========================
# Save CSV
# ===========================

os.makedirs("outputs/reports", exist_ok=True)

df.to_csv("outputs/reports/dataset_summary.csv", index=False)

print("\nCSV Saved Successfully")

# ===========================
# Plot Graph
# ===========================

plt.figure(figsize=(14,7))

for dataset in df["Dataset"].unique():

    temp = df[df["Dataset"] == dataset]

    plt.bar(temp["Class"], temp["Images"], label=dataset)

plt.xticks(rotation=90)

plt.ylabel("Number of Images")

plt.title("Dataset Distribution")

plt.legend()

plt.tight_layout()

os.makedirs("outputs/graphs", exist_ok=True)

plt.savefig("outputs/graphs/dataset_distribution.png")

plt.show()

print("\nGraph Saved Successfully")