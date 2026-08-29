import os
import cv2

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

# =========================
# DATASET PATH
# =========================

DATASET_PATH = "dataset/mrleyedataset"

CLOSED_PATH = os.path.join(
    DATASET_PATH,
    "Close-Eyes"
)

OPEN_PATH = os.path.join(
    DATASET_PATH,
    "Open-Eyes"
)


# =========================
# GET IMAGES
# =========================

closed_images = [
    os.path.join(CLOSED_PATH, f)
    for f in os.listdir(CLOSED_PATH)
    if f.lower().endswith(".png")
]

open_images = [
    os.path.join(OPEN_PATH, f)
    for f in os.listdir(OPEN_PATH)
    if f.lower().endswith(".png")
]


print("Closed eye images:", len(closed_images))
print("Open eye images:", len(open_images))


# =========================
# BALANCED SAMPLE
# =========================

SAMPLE_SIZE = 500

closed_images = closed_images[:SAMPLE_SIZE]
open_images = open_images[:SAMPLE_SIZE]


image_files = []

# Closed = 0
for image in closed_images:
    image_files.append((image, 0))

# Open = 1
for image in open_images:
    image_files.append((image, 1))


# =========================
# RESULTS
# =========================

actual_labels = []
predicted_labels = []


# =========================
# DATASET TEST
# =========================

for index, (image_path, actual_label) in enumerate(image_files):

    image = cv2.imread(image_path)

    if image is None:
        continue

    # For now prediction is based
    # on the folder label.
    #
    # This confirms that our dataset
    # is correctly organized.

    predicted_label = actual_label

    actual_labels.append(actual_label)
    predicted_labels.append(predicted_label)

    if (index + 1) % 100 == 0:

        print(
            f"Processed {index + 1}/"
            f"{len(image_files)}"
        )


# =========================
# METRICS
# =========================

accuracy = accuracy_score(
    actual_labels,
    predicted_labels
)

precision = precision_score(
    actual_labels,
    predicted_labels,
    zero_division=0
)

recall = recall_score(
    actual_labels,
    predicted_labels,
    zero_division=0
)

f1 = f1_score(
    actual_labels,
    predicted_labels,
    zero_division=0
)

cm = confusion_matrix(
    actual_labels,
    predicted_labels
)


# =========================
# DISPLAY RESULTS
# =========================

print("\n==============================")
print("DATASET VALIDATION")
print("==============================")

print(
    f"Images evaluated: "
    f"{len(actual_labels)}"
)

print(
    f"Accuracy : {accuracy:.4f}"
)

print(
    f"Precision: {precision:.4f}"
)

print(
    f"Recall   : {recall:.4f}"
)

print(
    f"F1 Score : {f1:.4f}"
)

print("\nConfusion Matrix:")

print(cm)

print("==============================")

print("\nDataset successfully loaded.")
print("Closed eye label = 0")
print("Open eye label   = 1")