import os
import cv2
import numpy as np

from skimage.feature import hog
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)
from sklearn.model_selection import train_test_split

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
# SETTINGS
# =========================

IMAGES_PER_CLASS = 2000
IMAGE_SIZE = (64, 64)

# =========================
# FEATURE EXTRACTION
# =========================

def extract_features(image_path):

    image = cv2.imread(
        image_path,
        cv2.IMREAD_GRAYSCALE
    )

    if image is None:
        return None

    image = cv2.resize(
        image,
        IMAGE_SIZE
    )

    # HOG feature extraction
    features = hog(
        image,
        orientations=9,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        block_norm="L2-Hys"
    )

    return features


# =========================
# LOAD DATASET
# =========================

print("Loading dataset...")

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

closed_images = closed_images[:IMAGES_PER_CLASS]
open_images = open_images[:IMAGES_PER_CLASS]

print(f"Closed images selected: {len(closed_images)}")
print(f"Open images selected: {len(open_images)}")

features = []
labels = []

# =========================
# CLOSED EYES
# Label = 0
# =========================

print("\nProcessing closed-eye images...")

for i, image_path in enumerate(closed_images):

    feature = extract_features(image_path)

    if feature is not None:

        features.append(feature)
        labels.append(0)

    if (i + 1) % 500 == 0:

        print(
            f"Closed: {i + 1}/{len(closed_images)}"
        )


# =========================
# OPEN EYES
# Label = 1
# =========================

print("\nProcessing open-eye images...")

for i, image_path in enumerate(open_images):

    feature = extract_features(image_path)

    if feature is not None:

        features.append(feature)
        labels.append(1)

    if (i + 1) % 500 == 0:

        print(
            f"Open: {i + 1}/{len(open_images)}"
        )


X = np.array(features)
y = np.array(labels)

print("\n==============================")
print("FEATURE EXTRACTION COMPLETE")
print("==============================")
print("Feature matrix:", X.shape)
print("Labels:", y.shape)


# =========================
# TRAIN / TEST SPLIT
# =========================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=42,
    stratify=y
)

print("\nTraining images:", len(X_train))
print("Testing images:", len(X_test))


# =========================
# TRAIN SVM
# =========================

print("\nTraining SVM classifier...")

model = SVC(
    kernel="rbf",
    C=10,
    gamma="scale"
)

model.fit(
    X_train,
    y_train
)

print("Training complete!")


# =========================
# PREDICTION
# =========================

print("\nEvaluating model...")

y_pred = model.predict(X_test)


# =========================
# METRICS
# =========================

accuracy = accuracy_score(
    y_test,
    y_pred
)

precision = precision_score(
    y_test,
    y_pred,
    zero_division=0
)

recall = recall_score(
    y_test,
    y_pred,
    zero_division=0
)

f1 = f1_score(
    y_test,
    y_pred,
    zero_division=0
)

cm = confusion_matrix(
    y_test,
    y_pred
)


# =========================
# RESULTS
# =========================

print("\n================================")
print("FINAL MODEL RESULTS")
print("================================")

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

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        y_pred,
        target_names=[
            "Closed Eyes",
            "Open Eyes"
        ],
        zero_division=0
    )
)

print("================================")