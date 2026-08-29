import os
import cv2
import mediapipe as mp
import numpy as np

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

# =========================
# DATASET
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

SAMPLE_SIZE = 500

EAR_THRESHOLD = 0.22

# =========================
# MEDIAPIPE
# =========================

mp_face_mesh = mp.solutions.face_mesh

face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=True,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5
)

# =========================
# EYE LANDMARKS
# =========================

LEFT_EYE = [362, 385, 387, 263, 373, 380]

RIGHT_EYE = [33, 160, 158, 133, 153, 144]


# =========================
# EAR FUNCTION
# =========================

def calculate_ear(landmarks, eye_points, width, height):

    points = []

    for point in eye_points:

        x = int(landmarks[point].x * width)
        y = int(landmarks[point].y * height)

        points.append([x, y])

    points = np.array(points)

    vertical_1 = np.linalg.norm(
        points[1] - points[5]
    )

    vertical_2 = np.linalg.norm(
        points[2] - points[4]
    )

    horizontal = np.linalg.norm(
        points[0] - points[3]
    )

    return (
        vertical_1 + vertical_2
    ) / (2.0 * horizontal)


# =========================
# GET DATASET
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

closed_images = closed_images[:SAMPLE_SIZE]

open_images = open_images[:SAMPLE_SIZE]

image_files = []

for image in closed_images:

    image_files.append((image, 0))

for image in open_images:

    image_files.append((image, 1))


print("================================")
print("ACTUAL MODEL EVALUATION")
print("================================")
print(f"Closed images: {len(closed_images)}")
print(f"Open images: {len(open_images)}")
print(f"Total images: {len(image_files)}")
print("================================")


# =========================
# PREDICTIONS
# =========================

actual_labels = []

predicted_labels = []

processed = 0

face_detected = 0


# =========================
# PROCESS IMAGES
# =========================

for image_path, actual_label in image_files:

    image = cv2.imread(image_path)

    if image is None:
        continue

    height, width, _ = image.shape

    rgb_image = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB
    )

    results = face_mesh.process(
        rgb_image
    )

    if not results.multi_face_landmarks:

        continue

    face_detected += 1

    landmarks = (
        results.multi_face_landmarks[0].landmark
    )

    left_ear = calculate_ear(
        landmarks,
        LEFT_EYE,
        width,
        height
    )

    right_ear = calculate_ear(
        landmarks,
        RIGHT_EYE,
        width,
        height
    )

    ear = (
        left_ear + right_ear
    ) / 2.0

    # =========================
    # MODEL PREDICTION
    # =========================

    if ear < EAR_THRESHOLD:

        predicted_label = 0

    else:

        predicted_label = 1

    actual_labels.append(actual_label)

    predicted_labels.append(predicted_label)

    processed += 1

    if processed % 50 == 0:

        print(
            f"Processed: {processed}"
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
# RESULTS
# =========================

print("\n================================")
print("ACTUAL MODEL RESULTS")
print("================================")

print(
    f"Images successfully processed: "
    f"{processed}"
)

print(
    f"Faces detected: {face_detected}"
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

print("================================")


face_mesh.close()