import os
import matplotlib.pyplot as plt
import numpy as np

# Create results folders
os.makedirs("results/graphs", exist_ok=True)

# =========================
# MODEL METRICS
# =========================

metrics = {
    "Accuracy": 97.90,
    "Precision": 96.50,
    "Recall": 99.40,
    "F1 Score": 97.93
}

# =========================
# METRICS GRAPH
# =========================

plt.figure(figsize=(8, 5))

plt.bar(
    metrics.keys(),
    metrics.values()
)

plt.ylim(0, 100)

plt.ylabel("Score (%)")
plt.xlabel("Evaluation Metric")

plt.title(
    "Performance of HOG-SVM Eye State Classifier"
)

for i, value in enumerate(metrics.values()):

    plt.text(
        i,
        value + 1,
        f"{value:.2f}%",
        ha="center"
    )

plt.tight_layout()

plt.savefig(
    "results/graphs/model_performance.png",
    dpi=300
)

plt.close()


# =========================
# CONFUSION MATRIX
# =========================

cm = np.array([
    [482, 18],
    [3, 497]
])

plt.figure(figsize=(6, 5))

plt.imshow(cm)

plt.title("Confusion Matrix")

plt.xlabel("Predicted Label")
plt.ylabel("Actual Label")

plt.xticks(
    [0, 1],
    ["Closed Eyes", "Open Eyes"]
)

plt.yticks(
    [0, 1],
    ["Closed Eyes", "Open Eyes"]
)

for i in range(2):

    for j in range(2):

        plt.text(
            j,
            i,
            cm[i, j],
            ha="center",
            va="center"
        )

plt.colorbar()

plt.tight_layout()

plt.savefig(
    "results/graphs/confusion_matrix.png",
    dpi=300
)

plt.close()


print("================================")
print("RESULTS GENERATED")
print("================================")
print("Saved:")
print("results/graphs/model_performance.png")
print("results/graphs/confusion_matrix.png")
print("================================")