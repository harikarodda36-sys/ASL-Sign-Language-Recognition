import os
import json
import numpy as np
import tensorflow as tf
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score
)

# ============================================================
# SETTINGS
# ============================================================

TEST_DIR = "asl_split/test"
MODEL_PATH = "models/asl_mobilenetv2_best.keras"
CLASS_NAMES_PATH = "models/class_names.json"

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 16

# ============================================================
# LOAD MODEL
# ============================================================

print("=" * 60)
print("ASL MODEL - FINAL TEST EVALUATION")
print("=" * 60)

print("\nLoading model...")

model = tf.keras.models.load_model(MODEL_PATH)

print("Model loaded successfully!")

# ============================================================
# LOAD CLASS NAMES
# ============================================================

with open(CLASS_NAMES_PATH, "r", encoding="utf-8") as f:
    class_names = json.load(f)

print("\nNumber of classes:", len(class_names))

# ============================================================
# LOAD TEST DATA
# ============================================================

print("\nLoading test dataset...")

test_ds = tf.keras.utils.image_dataset_from_directory(
    TEST_DIR,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="int",
    shuffle=False
)

print("\nTest images:", len(test_ds.file_paths))

# ============================================================
# PREDICTIONS
# ============================================================

print("\nGenerating predictions...")

y_true = []
y_pred = []

for images, labels in test_ds:

    predictions = model.predict(
        images,
        verbose=0
    )

    predicted_classes = np.argmax(
        predictions,
        axis=1
    )

    y_true.extend(labels.numpy())
    y_pred.extend(predicted_classes)

y_true = np.array(y_true)
y_pred = np.array(y_pred)

# ============================================================
# OVERALL ACCURACY
# ============================================================

accuracy = accuracy_score(
    y_true,
    y_pred
)

print("\n")
print("=" * 60)
print("FINAL TEST RESULT")
print("=" * 60)

print(f"\nTest Accuracy: {accuracy * 100:.2f}%")

# ============================================================
# CLASSIFICATION REPORT
# ============================================================

print("\n")
print("=" * 60)
print("CLASSIFICATION REPORT")
print("=" * 60)

report = classification_report(
    y_true,
    y_pred,
    target_names=class_names,
    digits=4,
    zero_division=0
)

print(report)

# ============================================================
# CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    y_true,
    y_pred
)

print("\n")
print("=" * 60)
print("CONFUSION MATRIX")
print("=" * 60)

print(cm)

# ============================================================
# SAVE RESULTS
# ============================================================

np.savetxt(
    "confusion_matrix.csv",
    cm,
    delimiter=",",
    fmt="%d"
)

print("\nConfusion matrix saved to:")
print("confusion_matrix.csv")

print("\nEvaluation complete!")