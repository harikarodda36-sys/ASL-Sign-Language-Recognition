import os
import sys
import json
import random
import cv2
import numpy as np

sys.stdout.reconfigure(encoding='utf-8')

try:
    import tf_keras as keras
    load_model_func = keras.models.load_model
except Exception:
    try:
        import keras
        load_model_func = keras.models.load_model
    except Exception:
        import tensorflow as tf
        load_model_func = tf.keras.models.load_model

# ============================================================
# SETTINGS
# ============================================================

MODEL_PATH = os.path.join("models", "asl_mobilenetv2_best.keras")
CLASS_NAMES_PATH = os.path.join("models", "class_names.json")

TEST_DIR = os.path.join("asl_split", "test")
if not os.path.exists(TEST_DIR):
    TEST_DIR = "dataset"

IMAGE_SIZE = 224

# ============================================================
# LOAD CLASS NAMES & MODEL
# ============================================================

print("=================================================================")
print("ASL 34-CLASS MODEL INDEPENDENT TEST")
print("=================================================================")

if not os.path.exists(CLASS_NAMES_PATH):
    raise FileNotFoundError(f"class_names.json not found: {CLASS_NAMES_PATH}")

with open(CLASS_NAMES_PATH, "r", encoding="utf-8") as f:
    class_names = json.load(f)

NUM_CLASSES = len(class_names)
print(f"\nLoaded {NUM_CLASSES} class names.")

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")

print(f"Loading model: {MODEL_PATH} ...")
model = load_model_func(MODEL_PATH, compile=False)
print("[OK] Model loaded successfully!")

# ============================================================
# LOAD ALL TEST IMAGES (BATCH PROCESSING)
# ============================================================

print("\nLoading test images into memory...")
all_images = []
all_labels = []
first_sample_per_class = {}

for idx, class_name in enumerate(class_names):
    class_folder = os.path.join(TEST_DIR, class_name)
    if not os.path.exists(class_folder):
        continue

    image_files = sorted([f for f in os.listdir(class_folder) if f.lower().endswith((".jpg", ".png", ".jpeg"))])
    if not image_files:
        continue

    first_sample_per_class[class_name] = len(all_images)

    for f in image_files:
        img_path = os.path.join(class_folder, f)
        img = cv2.imread(img_path)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_resized = cv2.resize(img_rgb, (IMAGE_SIZE, IMAGE_SIZE))
        all_images.append(img_resized.astype(np.float32))
        all_labels.append(idx)

all_images = np.array(all_images)
all_labels = np.array(all_labels)

# Run fast batch prediction
print("Performing model inference...")
all_probs = model.predict(all_images, batch_size=64, verbose=0)
all_preds = np.argmax(all_probs, axis=1)

# ============================================================
# PER-CLASS REPRESENTATIVE RESULTS
# ============================================================

print("\n" + "=" * 65)
print("PER-CLASS SAMPLE PREDICTION SUMMARY")
print("=" * 65)

print(f"{'STATUS':<6} {'ACTUAL CLASS':<15} {'PREDICTED CLASS':<15} {'CONFIDENCE':<12} {'RESULT'}")
print("-" * 65)

correct = 0
total = 0

for class_name in class_names:
    if class_name not in first_sample_per_class:
        print(f"[FAIL] {class_name:<15} FOLDER NOT FOUND")
        continue

    sample_idx = first_sample_per_class[class_name]
    pred_idx = all_preds[sample_idx]
    pred_class = class_names[pred_idx]
    confidence = float(all_probs[sample_idx, pred_idx])

    is_correct = (pred_class == class_name)
    status_str = "[OK]" if is_correct else "[X]"
    if is_correct:
        correct += 1
    total += 1

    print(f"{status_str:<6} {class_name:<15} {pred_class:<15} {confidence * 100:6.2f}%      {'CORRECT' if is_correct else 'WRONG'}")

print("-" * 65)
print(f"Random Per-Class Accuracy : {correct}/{total} ({correct/total*100:.2f}%)")

# ============================================================
# OVERALL ACCURACY & CONFUSION MATRIX
# ============================================================

overall_accuracy = np.mean(all_preds == all_labels)

print("\n" + "=" * 65)
print("FINAL TEST RESULTS")
print("=" * 65)
print(f"Total Test Samples     : {len(all_labels)}")
print(f"Correct Predictions    : {np.sum(all_preds == all_labels)}")
print(f"Overall Test Accuracy  : {overall_accuracy * 100:.2f}%")
print("=" * 65)

# Save confusion matrix using pure numpy
cm = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=int)
for true_lbl, pred_lbl in zip(all_labels, all_preds):
    cm[true_lbl, pred_lbl] += 1

cm_path = "confusion_matrix.csv"
np.savetxt(cm_path, cm, delimiter=",", fmt="%d")
print(f"\nSaved confusion matrix to: {cm_path}")
print("Test completed successfully!")
