import os
import json
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    ReduceLROnPlateau
)

# ============================================================
# SETTINGS
# ============================================================

TRAIN_DIR = "asl_split/train"
VAL_DIR = "asl_split/validation"

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 16
INITIAL_EPOCHS = 15

MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "asl_mobilenetv2_best.keras")
CLASS_NAMES_PATH = os.path.join(MODEL_DIR, "class_names.json")

os.makedirs(MODEL_DIR, exist_ok=True)

# ============================================================
# SYSTEM INFORMATION
# ============================================================

print("=" * 60)
print("ASL SIGN LANGUAGE RECOGNITION")
print("MODEL TRAINING")
print("=" * 60)

print("TensorFlow version:", tf.__version__)
print("Available devices:", tf.config.list_physical_devices())

# ============================================================
# LOAD TRAINING DATA
# ============================================================

print("\nLoading training dataset...")

train_ds = tf.keras.utils.image_dataset_from_directory(
    TRAIN_DIR,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="categorical",
    shuffle=True,
    seed=42
)

# ============================================================
# LOAD VALIDATION DATA
# ============================================================

print("\nLoading validation dataset...")

val_ds = tf.keras.utils.image_dataset_from_directory(
    VAL_DIR,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="categorical",
    shuffle=False
)

# ============================================================
# CLASS NAMES
# ============================================================

class_names = train_ds.class_names
NUM_CLASSES = len(class_names)

print("\nNumber of classes:", NUM_CLASSES)

print("\nClasses:")
for i, name in enumerate(class_names):
    print(f"{i:2d} -> {name}")

# Save class names for later live prediction
with open(CLASS_NAMES_PATH, "w", encoding="utf-8") as f:
    json.dump(class_names, f, indent=4)

print("\nClass names saved to:", CLASS_NAMES_PATH)

# ============================================================
# PERFORMANCE
# ============================================================

AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.prefetch(AUTOTUNE)
val_ds = val_ds.prefetch(AUTOTUNE)

# ============================================================
# DATA AUGMENTATION
# ============================================================

# IMPORTANT:
# We DO NOT use horizontal flipping.
#
# Point Left and Point Right are different classes.
# Flipping the image could change the meaning of the sign.

data_augmentation = tf.keras.Sequential([
    layers.RandomRotation(0.05),
    layers.RandomTranslation(
        height_factor=0.08,
        width_factor=0.08
    ),
    layers.RandomZoom(0.10),
    layers.RandomContrast(0.10),
], name="data_augmentation")

# ============================================================
# MOBILE NET V2
# ============================================================

print("\nLoading MobileNetV2 pretrained model...")

base_model = MobileNetV2(
    weights="imagenet",
    include_top=False,
    input_shape=(224, 224, 3)
)

# Freeze pretrained layers initially
base_model.trainable = False

# ============================================================
# BUILD MODEL
# ============================================================

inputs = layers.Input(shape=(224, 224, 3))

x = data_augmentation(inputs)

# MobileNetV2 preprocessing
x = preprocess_input(x)

x = base_model(x, training=False)

x = layers.GlobalAveragePooling2D()(x)

x = layers.Dropout(0.30)(x)

outputs = layers.Dense(
    NUM_CLASSES,
    activation="softmax"
)(x)

model = models.Model(
    inputs,
    outputs,
    name="ASL_MobileNetV2"
)

# ============================================================
# COMPILE
# ============================================================

model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=0.0001
    ),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

# ============================================================
# MODEL SUMMARY
# ============================================================

print("\nModel summary:\n")

model.summary()

# ============================================================
# CALLBACKS
# ============================================================

early_stopping = EarlyStopping(
    monitor="val_accuracy",
    patience=4,
    mode="max",
    restore_best_weights=True,
    verbose=1
)

checkpoint = ModelCheckpoint(
    MODEL_PATH,
    monitor="val_accuracy",
    mode="max",
    save_best_only=True,
    verbose=1
)

reduce_lr = ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.5,
    patience=2,
    min_lr=1e-6,
    verbose=1
)

# ============================================================
# TRAIN
# ============================================================

print("\n")
print("=" * 60)
print("STARTING TRAINING")
print("=" * 60)

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=INITIAL_EPOCHS,
    callbacks=[
        early_stopping,
        checkpoint,
        reduce_lr
    ]
)

# ============================================================
# SAVE FINAL MODEL
# ============================================================

final_model_path = os.path.join(
    MODEL_DIR,
    "asl_mobilenetv2_final.keras"
)

model.save(final_model_path)

print("\n")
print("=" * 60)
print("TRAINING COMPLETE")
print("=" * 60)

print("Best model:")
print(MODEL_PATH)

print("\nFinal model:")
print(final_model_path)

print("\nClass names:")
print(CLASS_NAMES_PATH)

print("=" * 60)