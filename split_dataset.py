import os
import shutil
import random

# ============================================================
# SETTINGS
# ============================================================

SOURCE_DIR = "new_dataset"
OUTPUT_DIR = "asl_split"

TRAIN_RATIO = 0.70
VALIDATION_RATIO = 0.15
TEST_RATIO = 0.15

RANDOM_SEED = 42


# ============================================================
# CHECK RATIOS
# ============================================================

if TRAIN_RATIO + VALIDATION_RATIO + TEST_RATIO != 1.0:
    raise ValueError("Train, validation and test ratios must add up to 1.0")


# ============================================================
# RANDOM SEED
# ============================================================

random.seed(RANDOM_SEED)


# ============================================================
# CREATE OUTPUT FOLDERS
# ============================================================

train_dir = os.path.join(OUTPUT_DIR, "train")
validation_dir = os.path.join(OUTPUT_DIR, "validation")
test_dir = os.path.join(OUTPUT_DIR, "test")

os.makedirs(train_dir, exist_ok=True)
os.makedirs(validation_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)


# ============================================================
# GET CLASSES
# ============================================================

classes = sorted([
    folder for folder in os.listdir(SOURCE_DIR)
    if os.path.isdir(os.path.join(SOURCE_DIR, folder))
])


print("=" * 60)
print("ASL DATASET SPLITTING")
print("=" * 60)

print("Total classes:", len(classes))
print("Random seed:", RANDOM_SEED)
print()


# ============================================================
# PROCESS EACH CLASS
# ============================================================

total_train = 0
total_validation = 0
total_test = 0

for class_name in classes:

    source_class_dir = os.path.join(
        SOURCE_DIR,
        class_name
    )

    # Get JPG images
    images = [
        file for file in os.listdir(source_class_dir)
        if file.lower().endswith(".jpg")
    ]

    # Shuffle images
    random.shuffle(images)

    total_images = len(images)

    # --------------------------------------------------------
    # CALCULATE SPLIT
    # --------------------------------------------------------

    train_count = int(total_images * TRAIN_RATIO)
    validation_count = int(total_images * VALIDATION_RATIO)

    test_count = total_images - train_count - validation_count

    train_images = images[:train_count]

    validation_images = images[
        train_count:
        train_count + validation_count
    ]

    test_images = images[
        train_count + validation_count:
    ]

    # --------------------------------------------------------
    # CREATE CLASS FOLDERS
    # --------------------------------------------------------

    train_class_dir = os.path.join(
        train_dir,
        class_name
    )

    validation_class_dir = os.path.join(
        validation_dir,
        class_name
    )

    test_class_dir = os.path.join(
        test_dir,
        class_name
    )

    os.makedirs(train_class_dir, exist_ok=True)
    os.makedirs(validation_class_dir, exist_ok=True)
    os.makedirs(test_class_dir, exist_ok=True)

    # --------------------------------------------------------
    # COPY TRAIN IMAGES
    # --------------------------------------------------------

    for image in train_images:

        src = os.path.join(
            source_class_dir,
            image
        )

        dst = os.path.join(
            train_class_dir,
            image
        )

        shutil.copy2(src, dst)

    # --------------------------------------------------------
    # COPY VALIDATION IMAGES
    # --------------------------------------------------------

    for image in validation_images:

        src = os.path.join(
            source_class_dir,
            image
        )

        dst = os.path.join(
            validation_class_dir,
            image
        )

        shutil.copy2(src, dst)

    # --------------------------------------------------------
    # COPY TEST IMAGES
    # --------------------------------------------------------

    for image in test_images:

        src = os.path.join(
            source_class_dir,
            image
        )

        dst = os.path.join(
            test_class_dir,
            image
        )

        shutil.copy2(src, dst)

    # --------------------------------------------------------
    # TOTALS
    # --------------------------------------------------------

    total_train += len(train_images)
    total_validation += len(validation_images)
    total_test += len(test_images)

    print(
        f"{class_name:<20}"
        f"Train: {len(train_images):<5}"
        f"Validation: {len(validation_images):<5}"
        f"Test: {len(test_images):<5}"
    )


# ============================================================
# FINAL SUMMARY
# ============================================================

print()
print("=" * 60)
print("SPLIT COMPLETE")
print("=" * 60)

print(f"Train images       : {total_train}")
print(f"Validation images  : {total_validation}")
print(f"Test images        : {total_test}")

print(
    f"Total images       : "
    f"{total_train + total_validation + total_test}"
)

print("=" * 60)