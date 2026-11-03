import cv2
import os
import numpy as np

# ============================================================
# SETTINGS
# ============================================================

DATASET_DIR = "new_dataset"
OUTPUT_DIR = "sample_processed"

# We only test a few classes first
TEST_CLASSES = ["A", "B", "R", "Open Hand"]

# Number of images to process from each class
SAMPLE_COUNT = 5

OUTPUT_SIZE = 224


# ============================================================
# FIND GREEN RECTANGLE
# ============================================================

def find_green_box(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Bright green used by your collection program
    lower_green = np.array([40, 150, 150])
    upper_green = np.array([90, 255, 255])

    mask = cv2.inRange(hsv, lower_green, upper_green)

    # Find contours
    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    if not contours:
        return None

    # Look for the largest green contour
    contours = sorted(
        contours,
        key=cv2.contourArea,
        reverse=True
    )

    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)

        # The bounding box should be reasonably large
        if w > 50 and h > 50:
            return x, y, w, h

    return None


# ============================================================
# MAKE SQUARE IMAGE WITHOUT DISTORTION
# ============================================================

def make_square(image, size=224):

    h, w = image.shape[:2]

    # Create square canvas
    side = max(h, w)

    canvas = np.zeros(
        (side, side, 3),
        dtype=np.uint8
    )

    # Put image in center
    y_offset = (side - h) // 2
    x_offset = (side - w) // 2

    canvas[
        y_offset:y_offset + h,
        x_offset:x_offset + w
    ] = image

    # Resize
    return cv2.resize(
        canvas,
        (size, size),
        interpolation=cv2.INTER_AREA
    )


# ============================================================
# PROCESS ONE IMAGE
# ============================================================

def process_image(input_path, output_path):

    image = cv2.imread(input_path)

    if image is None:
        return False, "Could not read image"

    box = find_green_box(image)

    if box is None:
        return False, "Green bounding box not found"

    x, y, w, h = box

    # Remove the green border itself.
    # 4 pixels inward from each side.
    margin = 5

    x1 = max(0, x + margin)
    y1 = max(0, y + margin)
    x2 = min(image.shape[1], x + w - margin)
    y2 = min(image.shape[0], y + h - margin)

    crop = image[y1:y2, x1:x2]

    if crop.size == 0:
        return False, "Empty crop"

    # Keep proportions instead of stretching
    processed = make_square(crop, OUTPUT_SIZE)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    cv2.imwrite(output_path, processed)

    return True, "OK"


# ============================================================
# MAIN
# ============================================================

os.makedirs(OUTPUT_DIR, exist_ok=True)

success = 0
failed = 0

print("=" * 60)
print("ASL SAMPLE PROCESSING")
print("=" * 60)

for class_name in TEST_CLASSES:

    class_dir = os.path.join(DATASET_DIR, class_name)

    if not os.path.isdir(class_dir):
        print(f"\nWARNING: Class not found: {class_name}")
        continue

    output_class_dir = os.path.join(
        OUTPUT_DIR,
        class_name
    )

    os.makedirs(output_class_dir, exist_ok=True)

    files = [
        f for f in os.listdir(class_dir)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ]

    files = sorted(files)[:SAMPLE_COUNT]

    print(f"\n{class_name}: processing {len(files)} images")

    for filename in files:

        input_path = os.path.join(
            class_dir,
            filename
        )

        output_path = os.path.join(
            output_class_dir,
            filename
        )

        ok, message = process_image(
            input_path,
            output_path
        )

        if ok:
            success += 1
            print(f"  OK     {filename}")
        else:
            failed += 1
            print(f"  FAILED {filename} -> {message}")


print("\n" + "=" * 60)
print("RESULT")
print("=" * 60)

print(f"Successfully processed : {success}")
print(f"Failed                 : {failed}")

print("\nOpen this folder and inspect the images:")
print(os.path.abspath(OUTPUT_DIR))