import os
from PIL import Image
from collections import Counter

DATASET_DIR = "new_dataset"

dimensions = Counter()
formats = Counter()
total = 0

for class_name in sorted(os.listdir(DATASET_DIR)):
    class_path = os.path.join(DATASET_DIR, class_name)

    if not os.path.isdir(class_path):
        continue

    for filename in os.listdir(class_path):
        file_path = os.path.join(class_path, filename)

        if not os.path.isfile(file_path):
            continue

        try:
            with Image.open(file_path) as img:
                dimensions[img.size] += 1
                formats[img.format] += 1
                total += 1

        except Exception:
            pass

print("=" * 50)
print("IMAGE DIMENSION & FORMAT CHECK")
print("=" * 50)

print(f"Total images: {total}")

print("\nDimensions:")
for size, count in dimensions.items():
    print(f"{size}: {count}")

print("\nFormats:")
for fmt, count in formats.items():
    print(f"{fmt}: {count}")