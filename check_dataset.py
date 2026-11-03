import os
from PIL import Image

DATASET_DIR = "new_dataset"

total = 0
bad = []

for class_name in sorted(os.listdir(DATASET_DIR)):
    class_path = os.path.join(DATASET_DIR, class_name)

    if not os.path.isdir(class_path):
        continue

    for filename in os.listdir(class_path):
        file_path = os.path.join(class_path, filename)

        if not os.path.isfile(file_path):
            continue

        total += 1

        try:
            with Image.open(file_path) as img:
                img.verify()
        except Exception:
            bad.append(file_path)

print("=" * 50)
print("DATASET VALIDATION")
print("=" * 50)

print(f"Total images : {total}")
print(f"Bad images   : {len(bad)}")

if bad:
    print("\nBad files:")
    for file in bad:
        print(file)
else:
    print("\nAll images are valid! ✅")