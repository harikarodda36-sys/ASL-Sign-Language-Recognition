import os
import hashlib
from collections import defaultdict

DATASET_DIR = "new_dataset"

hashes = defaultdict(list)
total = 0

for class_name in sorted(os.listdir(DATASET_DIR)):
    class_path = os.path.join(DATASET_DIR, class_name)

    if not os.path.isdir(class_path):
        continue

    for filename in os.listdir(class_path):
        file_path = os.path.join(class_path, filename)

        if not os.path.isfile(file_path):
            continue

        total += 1

        with open(file_path, "rb") as f:
            file_hash = hashlib.md5(f.read()).hexdigest()

        hashes[file_hash].append(file_path)

duplicate_groups = {
    h: files for h, files in hashes.items()
    if len(files) > 1
}

duplicate_files = sum(len(files) - 1 for files in duplicate_groups.values())

print("=" * 50)
print("DUPLICATE IMAGE CHECK")
print("=" * 50)

print(f"Total images       : {total}")
print(f"Unique images      : {len(hashes)}")
print(f"Duplicate groups   : {len(duplicate_groups)}")
print(f"Duplicate copies   : {duplicate_files}")

if duplicate_groups:
    print("\nDuplicate groups found:\n")

    for i, files in enumerate(duplicate_groups.values(), 1):
        print(f"Group {i}:")
        for file in files:
            print("  ", file)
else:
    print("\nNo exact duplicate images found! ✅")