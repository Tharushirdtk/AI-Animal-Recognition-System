import os, shutil, random

SOURCE_DIR = "Animals-10/raw-img"
DEST_DIR = "dataset"
TRAIN_RATIO = 0.8
random.seed(42)

for cls in os.listdir(SOURCE_DIR):
    cls_path = os.path.join(SOURCE_DIR, cls)
    if not os.path.isdir(cls_path):
        continue

    images = os.listdir(cls_path)
    random.shuffle(images)

    split = int(len(images) * TRAIN_RATIO)

    train_imgs = images[:split]
    test_imgs = images[split:]

    train_dir = os.path.join(DEST_DIR, "train", cls)
    test_dir = os.path.join(DEST_DIR, "test", cls)

    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    for img in train_imgs:
        shutil.copy(os.path.join(cls_path, img), os.path.join(train_dir, img))

    for img in test_imgs:
        shutil.copy(os.path.join(cls_path, img), os.path.join(test_dir, img))

    print(f"{cls}: {len(train_imgs)} train | {len(test_imgs)} test")

print("✅ Dataset split correctly")
