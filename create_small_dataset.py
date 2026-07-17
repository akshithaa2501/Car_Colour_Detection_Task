import os
import shutil

SOURCE = "."
DEST = "small_dataset"

TRAIN_LIMIT = 500
VAL_LIMIT = 100

def copy_split(split, limit):
    img_src = os.path.join(SOURCE, split, "images")
    lbl_src = os.path.join(SOURCE, split, "labels")

    img_dst = os.path.join(DEST, split, "images")
    lbl_dst = os.path.join(DEST, split, "labels")

    os.makedirs(img_dst, exist_ok=True)
    os.makedirs(lbl_dst, exist_ok=True)

    if not os.path.exists(img_src):
        print(f"Folder not found: {img_src}")
        return

    images = sorted(os.listdir(img_src))

    print(f"{split}: Found {len(images)} images")

    count = 0

    for image in images[:limit]:
        shutil.copy2(
            os.path.join(img_src, image),
            os.path.join(img_dst, image)
        )

        label = os.path.splitext(image)[0] + ".txt"

        label_file = os.path.join(lbl_src, label)

        if os.path.exists(label_file):
            shutil.copy2(
                label_file,
                os.path.join(lbl_dst, label)
            )

        count += 1

    print(f"{split}: Copied {count} images")


copy_split("train", TRAIN_LIMIT)
copy_split("val", VAL_LIMIT)

print("\nSmall dataset created successfully!")