"""
================================================================================
FILE 01: split_dataset_by_class.py
================================================================================

HOW TO RUN THIS FILE (type this in your terminal):
    python scripts/01_split_dataset_by_class.py

WHAT THIS FILE DOES (super simple explanation):
    Imagine you have one box of photos. Each photo has TWO kinds of stickers
    on it: a "player" sticker and a "shuttlecock" sticker.
    This file makes TWO new boxes:
        - Box A: same photos, but only "player" stickers kept
        - Box B: same photos, but only "shuttlecock" stickers kept
    We do this because later we teach two SEPARATE robots (YOLO models):
    one robot only learns to find players, the other only learns to find
    the shuttlecock. Each robot gets better and faster if it only has to
    learn ONE thing instead of two things at once.

WHAT GOES IN, WHAT COMES OUT:
    IN:  dataset/            <- the original folder with photos + both stickers
    OUT: dataset_player/     <- new folder, player stickers only
    OUT: dataset_shuttlecock/<- new folder, shuttlecock stickers only

HOW THIS FILE CONNECTS TO OTHER FILES:
    This is the VERY FIRST file in the whole project.
    Its OUTPUT folders (dataset_player/, dataset_shuttlecock/) are the INPUT
    for the two training files: 02_train_player.py and 02_train_shuttlecock.py.

    01_split_dataset_by_class.py
              |
              v
    dataset_player/  +  dataset_shuttlecock/
              |
              v
    02_train_player.py   02_train_shuttlecock.py
================================================================================
"""

import os
import shutil

# Where the original mixed dataset lives (downloaded from Roboflow)
SOURCE_DIR = "dataset"

# A dataset is normally split into 3 folders: pictures to learn from (train),
# pictures to check progress during learning (valid), and pictures to test
# at the very end (test). We do the same split-by-class job on all 3.
SPLITS = ["train", "valid", "test"]

# In the ORIGINAL dataset, each sticker (called a "class") has a number.
# 0 = player, 1 = shuttlecock. These numbers come from how the dataset
# was labeled on Roboflow.
PLAYER_CLASS_ID = 0
SHUTTLECOCK_CLASS_ID = 1


def make_single_class_dataset(output_dir, keep_class_id, class_name):
    """
    SIMPLE EXPLANATION:
    This function is like a sorting machine. You tell it "keep only stickers
    number X" and it builds a brand new folder that only has those stickers,
    plus the matching photos.

    Every photo has a matching ".txt" label file that lists which stickers
    are on it (one line per sticker, first number = which class).
    We read that file, throw away lines that aren't our class, and change
    the kept class number to "0" (because the NEW dataset only has ONE
    class, so it must be numbered 0, 0, 0... not 1).
    """
    for split in SPLITS:
        # Where to read FROM (the big mixed dataset)
        src_images = os.path.join(SOURCE_DIR, split, "images")
        src_labels = os.path.join(SOURCE_DIR, split, "labels")
        # Where to write TO (the new single-class dataset)
        dst_images = os.path.join(output_dir, split, "images")
        dst_labels = os.path.join(output_dir, split, "labels")

        # Make sure the destination folders exist before we write into them
        os.makedirs(dst_images, exist_ok=True)
        os.makedirs(dst_labels, exist_ok=True)

        # Go through every label file (one per photo) in this split
        for label_filename in os.listdir(src_labels):
            # Step 1: read the original label file line by line
            label_path = os.path.join(src_labels, label_filename)
            with open(label_path, "r") as f:
                lines = f.readlines()

            # Step 2: keep only the lines that belong to our class,
            # and rename that class number to "0"
            kept_lines = []
            for line in lines:
                line = line.strip()
                if not line:
                    continue  # skip blank lines
                parts = line.split()
                if parts[0] == str(keep_class_id):
                    parts[0] = "0"  # new dataset has only 1 class -> must be "0"
                    kept_lines.append(" ".join(parts))

            # Step 3: write the new (filtered) label file.
            # Even if kept_lines is empty, we still write an empty file --
            # YOLO expects every image to have a label file, even a blank one.
            dst_label_path = os.path.join(dst_labels, label_filename)
            with open(dst_label_path, "w") as f:
                for line in kept_lines:
                    f.write(line + "\n")

            # Step 4: copy the matching photo over too (labels are useless
            # without their picture)
            image_extension = ".jpg"  # change to ".png" if your photos are png
            image_name = os.path.splitext(label_filename)[0] + image_extension
            src_image_path = os.path.join(src_images, image_name)
            dst_image_path = os.path.join(dst_images, image_name)

            if os.path.exists(src_image_path):
                shutil.copy(src_image_path, dst_image_path)
            else:
                print(f"Cảnh báo: không tìm thấy ảnh {src_image_path}")

        print(f"{split}: processed {len(os.listdir(src_labels))} label files")

    # Every YOLO dataset needs a small "data.yaml" file that tells YOLO:
    # - where the train/valid/test image folders are
    # - how many classes exist (nc = number of classes)
    # - what those classes are called
    # Since our new dataset only has ONE class, nc=1.
    yaml_content = (
        "train: ../train/images\n"
        "val: ../valid/images\n"
        "test: ../test/images\n"
        "\n"
        "nc: 1\n"
        f"names: ['{class_name}']\n"
    )
    with open(os.path.join(output_dir, "data.yaml"), "w") as f:
        f.write(yaml_content)


# ------------------------------------------------------------------------
# MAIN: this is what actually runs when you type "python 01_split_dataset_by_class.py"
# ------------------------------------------------------------------------
if __name__ == "__main__":
    print("Building player-only dataset...")
    make_single_class_dataset("dataset_player", PLAYER_CLASS_ID, "player")

    print("\nBuilding shuttlecock-only dataset...")
    make_single_class_dataset("dataset_shuttlecock", SHUTTLECOCK_CLASS_ID, "shuttlecock")

    print("\nDone. Check dataset_player/ and dataset_shuttlecock/ before training.")