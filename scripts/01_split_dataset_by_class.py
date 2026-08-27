"""
Tách dataset gộp 2 lớp (player + shuttlecock) thành 2 dataset riêng, mỗi cái 1 lớp.

Chạy:
    python scripts/split_dataset_by_class.py
"""

import os
import shutil

SOURCE_DIR = "dataset"
SPLITS = ["train", "valid", "test"]

PLAYER_CLASS_ID = 0
SHUTTLECOCK_CLASS_ID = 1


def make_single_class_dataset(output_dir, keep_class_id, class_name):
    """Tạo dataset mới chỉ giữ 1 lớp, class_id đổi về 0."""
    for split in SPLITS:
        src_images = os.path.join(SOURCE_DIR, split, "images")
        src_labels = os.path.join(SOURCE_DIR, split, "labels")
        dst_images = os.path.join(output_dir, split, "images")
        dst_labels = os.path.join(output_dir, split, "labels")

        os.makedirs(dst_images, exist_ok=True)
        os.makedirs(dst_labels, exist_ok=True)

        for label_filename in os.listdir(src_labels):
            # Lọc dòng: chỉ giữ class_id khớp, đổi số về 0
            label_path = os.path.join(src_labels, label_filename)
            with open(label_path, "r") as f:
                lines = f.readlines()

            kept_lines = []
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                if parts[0] == str(keep_class_id):
                    parts[0] = "0"
                    kept_lines.append(" ".join(parts))

            # Ghi nhãn mới (kể cả khi rỗng)
            dst_label_path = os.path.join(dst_labels, label_filename)
            with open(dst_label_path, "w") as f:
                for line in kept_lines:
                    f.write(line + "\n")

            # Copy ảnh gốc tương ứng
            image_extension = ".jpg"  # đổi ".png" nếu ảnh bạn là png
            image_name = os.path.splitext(label_filename)[0] + image_extension
            src_image_path = os.path.join(src_images, image_name)
            dst_image_path = os.path.join(dst_images, image_name)

            if os.path.exists(src_image_path):
                shutil.copy(src_image_path, dst_image_path)
            else:
                print(f"Cảnh báo: không tìm thấy ảnh {src_image_path}")

        print(f"{split}: processed {len(os.listdir(src_labels))} label files")

    # Tạo data.yaml cho dataset mới (nc=1)
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


if __name__ == "__main__":
    print("Building player-only dataset...")
    make_single_class_dataset("dataset_player", PLAYER_CLASS_ID, "player")

    print("\nBuilding shuttlecock-only dataset...")
    make_single_class_dataset("dataset_shuttlecock", SHUTTLECOCK_CLASS_ID, "shuttlecock")

    print("\nDone. Check dataset_player/ and dataset_shuttlecock/ before training.")