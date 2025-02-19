import os
import json
from tqdm import tqdm


def convert_coco_to_yolo(coco_json_path, images_dir):
    labels_dir = os.path.join(images_dir, "labels")
    os.makedirs(labels_dir, exist_ok=True)

    with open(coco_json_path, "r") as f:
        coco_data = json.load(f)

    categories = {cat["id"]: cat["name"] for cat in coco_data.get("categories", [])}

    for img in tqdm(coco_data.get("images", []), desc=f"Processing {os.path.basename(images_dir)}"):
        img_id = img["id"]
        img_w, img_h = img["width"], img["height"]
        img_name = img["file_name"]

        label_file = os.path.join(labels_dir, f"{os.path.splitext(img_name)[0]}.txt")
        with open(label_file, "w") as f:
            for ann in coco_data.get("annotations", []):
                if ann["image_id"] == img_id:
                    category_id = ann["category_id"] - 1  # Chỉnh về index bắt đầu từ 0
                    bbox = ann["bbox"]

                    x_center = (bbox[0] + bbox[2] / 2) / img_w
                    y_center = (bbox[1] + bbox[3] / 2) / img_h
                    width = bbox[2] / img_w
                    height = bbox[3] / img_h

                    f.write(f"{category_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")


def create_yaml(output_dir, train_dir, val_dir, test_dir, class_file):
    with open(class_file, "r") as f:
        class_names = [line.strip() for line in f.readlines()]

    # 🔹 Sửa lỗi Python 3.10 (Không dùng f-string với `\`)
    yaml_content = "path: \"" + output_dir.replace("\\", "/") + "\"\n" + \
                   "train: \"" + train_dir.replace("\\", "/") + "\"\n" + \
                   "val: \"" + val_dir.replace("\\", "/") + "\"\n" + \
                   "test: \"" + test_dir.replace("\\", "/") + "\"\n\n" + \
                   "nc: " + str(len(class_names)) + "\n" + \
                   "names: " + str(class_names)

    yaml_path = os.path.join(output_dir, "data.yaml")
    with open(yaml_path, "w") as f:
        f.write(yaml_content)

    print(f"✅ File YAML đã được tạo: {yaml_path}")


def main():
    output_dir = r"D:\projectKHMT"
    data_dirs = {
        "train": r"D:\projectKHMT\data\train",
        "valid": r"D:\projectKHMT\data\valid",
        "test": r"D:\projectKHMT\data\test"
    }

    class_file = os.path.join(output_dir, "classes.txt")

    # 🔹 Lấy danh sách class từ file train
    json_path = os.path.join(data_dirs["train"], "_annotations.coco.json")
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"🚨 Không tìm thấy {json_path}!")

    with open(json_path, "r") as f:
        coco_data = json.load(f)
    categories = {cat["id"]: cat["name"] for cat in coco_data.get("categories", [])}

    with open(class_file, "w") as f:
        for _, class_name in categories.items():
            f.write(class_name + "\n")

    # 🔹 Chuyển đổi annotation cho train/valid/test
    for split, data_dir in data_dirs.items():
        json_path = os.path.join(data_dir, "_annotations.coco.json")
        if not os.path.exists(json_path):
            raise FileNotFoundError(f"🚨 Không tìm thấy {json_path}!")

        convert_coco_to_yolo(json_path, data_dir)

    # 🔹 Tạo file YAML
    create_yaml(output_dir, data_dirs["train"], data_dirs["valid"], data_dirs["test"], class_file)


if __name__ == "__main__":
    main()
