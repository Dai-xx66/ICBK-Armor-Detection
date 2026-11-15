from pathlib import Path
from collections import Counter

dataset = Path("/home/dx/Downloads/RMPlateDetection-main/data yolo format")

all_classes = Counter()

for split in ["train", "val", "test"]:
    image_dir = dataset / "images" / split
    label_dir = dataset / "labels" / split

    images = [
        p for p in image_dir.iterdir()
        if p.suffix.lower() in [".jpg", ".jpeg", ".png"]
    ]

    labels = list(label_dir.glob("*.txt"))
    image_names = {p.stem for p in images}
    label_names = {p.stem for p in labels}

    print(f"\n===== {split} =====")
    print("图片数量:", len(images))
    print("标签数量:", len(labels))
    print("缺少标签的图片:", len(image_names - label_names))
    print("找不到图片的标签:", len(label_names - image_names))

    for label in labels:
        for line in label.read_text().splitlines():
            parts = line.split()
            if parts:
                all_classes[parts[0]] += 1

print("\n数据集类别编号统计:", dict(all_classes))
print("检查结束")