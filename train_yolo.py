from ultralytics import YOLO
from pathlib import Path

folder = Path(__file__).resolve().parent

#加载轻量级预训练模型
model = YOLO("yolo11n.yaml")

#开始 CPU 小规模训练
results = model.train(
    data=str(folder / "data.yaml"),
    epochs=1,
    imgsz=320,
    batch=4,
    device="cpu",
    workers=0,
    fraction=0.1,
    project=str(folder / "runs"),
    name="armor_experiment"
)

print("YOLO 训练完成")