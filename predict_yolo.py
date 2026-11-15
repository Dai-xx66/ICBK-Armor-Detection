from ultralytics import YOLO
from pathlib import Path

folder = Path(__file__).resolve().parent

#加载第五周训练的模型
model = YOLO(
    str(folder / "runs" / "armor_experiment" / "weights" / "best.pt")
)

#对装甲板图片进行预测
results = model.predict(
    
    source=str(folder / "images" / "armor_01.png"),
    device="cpu",
    imgsz=320,
    conf=0.01,
    save=True,
    project=str(folder / "runs"),
    name="armor_predict_lowconf"
)


print("YOLO 图片推理完成")
print("检测到的目标数量：", len(results[0].boxes))