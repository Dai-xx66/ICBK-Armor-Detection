from ultralytics import YOLO
from pathlib import Path

folder = Path(__file__).resolve().parent

#加载第五周训练得到的模型
model = YOLO(
    str(folder / "runs" / "armor_experiment" / "weights" / "best.pt")
)

#读取公开项目中的比赛视频
video_path = Path(
    "/home/dx/Downloads/RMPlateDetection-main/vid.mp4"
)

if not video_path.exists():
    raise FileNotFoundError(f"找不到视频：{video_path}")

#使用 CPU 进行视频推理
results = model.predict(
    source=str(video_path),
    device="cpu",
    imgsz=320,
    conf=0.01,
    save=True,
    stream=True,
    project=str(folder / "runs"),
    name="armor_video_predict",
    exist_ok=True
)

frame_count = 0
detected_frames = 0

for result in results:
    frame_count += 1

    if len(result.boxes) > 0:
        detected_frames += 1

print("视频推理完成")
print("处理帧数：", frame_count)
print("存在候选框的帧数：", detected_frames)