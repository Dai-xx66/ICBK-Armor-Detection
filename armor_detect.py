import cv2
import numpy as np
from pathlib import Path

#找到当前 Python 文件所在目录
folder = Path(__file__).resolve().parent

#读取装甲板图片
img = cv2.imread(str(folder / "images" / "armor_01.png"))

if img is None:
    raise FileNotFoundError("装甲板图片读取失败")

#BGR 转换为 HSV
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

#红色位于 HSV 色相范围两端
lower1 = np.array([0, 30, 150])
upper1 = np.array([10, 255, 255])

lower2 = np.array([170, 30, 150])
upper2 = np.array([179, 255, 255])

#分别提取两段红色
mask1 = cv2.inRange(hsv, lower1, upper1)
mask2 = cv2.inRange(hsv, lower2, upper2)

#合并两个掩膜
mask = cv2.bitwise_or(mask1, mask2)

#保存结果
cv2.imwrite(str(folder / "red_mask.png"), mask)

print("红色掩膜已保存：red_mask.png")
#提取红色区域的外部轮廓
contours, _ = cv2.findContours(
    mask,
    cv2.RETR_EXTERNAL,
    cv2.CHAIN_APPROX_SIMPLE
)

result = img.copy()
light_bars = []

#遍历每个轮廓
for contour in contours:

    # 计算轮廓面积
    area = cv2.contourArea(contour)

    if area < 10:
        continue

    # 计算最小外接旋转矩形
    rect = cv2.minAreaRect(contour)
    (cx, cy), (w, h), angle = rect

    if min(w, h) == 0:
        continue

    # 计算长宽比
    ratio = max(w, h) / min(w, h)

    # 初步筛选细长轮廓
    if ratio < 2.0:
        continue

    #将旋转矩形的长边方向换算成竖直偏差
    long_axis_angle = angle if w >= h else angle + 90

    vertical_error = abs((long_axis_angle % 180) - 90)

    #过滤明显不竖直的候选区域
    if vertical_error > 25:
        continue

    # 记录候选灯条
    light_bars.append(rect)

    # 绘制绿色旋转矩形
    box = cv2.boxPoints(rect)
    box = box.astype(int)

    cv2.drawContours(result, [box], 0, (0, 255, 0), 2)

#输出结果
print("检测到候选灯条数量：", len(light_bars))

cv2.imwrite(str(folder / "light_bars.png"), result)
#开始灯条配对
pairs = []

for i in range(len(light_bars)):
    for j in range(i + 1, len(light_bars)):

        rect1 = light_bars[i]
        rect2 = light_bars[j]

        (x1, y1), (w1, h1), a1 = rect1
        (x2, y2), (w2, h2), a2 = rect2

        # 两根灯条的长度
        length1 = max(w1, h1)
        length2 = max(w2, h2)

        # 两根灯条的中心距离
        dx = abs(x1 - x2)
        dy = abs(y1 - y2)

        # 条件1：高度差不能过大
        if dy > max(length1, length2) * 0.5:
            continue

        # 条件2：长度应接近
        if max(length1, length2) / min(length1, length2) > 1.5:
            continue

        # 条件3：水平间距合理
        avg_length = (length1 + length2) / 2

        if not (0.5 * avg_length < dx < 5 * avg_length):
            continue

        # 满足条件，记录这一对候选灯条
        pairs.append((rect1, rect2))

print("候选灯条配对数量：", len(pairs))

        #手动指定装甲板区域，坐标需要按原图调整
x1, y1 = 200, 60
x2, y2 = 750, 650

     #提取感兴趣区域
roi = img[y1:y2, x1:x2]

      #保存 ROI
cv2.imwrite(str(folder / "armor_roi.png"), roi)

print("ROI 提取完成")