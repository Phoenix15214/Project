import cv2
import numpy as np

# ---------- 全局参数（可通过滑动条调节） ----------
black_thresh = 30
max_corners = 100
quality_level = 0.01
min_distance = 10

def nothing(x):
    pass

# ---------- 创建窗口和滑动条 ----------
cv2.namedWindow('Controls')
cv2.createTrackbar('Black Thresh', 'Controls', black_thresh, 255, nothing)
cv2.createTrackbar('Max Corners', 'Controls', max_corners, 500, nothing)
cv2.createTrackbar('Quality (x1000)', 'Controls', int(quality_level * 1000), 100, nothing)
cv2.createTrackbar('Min Distance', 'Controls', min_distance, 50, nothing)

# ---------- 打开摄像头 ----------
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("无法打开摄像头，请检查连接")
    exit()

print("按 'q' 退出，滑动条实时调节参数")

while True:
    ret, frame = cap.read()
    if not ret:
        print("读取帧失败")
        break

    # 获取当前滑动条值
    black_thresh = cv2.getTrackbarPos('Black Thresh', 'Controls')
    max_corners = cv2.getTrackbarPos('Max Corners', 'Controls')
    if max_corners < 1:
        max_corners = 1  # 最少检测1个角点
    quality_val = cv2.getTrackbarPos('Quality (x1000)', 'Controls')
    quality_level = max(quality_val / 1000.0, 0.001)  # 避免为0
    min_distance = cv2.getTrackbarPos('Min Distance', 'Controls')

    # 转为灰度图
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # ---------- 1. 黑色内容提取 ----------
    _, black_mask = cv2.threshold(gray, black_thresh, 255, cv2.THRESH_BINARY_INV)
    kernel = np.ones((3, 3), np.uint8)
    black_mask_cleaned = cv2.morphologyEx(black_mask, cv2.MORPH_OPEN, kernel, iterations=1)
    gray_masked = cv2.bitwise_and(gray, gray, mask=black_mask)

    # ---------- 2. 角点检测 (Shi-Tomasi) ----------
    corners = cv2.goodFeaturesToTrack(gray,
                                      maxCorners=max_corners,
                                      qualityLevel=quality_level,
                                      minDistance=min_distance)

    # 绘制角点（修复 np.int0 问题）
    img_corners = frame.copy()
    if corners is not None:
        # 转换为整数坐标（使用 astype(np.int32) 代替 np.int0）
        corners = corners.astype(np.int32)
        for corner in corners:
            x, y = corner.ravel()
            cv2.circle(img_corners, (x, y), 5, (0, 0, 255), -1)

    # ---------- 3. 显示 ----------
    cv2.imshow('Original', frame)
    cv2.imshow('Black Mask (raw)', black_mask)
    cv2.imshow('Black Mask (cleaned)', black_mask_cleaned)
    cv2.imshow('Corners Detected', img_corners)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()