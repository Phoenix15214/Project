import cv2
import numpy as np

# ---------- 全局参数（可通过滑动条调节） ----------
black_thresh = 30
max_corners = 100
quality_level = 0.01
min_distance = 10
CAMERA_HEIGHT = 720
CAMERA_WIDTH = 1280

def nothing(x):
    pass

# ---------- 打开摄像头 ----------
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("无法打开摄像头，请检查连接")
    exit()

cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
cap.set(cv2.CAP_PROP_FPS, 30)
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))  # 设置为 MJPG 编码，减少延迟

print("按 'q' 退出，滑动条实时调节参数")

def filter_lines(binary, min_area=1000):
    contours, _ = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    mask = np.zeros_like(binary)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > min_area:
            cv2.drawContours(mask, [cnt], -1, 255, thickness=cv2.FILLED)
    
    return mask

while True:
    ret, frame = cap.read()
    if not ret:
        print("读取帧失败")
        break

    # 转为灰度图
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, 9, 75, 75)
    binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 5, 2)
    binary = cv2.medianBlur(binary, 5)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
    binary = cv2.dilate(binary, kernel, iterations=5)

    binary = filter_lines(binary, min_area=3000)


    # ---------- 1. 黑色内容提取 ----------
    # _, black_mask = cv2.threshold(gray, black_thresh, 255, cv2.THRESH_BINARY_INV)
    # kernel = np.ones((3, 3), np.uint8)
    # black_mask_cleaned = cv2.morphologyEx(black_mask, cv2.MORPH_OPEN, kernel, iterations=1)
    # gray_masked = cv2.bitwise_and(gray, gray, mask=black_mask)

    # ---------- 2. 角点检测 (Shi-Tomasi) ----------
    corners = cv2.goodFeaturesToTrack(binary,
                                      maxCorners=20,
                                      qualityLevel=0.05,
                                      minDistance=20)

    # 绘制角点（修复 np.int0 问题）
    img_corners = frame.copy()
    if corners is not None:
        # 转换为整数坐标（使用 astype(np.int32) 代替 np.int0）
        corners = corners.astype(np.int32)
        for corner in corners:
            x, y = corner.ravel()
            cv2.circle(img_corners, (x, y), 5, (0, 0, 255), -1)

    # ---------- 3. 显示 ----------
    # frame = cv2.resize(frame, (640, 480))
    gray = cv2.resize(gray, (640, 480))
    img_corners = cv2.resize(img_corners, (640, 480))
    binary = cv2.resize(binary, (640, 480))
    # cv2.imshow('Original', frame)
    cv2.imshow('Gray', gray)
    # cv2.imshow('Black Mask (raw)', black_mask)
    # cv2.imshow('Black Mask (cleaned)', black_mask_cleaned)
    cv2.imshow('Corners Detected', img_corners)
    cv2.imshow('Binary (Adaptive Threshold)', binary)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()