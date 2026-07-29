import cv2
import numpy as np
import process_lib.image_lib as lb
import process_lib.control_lib as ctrl
from multiprocessing import shared_memory, Value, Pipe
from filterpy.kalman import KalmanFilter
import time
import math
import os

CAMERA_FPS = 120
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
FRAME_CENTER_X = CAMERA_WIDTH // 2
FRAME_CENTER_Y = CAMERA_HEIGHT // 2
white = np.full((CAMERA_HEIGHT, CAMERA_WIDTH), 255, dtype=np.uint8)

# 打开摄像头
def open_camera():
    try:
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FPS, CAMERA_FPS)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M','J','P','G'))
        cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 3)
        cap.set(cv2.CAP_PROP_EXPOSURE, 20)
        actual_auto_exp = cap.get(cv2.CAP_PROP_AUTO_EXPOSURE)
        actual_exp = cap.get(cv2.CAP_PROP_EXPOSURE)
        print(f"Camera settings: Auto Exposure={actual_auto_exp}, Exposure={actual_exp}")
        return cap
    except Exception as e:
        print(f"Error opening camera: {e}")
        raise RuntimeError("Failed to open camera.")
        return None

# 预处理图像，返回水管内壁的颜色
def preprocess_frame(frame):
    return frame

# 寻找轮廓
def find_contours(white_frame, min_area=None, max_area=None):
    contours, _ = cv2.findContours(white_frame, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    valid_contours = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if (min_area is None or area >= min_area) and (max_area is None or area <= max_area):
            valid_contours.append(contour)
    return valid_contours

def get_on_line_position(ball_center, line_start, line_end):
    x1, y1 = line_start
    x2, y2 = line_end
    bx, by = ball_center
    # 处理线段退化为点的情况
    if x1 == x2 and y1 == y2:
        return (x1, y1)
    # 线段向量
    dx = x2 - x1
    dy = y2 - y1
    # 计算投影参数 t
    # t = ((bx-x1)*dx + (by-y1)*dy) / (dx^2 + dy^2)
    numerator = (bx - x1) * dx + (by - y1) * dy
    denominator = dx * dx + dy * dy
    if denominator == 0:  # 线段长度为0
        return (x1, y1)
    t = numerator / denominator
    # 计算垂足坐标（直线上的投影）
    px = x1 + t * dx
    py = y1 + t * dy
    # 将垂足限制在线段范围内
    if t < 0:
        px, py = x1, y1
        t = 0
    elif t > 1:
        px, py = x2, y2
        t = 1
    return (int(px), int(py)), t
    

def main(conn=None, stop_event=None):
    line_start = (0, 100) # 未定,需要变化
    line_end = (640, 300)
    valid_center = (0, 0)
    last_position = (0, 0)
    last_offset_distance = 0.0
    last_speed = 0.0
    speed = 0.0
    last_send_time = 0.0
    
    # ================= 初始化卡尔曼滤波器 =================
    # 状态向量 x = [offset_distance, speed]^T
    kf = KalmanFilter(dim_x=2, dim_z=1)
    kf.x = np.array([[0.], [0.]])  # 初始状态: [位置, 速度]
    kf.F = np.array([[1., 0.01],   # 状态转移矩阵 (dt 初始设为 0.01)
                     [0., 1.]])
    kf.H = np.array([[1., 0.]])    # 观测矩阵 (只观测 offset_distance)
    kf.P *= 1000.                  # 初始状态协方差 (初始不确定性较大)
    kf.R = 15.0                    # 观测噪声协方差 (YOLO中心点约有 7 像素的抖动)
    kf.Q = np.array([[0.1, 0.0], 
                     [0.0, 5.0]])  # 过程噪声协方差
    # ======================================================

    with lb.YOLODetector(method="rknn", model_path="/home/ubuntu/Project/Project/run/best.rknn", num_classes=1, conf_thresh=0.5, iou_thresh=0.5, imgsz=(256, 256), cores=2) as detector:
        # 显示FPS
        last_time = time.time()
        last_speed_time = time.time()
        current_time = time.time()
        fps = 0
        lost_frame = 0
        frame_count = 0
        # 打开摄像头
        cap = open_camera()
        if cap is None:
            raise RuntimeError("Failed to open camera")
        ret, frame = cap.read()
        if not ret:
            cap.release()
            raise RuntimeError("Failed to grab initial frame")
        try:
            while stop_event is None or not stop_event.is_set():
                # 获取图像
                ret, frame = cap.read()
                if not ret:
                    raise RuntimeError("Failed to grab frame")
                
                boxes, scores, class_ids = detector.detect(frame)
                frame = detector.draw_boxes(frame, boxes, scores, class_ids)
                centers = []
                for box in boxes:
                    x1, y1, x2, y2 = box
                    center_x = (x1 + x2) // 2
                    center_y = (y1 + y2) // 2
                    centers.append((center_x, center_y))
                centers = sorted(centers, key=lambda c: c[1])  # 按y坐标排序
                if len(centers) > 0:
                    center = centers[0]  # 选择最上方的中心点
                    cv2.circle(frame, center, 5, (0, 255, 0), -1)
                    valid_center = center
                    last_position = valid_center
                    # [修改说明] 移除了原有的 last_speed_time = time.time() 
                    # 原因：它会导致下方计算 dt 时趋近于0，引发速度计算爆炸。现在统一由下方卡尔曼模块管理时间戳。
                else:
                    valid_center = last_position
                    lost_frame += 1

                on_line_position, t = get_on_line_position(valid_center, line_start, line_end)
                line_length = math.hypot(line_end[0] - line_start[0], line_end[1] - line_start[1])
                offset_distance = line_length * (t - 0.5)  # 偏移距离，正负表示在直线的哪一侧

                # ================= 卡尔曼滤波速度计算 (替换原有简单滤波) =================
                current_time_kf = time.time()
                dt = current_time_kf - last_speed_time if current_time_kf - last_speed_time > 0 else 0.01
                last_speed_time = current_time_kf  # 每帧统一更新一次时间戳
                
                # 动态更新时间步长
                kf.F[0, 1] = dt
                # 预测与更新
                kf.predict()
                kf.update(np.array([[offset_distance]]))
                
                # 获取滤波后的速度
                speed = float(kf.x[1, 0])
                # 更新 last_offset_distance 为滤波后的位置，保持状态一致性
                last_offset_distance = float(kf.x[0, 0])
                # =========================================================================

                # (保留你原有的后续缩放逻辑，完全不动)
                offset_distance = int(offset_distance) + 1000
                # speed = speed * 10
                speed = int(speed) + 1000
                last_speed = speed

                cv2.circle(frame, on_line_position, 5, (0, 0, 255), -1)
                cv2.line(frame, line_start, line_end, (255, 255, 0), 2)

                if conn is not None:
                    send_message = [0, offset_distance, speed]
                    now = time.time()
                    if now - last_send_time >= 0.005:
                        try:
                            conn.send(send_message)
                            last_send_time = now
                        except (BrokenPipeError, EOFError, OSError):
                            conn = None

                # frame = cv2.resize(frame, (640, 480))
                # cv2.imshow("Camera Feed", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    if stop_event is not None:
                        stop_event.set()
                    break
                frame_count += 1
                current_time = time.time()
                if current_time - last_time >= 1.0:
                    fps = frame_count / (current_time - last_time)
                    print(f"FPS: {fps:.2f}")
                    frame_count = 0
                    last_time = current_time
    
        except KeyboardInterrupt:
            print("Interrupted by user")
        except Exception as e:
            print(f"An error occurred: {e}")
            if stop_event is not None:
                stop_event.set()
            raise
        finally:
            cap.release()
            cv2.destroyAllWindows()

if __name__ == "__main__":
    main()