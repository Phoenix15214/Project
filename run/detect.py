import cv2
import numpy as np
import process_lib.image_lib as lb
import process_lib.control_lib as ctrl
from multiprocessing import shared_memory, Value, Pipe
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

# 预处理图像，返回纯白色图像
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
    line_end = (640, 100)
    valid_center = (0, 0)
    last_position = (0, 0)
    with lb.YOLODetector(method="rknn", model_path="/home/ubuntu/Project/Project/run/best.rknn", num_classes=1, conf_thresh=0.5, iou_thresh=0.5, imgsz=(256, 256), cores=2) as detector:
        # 显示FPS
        last_time = time.time()
        current_time = time.time()
        fps = 0
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
                else:
                    valid_center = last_position
                
                on_line_position, t = get_on_line_position(valid_center, line_start, line_end)

                cv2.circle(frame, on_line_position, 5, (0, 0, 255), -1)
                cv2.line(frame, line_start, line_end, (255, 255, 0), 2)

                if conn is not None:
                    send_message = [0, on_line_position[0], on_line_position[1]]
                    conn.send(send_message)

                frame = cv2.resize(frame, (640, 480))
                cv2.imshow("Camera Feed", frame)
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
