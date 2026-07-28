import cv2
import numpy as np
import process_lib.image_lib as lb
import process_lib.control_lib as ctrl
from multiprocessing import shared_memory, Value, Pipe
import time
import math
import os

CAMERA_FPS = 30
CAMERA_WIDTH = 1280 # 1080p 1920*1080
CAMERA_HEIGHT = 720 # 1080p 1920*1080
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


def main(conn=None, stop_event=None):
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
