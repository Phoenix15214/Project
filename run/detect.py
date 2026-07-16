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
MIN_AREA = 10000
MAX_AREA = 70000
MIN_CHESS_AREA = 1000
MAX_CHESS_AREA = 10000
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

# 对图像进行预处理
def preprocess_frame(frame):
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    pink_mask = cv2.inRange(frame, (130, 40, 100), (160, 150, 255))
    black_mask = cv2.inRange(frame, (0, 0, 0), (180, 255, 160))
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    pink_mask = cv2.morphologyEx(pink_mask, cv2.MORPH_CLOSE, kernel)
    black_mask = cv2.morphologyEx(black_mask, cv2.MORPH_CLOSE, kernel)
    pink_frame = cv2.bitwise_and(white, white, mask=pink_mask)
    black_frame = cv2.bitwise_and(white, white, mask=black_mask)

    return pink_frame, black_frame

def find_contours(binary):
    contours, _ = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, None
    
    valid_contour_vertex = lb.Find_Poly(contours, shape=4, min_area=MIN_AREA, max_area=MAX_AREA, factor=0.1)

    # if area < MIN_AREA or area > MAX_AREA:
    #     return None, None

    return valid_contour_vertex if valid_contour_vertex is not None else None

# 实验性程序，记得当前只支持单个棋子
def find_chess(black_frame):
    contours, _ = cv2.findContours(black_frame, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, None

    for contour in contours:
        area = cv2.contourArea(contour)
        if MIN_CHESS_AREA < area < MAX_CHESS_AREA:
            M = cv2.moments(contour)
            if M["m00"] != 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
                return contour, (cX, cY)

    return None, None

def main(conn=None):
    # 显示FPS
    last_time = time.time()
    current_time = time.time()
    fps = 0
    frame_count = 0
    target_point = (640, 360)  # 目标点坐标，位于图像中心
    current_point = (640, 360)  # 当前点坐标，初始化为图像中心
    # 打开摄像头
    cap = open_camera()
    if cap is None:
        return
    ret, frame = cap.read()
    warped = frame
    if not ret:
        print("Failed to grab initial frame")
        cap.release()
        return
    phase = 0
    try:
        while True:
            # 获取图像
            _, frame = cap.read()
            frame = frame[CAMERA_HEIGHT//4:3*CAMERA_HEIGHT//4, CAMERA_WIDTH//4:3*CAMERA_WIDTH//4]
            frame = cv2.resize(frame, (CAMERA_WIDTH, CAMERA_HEIGHT), interpolation=cv2.INTER_LINEAR)
            pink_frame, black_frame = preprocess_frame(frame)
            # contours, _ = cv2.findContours(black_frame, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            # cv2.drawContours(frame, contours, -1, (0, 255, 0), 3)
            _, chess_point = find_chess(black_frame)
            valid_contour_vertex = find_contours(black_frame)
            if chess_point is not None:
                current_point = chess_point
                cv2.circle(frame, current_point, 5, (0, 0, 255), -1)
                cv2.putText(frame, f"Chess Point: {current_point}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            else:
                cv2.putText(frame, "Chess Point: Not Found", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            if valid_contour_vertex is not None and len(valid_contour_vertex) > 1:
                for vertex in valid_contour_vertex:
                    if vertex is None:
                        continue
                    try:
                        contour = np.array(vertex, dtype=np.int32).reshape(-1, 1, 2)
                        cv2.drawContours(frame, [contour], -1, (0, 255, 0), 3)
                    except Exception as e:
                        print(f"Error drawing contour: {e}")
                        continue
            pink_frame = cv2.resize(pink_frame, (640, 480))
            black_frame = cv2.resize(black_frame, (640, 480))
            frame = cv2.resize(frame, (640, 480))
            cv2.imshow("Original Frame", frame)
            cv2.imshow("Pink Frame", pink_frame)
            cv2.imshow("Black Frame", black_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    except KeyboardInterrupt:
        print("Interrupted by user")
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
