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
        return cap
    except Exception as e:
        print(f"Error opening camera: {e}")
        raise RuntimeError("Failed to open camera.")
        return None

# 预处理图像，返回纯白色图像
def preprocess_frame(frame):
    img_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    red_mask1 = cv2.inRange(img_hsv, (0, 25, 50), (15, 255, 255))
    red_mask2 = cv2.inRange(img_hsv, (165, 25, 50), (180, 255, 255))
    red_frame = cv2.bitwise_and(white, white, mask=cv2.bitwise_or(red_mask1, red_mask2))
    red_mask = cv2.bitwise_or(red_mask1, red_mask2)
    img_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    black_mask = cv2.inRange(img_hsv, (0, 0, 0), (180, 255, 150))
    black_frame = cv2.bitwise_and(white, white, mask=black_mask)

    return red_frame, black_frame, img_gray

def find_contours(white_frame, min_area=None, max_area=None):
    # 查找轮廓
    contours, _ = cv2.findContours(white_frame, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    valid_contours = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if (min_area is None or area >= min_area) and (max_area is None or area <= max_area):
            valid_contours.append(contour)
    return valid_contours


def find_object(red_frame, min_area=None, max_area=None):
    # 查找红色区域的轮廓
    contours = find_contours(red_frame, min_area=min_area, max_area=max_area)
    if contours:
        centers = []
        for contour in contours:
            M = cv2.moments(contour)
            if M["m00"] != 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
                centers.append((cX, cY))
        return centers

    return None

def find_lines(black_frame):
    lines = cv2.HoughLinesP(black_frame, 1, np.pi / 180, threshold=80, minLineLength=50, maxLineGap=20)
    return lines

def find_poly(black_frame, min_area=None, max_area=None):
    contours = find_contours(black_frame, min_area=min_area, max_area=max_area)
    valid_contours = lb.Find_Poly(contours)
    return valid_contours

def main(conn=None):
    # 显示FPS
    last_time = time.time()
    current_time = time.time()
    fps = 0
    frame_count = 0
    last_red = (0, 0)
    last_green = (0, 0)
    # 打开摄像头
    cap = open_camera()
    if cap is None:
        return
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab initial frame")
        cap.release()
        return
    phase = 0
    try:
        while True:
            # 获取图像
            _, frame = cap.read()
            # 预处理
            red_frame, black_frame, gray= preprocess_frame(frame)
            gray = cv2.createCLAHE(clipLimit=8.0, tileGridSize=(8, 8)).apply(gray)
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            black_frame = cv2.dilate(black_frame, kernel, iterations=2)
            # edges = cv2.Canny(black_frame, 50, 150)
            # contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            poly = find_poly(black_frame, min_area=2000, max_area=500000)
            centers = find_object(red_frame, min_area=2000, max_area=500000)
            binary = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)[1]
            binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=3)
            # poly = find_poly(binary, min_area=1000, max_area=50000)


            if poly is not None:
                for p in poly:
                    cv2.polylines(frame, [p], True, (0, 255, 0), 2)
                    M = cv2.moments(p)
                    if M["m00"] != 0:
                        cX = int(M["m10"] / M["m00"])
                        cY = int(M["m01"] / M["m00"])
                        cv2.circle(frame, (cX, cY), 5, (255, 0, 0), -1)
                        cv2.putText(frame, f"({cX}, {cY})", (cX + 10, cY - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

            # if centers is not None:
            #     for center in centers:
            #         cv2.circle(frame, center, 5, (0, 255, 0), -1)
            #         cv2.putText(frame, f"({center[0]}, {center[1]})", (center[0] + 10, center[1] - 10),
            #                     cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

            # 显示图像
            frame = cv2.resize(frame, (640, 360))
            gray = cv2.resize(gray, (640, 360))
            # black_frame = cv2.resize(black_frame, (640, 360))
            binary = cv2.resize(binary, (640, 360))
            cv2.imshow('Original Frame', frame)
            cv2.imshow('Gray Frame', gray)
            # cv2.imshow('Black Frame', black_frame)
            cv2.imshow("Binary", binary)
            if cv2.waitKey(1) & 0xFF == ord('q'):
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
    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
