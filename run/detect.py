import cv2
import numpy as np
import process_lib.image_lib as lb
import process_lib.control_lib as ctrl
from multiprocessing import shared_memory, Value
import time
import math
import os

CAMERA_FPS = 30
CAMERA_WIDTH = 1280 # 1080p 1920*1080
CAMERA_HEIGHT = 720 # 1080p 1920*1080
MIN_AREA = 500
MAX_AREA = 500000
white = np.full((CAMERA_HEIGHT, CAMERA_WIDTH), 255, dtype=np.uint8)

# 打开摄像头
def open_camera():
    try:
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FPS, CAMERA_FPS)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
        return cap
    except Exception as e:
        print(f"Error opening camera: {e}")
        raise RuntimeError("Failed to open camera.")
        return None

# 预处理图像，返回纯白色图像
def preprocess_frame(frame):
    img_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    white_mask = cv2.inRange(img_hsv, (0, 0, 150), (180, 30, 255))
    black_mask = cv2.inRange(img_hsv, (0, 0, 0), (180, 255, 100))
    black_frame = cv2.bitwise_and(white, white, mask=black_mask)
    white_frame = cv2.bitwise_and(white, white, mask=white_mask)
    return white_frame, black_frame

# 寻找轮廓
def find_contours(white_frame):
    contours, _ = cv2.findContours(white_frame, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    valid_contours = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area >= MIN_AREA and area <= MAX_AREA:
            valid_contours.append(contour)
    return valid_contours

# 得到轨迹顶点
def get_route_vertices(vertices):
    result_vertice = []
    first_point = vertices[0]
    second_point = vertices[1]
    for i in range(len(first_point)):
        resultx = first_point[i][0] + second_point[i][0]
        resulty = first_point[i][1] + second_point[i][1]
        resultx = resultx // 2
        resulty = resulty // 2
        result_vertice.append([resultx, resulty])
    result_vertice = np.array(result_vertice, dtype=np.int32).reshape((-1, 1, 2))
    return result_vertice

def vertice_to_box(vertices):
    boxes = []
    for vertice in vertices:
        box = vertice.reshape(-1, 2) # Reshape to (4, 2)
        box = np.int32(box)
        box = ctrl.Reorder_Vertex_Pole(box)
        boxes.append(box)
    return boxes

# 透视变换
def warp(frame, warped, vertice):
    rotating_box = cv2.minAreaRect(vertice)
    box = cv2.boxPoints(rotating_box)
    box = np.int32(box)
    box = ctrl.Reorder_Vertex(box)
    warp_temp = lb.Perspective_Transform(frame, box)
    warp_gray = cv2.cvtColor(warp_temp, cv2.COLOR_BGR2GRAY)
    mean_value = np.mean(warp_gray)
    if mean_value > 50:
        warped = warp_temp
    else:
        print(f"Mean value too low: {mean_value}, skipping this contour.")
    return warped

def main():
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
    try:
        while True:
            # 获取图像
            # _, frame = cap.read()
            frame = cv2.imread("black.jpg")
            frame = cv2.resize(frame, (CAMERA_WIDTH, CAMERA_HEIGHT))  # Resize for consistency
            # 预处理
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # 寻找轮廓(包含尺寸筛选)
            white_frame, black_frame = preprocess_frame(frame)
            contours = find_contours(black_frame)
            # 筛选轮廓
            valid_vertices = lb.Find_Poly(contours, shape=4, min_area=None, max_area=None, factor=0.1)
            valid_vertices = vertice_to_box(valid_vertices)
            route_vertices = get_route_vertices(valid_vertices)
            warped = warp(frame, warped, valid_vertices[0])
            cv2.drawContours(frame, valid_vertices, -1, (0, 255, 0), 2)
            cv2.drawContours(frame, [route_vertices], -1, (0, 0, 255), 2)
            cv2.imshow('Processed Frame', white_frame)
            cv2.imshow('Black Frame', black_frame)

            # 显示图像
            frame = cv2.resize(frame, (640, 360))  # Resize for better display
            warped = cv2.resize(warped, (640, 360))  # Resize for better display
            # edges = cv2.resize(edges, (640, 360))  # Resize for better display
            # cv2.imshow('Edges', edges)
            cv2.imshow('Warped Frame', warped)
            cv2.imshow('Original Frame', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()