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
MIN_AREA = 5000
MAX_AREA = 500000
MIN_LASER_AREA = 20
MAX_LASER_AREA = 5000
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
def find_contours(white_frame, min_area=MIN_AREA, max_area=MAX_AREA):
    contours, _ = cv2.findContours(white_frame, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    valid_contours = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area >= min_area and area <= max_area:
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

# 得到下一个目标点，用于激光PID控制
def get_target_pixel(route_vertices, phase, split):
    start_vertice = route_vertices[phase][0]
    end_vertice = route_vertices[(phase + 1) % 4][0]
    target_pixel = []
    targetx = int(start_vertice[0] + ((end_vertice[0] - start_vertice[0]) / 10) * split)
    targety = int(start_vertice[1] + ((end_vertice[1] - start_vertice[1]) / 10) * split)
    target_pixel.append(targetx)
    target_pixel.append(targety)
    return target_pixel

def _find_intersection(line1, line2):
    x1, y1, x2, y2, dx1, dy1, length1 = line1
    x3, y3, x4, y4, dx2, dy2, length2 = line2

    denom = dy2 * dx1 - dx2 * dy1
    if abs(denom) < 1e-6:
        return None

    ua = ((x4 - x3) * (y1 - y3) - (y4 - y3) * (x1 - x3)) / denom
    ub = ((x2 - x1) * (y1 - y3) - (y2 - y1) * (x1 - x3)) / denom

    eps = 1e-6
    return_x = int(x1 + ua * (x2 - x1))
    return_y = int(y1 + ua * (y2 - y1))
    if return_x < 0 or return_x >= 640 or return_y < 0 or return_y >= 480:
        return None

    return return_x, return_y

def get_pixel_online(route_vertices, current_pixel, phase):
    start_vertice = route_vertices[phase][0]
    end_vertice = route_vertices[(phase + 1) % 4][0]
    line1 = [start_vertice[0], start_vertice[1], end_vertice[0], end_vertice[1]]
    dx = line1[2] - line1[0]
    dy = line1[3] - line1[1]
    k = dy / dx if dx != 0 else float('inf')
    line2 = [current_pixel[0], current_pixel[1], current_pixel[0] + 1, current_pixel[1] + k]
    intersectionx, intersectiony = _find_intersection(line1, line2)
    direction_vector = np.array(end_vertice) - np.array(start_vertice)
    direction_vector = direction_vector / np.linalg.norm(direction_vector)  # Normalize
    step_size = 5  # Define how much to move in each iteration
    next_pixel = np.array([intersectionx, intersectiony]) + direction_vector * step_size
    move_vector = next_pixel - np.array(current_pixel)
    if np.linalg.norm(move_vector) < step_size:
        return (np.array(end_vertice) -np.array(current_pixel)).astype(int).tolist()
    else:
        return np.array(move_vector).astype(int).tolist()

def get_target_pixel_continuous(route_vertices, current_pixel, phase):
    start_vertice = route_vertices[phase][0]
    end_vertice = route_vertices[(phase + 1) % 4][0]
    direction_vector = np.array(end_vertice) - np.array(start_vertice)
    direction_vector = direction_vector / np.linalg.norm(direction_vector)  # Normalize
    step_size = 5  # Define how much to move in each iteration
    if np.linalg.norm(np.array(current_pixel) - np.array(end_vertice)) < step_size:
        return end_vertice.tolist()  # If close enough to the end, snap to it
    next_pixel = np.array(current_pixel) + direction_vector * step_size
    return next_pixel.astype(int).tolist()

def get_laser_point(img, last_red, last_green, threshold=10):
    enable_cover = False
    if np.linalg.norm(np.array(last_red) - np.array(last_green)) < threshold:
        enable_cover = True
    red = lb.Color_Extraction(img, color=lb.RED)
    green = lb.Color_Extraction(img, color=lb.GREEN)
    red_contours = find_contours(red, min_area=MIN_LASER_AREA, max_area=MAX_LASER_AREA)
    green_contours = find_contours(green, min_area=MIN_LASER_AREA, max_area=MAX_LASER_AREA)
    if red_contours is None or len(red_contours) == 0:
        red_center = last_red
    if green_contours is None or len(green_contours) == 0:
        green_center = last_green
    if enable_cover and len(red_contours) == 0 and len(green_contours) != 0:
        red_contours = green_contours
    elif enable_cover and len(green_contours) == 0 and len(red_contours) != 0:
        green_contours = red_contours
    red_center = lb.Get_Center_Point(red_contours, mode=lb.CENTER_MAX)
    green_center = lb.Get_Center_Point(green_contours, mode=lb.CENTER_MAX)
    return red_center, green_center

def main():
    print(len([]))
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
    split = 0
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
            if split >= 10:
                split = 1
                phase = (phase + 1) % 4
            else:
                split += 1
            target_pixel = get_target_pixel(route_vertices, phase=phase, split=split)
            # warped = warp(frame, warped, valid_vertices[0])
            cv2.circle(frame, (target_pixel[0], target_pixel[1]), 5, (255, 255, 255), -1)  # Draw target pixel
            cv2.drawContours(frame, valid_vertices, -1, (0, 255, 0), 2)
            cv2.drawContours(frame, [route_vertices], -1, (0, 0, 255), 2)
            cv2.imshow('Processed Frame', white_frame)
            cv2.imshow('Black Frame', black_frame)

            # 显示图像
            frame = cv2.resize(frame, (640, 360))  # Resize for better display
            # warped = cv2.resize(warped, (640, 360))  # Resize for better display
            # edges = cv2.resize(edges, (640, 360))  # Resize for better display
            # cv2.imshow('Edges', edges)
            # cv2.imshow('Warped Frame', warped)
            cv2.imshow('Original Frame', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()