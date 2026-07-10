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
MIN_LASER_AREA = 50
MAX_LASER_AREA = 5000
white = np.full((CAMERA_HEIGHT, CAMERA_WIDTH), 255, dtype=np.uint8)

# 打开摄像头
def open_camera():
    try:
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FPS, CAMERA_FPS)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M','J','P','G'))
        cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)
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
    img_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    white_mask = cv2.inRange(img_hsv, (0, 0, 150), (180, 30, 255))
    black_mask = cv2.inRange(img_hsv, (0, 0, 0), (180, 255, 100))
    red_mask1 = cv2.inRange(img_hsv, (0, 43, 46), (10, 255, 255))
    red_mask2 = cv2.inRange(img_hsv, (156, 43, 46), (180, 255, 255))
    green_mask = cv2.inRange(img_hsv, (35, 43, 46), (85, 255, 255))
    red_mask = cv2.bitwise_or(red_mask1, red_mask2)
    red_frame = cv2.bitwise_and(white, white, mask=red_mask)
    green_frame = cv2.bitwise_and(white, white, mask=green_mask)
    black_frame = cv2.bitwise_and(white, white, mask=black_mask)
    white_frame = cv2.bitwise_and(white, white, mask=white_mask)
    return white_frame, black_frame, red_frame, green_frame

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
    # 结束标志位，若距离足够近则进入下一个phase
    ending = False
    # 得到起点和终点
    start_vertice = route_vertices[phase][0]
    end_vertice = route_vertices[(phase + 1) % 4][0]
    # 计算直线方程
    line1 = [start_vertice[0], start_vertice[1], end_vertice[0], end_vertice[1]]
    dx = line1[2] - line1[0]
    dy = line1[3] - line1[1]
    k = dy / dx if dx != 0 else float('inf')
    # 根据当前像素点和直线方程计算交点
    if k == float('inf'):
        line2 = [current_pixel[0], current_pixel[1], current_pixel[0], current_pixel[1] - 1]
    elif k == 0:
        line2 = [current_pixel[0], current_pixel[1], current_pixel[0] + 1, current_pixel[1]]
    else:
        line2 = [current_pixel[0], current_pixel[1], current_pixel[0] + 1, current_pixel[1] - (1 / k)]
    intersectionx, intersectiony = _find_intersection(line1, line2)
    direction_vector = np.array(end_vertice) - np.array(start_vertice)
    direction_vector = direction_vector / np.linalg.norm(direction_vector)  # Normalize
    step_size = 5  # Define how much to move in each iteration
    # 得到目标像素
    next_pixel = np.array([intersectionx, intersectiony]) + direction_vector * step_size
    # 计算移动向量
    move_vector = next_pixel - np.array(current_pixel)
    if np.linalg.norm(move_vector) < step_size:
        ending = True
        return (np.array(end_vertice) -np.array(current_pixel)).astype(int).tolist(), ending
    else:
        return np.array(move_vector).astype(int).tolist(), ending

def get_target_pixel_continuous(route_vertices, current_pixel, phase):
    # 结束标志位，若距离足够近则进入下一个phase
    ending = False
    # 得到起点和终点
    start_vertice = route_vertices[phase][0]
    end_vertice = route_vertices[(phase + 1) % 4][0]
    # 计算方向向量并归一化
    direction_vector = np.array(end_vertice) - np.array(start_vertice)
    direction_vector = direction_vector / np.linalg.norm(direction_vector)
    # 移动步长
    step_size = 5
    if np.linalg.norm(np.array(current_pixel) - np.array(end_vertice)) < step_size:
        ending = True
        return end_vertice.tolist(), ending  # If close enough to the end, snap to it
    next_pixel = np.array(current_pixel) + direction_vector * step_size
    return next_pixel.astype(int).tolist(), ending

def get_laser_point(binary_img):
    valid_contours = []
    contours = find_contours(binary_img, min_area=MIN_LASER_AREA, max_area=MAX_LASER_AREA)
    if contours is None or len(contours) == 0:
        return None, None
    for contour in contours:
        (x, y), radius = cv2.minEnclosingCircle(contour)
        circle_area = np.pi * radius * radius
        if circle_area != 0 and abs((circle_area - circle_area)) / circle_area < 0.2:
            valid_contours.append(contour)
    if len(valid_contours) == 0:
        return None, None
    cx, cy = lb.Get_Center_Point(valid_contours, mode=lb.CENTER_MAX)
    return cx, cy

def get_laser_point_simultaneous(img, last_red, last_green, threshold=10):
    # 允许覆盖标志位，防止激光点覆盖时检测不到
    enable_cover = False
    if np.linalg.norm(np.array(last_red) - np.array(last_green)) < threshold:
        enable_cover = True
    # 提取红色和绿色激光点
    red = lb.Color_Extraction(img, color=lb.RED)
    green = lb.Color_Extraction(img, color=lb.GREEN)
    red_contours = find_contours(red, min_area=MIN_LASER_AREA, max_area=MAX_LASER_AREA)
    green_contours = find_contours(green, min_area=MIN_LASER_AREA, max_area=MAX_LASER_AREA)
    # 如果没有检测到激光点，则使用上一次的激光点位置
    if red_contours is None or len(red_contours) == 0:
        red_center = last_red
    if green_contours is None or len(green_contours) == 0:
        green_center = last_green
    # 如果激光点被覆盖，则使用另一种颜色的激光点位置
    if enable_cover and len(red_contours) == 0 and len(green_contours) != 0:
        red_contours = green_contours
    elif enable_cover and len(green_contours) == 0 and len(red_contours) != 0:
        green_contours = red_contours
    red_center = lb.Get_Center_Point(red_contours, mode=lb.CENTER_MAX)
    green_center = lb.Get_Center_Point(green_contours, mode=lb.CENTER_MAX)
    return red_center, green_center

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
    phase = 0
    split = 0
    try:
        while True:
            route_vertices = []
            # 获取图像
            _, frame = cap.read()
            # frame = cv2.imread("black.jpg")
            # frame = cv2.resize(frame, (CAMERA_WIDTH, CAMERA_HEIGHT))  # Resize for consistency
            # 预处理
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # 寻找轮廓(包含尺寸筛选)
            white_frame, black_frame, red_frame, green_frame = preprocess_frame(frame)
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            red_frame = cv2.morphologyEx(red_frame, cv2.MORPH_OPEN, kernel, iterations=1)
            red_frame = cv2.morphologyEx(red_frame, cv2.MORPH_CLOSE, kernel, iterations=1)
            redx, redy = get_laser_point(red_frame)
            black_contours = find_contours(black_frame)
            # 筛选轮廓
            valid_vertices = lb.Find_Poly(black_contours, shape=4, min_area=None, max_area=None, factor=0.1)
            # print(len(valid_vertices))
            valid_vertices = vertice_to_box(valid_vertices)
            # print(len(valid_vertices))
            if len(valid_vertices) > 1:
                route_vertices = get_route_vertices(valid_vertices)
                if split >= 10:
                    split = 1
                    phase = (phase + 1) % 4
                else:
                    split += 1
            if len(route_vertices) > 0:
                target_pixel = get_target_pixel(route_vertices, phase=phase, split=split)
                # warped = warp(frame, warped, valid_vertices[0])
                cv2.circle(frame, (target_pixel[0], target_pixel[1]), 5, (255, 255, 255), -1)  # Draw target pixel
                cv2.drawContours(frame, [route_vertices], -1, (0, 0, 255), 2)
            cv2.drawContours(frame, valid_vertices, -1, (0, 255, 0), 2)
            if redx is not None and redy is not None:
                cv2.circle(frame, (redx, redy), 5, (0, 255, 0), -1)  # Draw red laser point
            # cv2.imshow('Processed Frame', white_frame)
            # cv2.imshow('Black Frame', black_frame)
            red_frame = cv2.resize(red_frame, (640, 360))  # Resize for better display
            green_frame = cv2.resize(green_frame, (640, 360))  # Resize for better display
            cv2.imshow('Red Frame', red_frame)
            cv2.imshow('Green Frame', green_frame)

            # 显示图像
            frame = cv2.resize(frame, (640, 360))  # Resize for better display
            cv2.imshow('Original Frame', frame)
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
