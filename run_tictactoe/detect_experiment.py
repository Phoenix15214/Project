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
MIN_AREA = 35000
MAX_AREA = 1000000 # 原阈值为70000
MIN_LASER_AREA = 0
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
    img_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    white_mask = cv2.inRange(img_hsv, (0, 0, 240), (180, 15, 255))
    black_mask = cv2.inRange(img_hsv, (0, 0, 0), (180, 255, 100))
    red_mask1 = cv2.inRange(img_hsv, (0, 25, 50), (15, 255, 255))
    red_mask2 = cv2.inRange(img_hsv, (165, 25, 50), (180, 255, 255))
    green_mask = cv2.inRange(img_hsv, (60, 50, 50), (80, 255, 255))
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
        centerx, centery = lb._cal_single_center(vertice)
        distance = np.linalg.norm(np.array([centerx, centery]) - np.array([FRAME_CENTER_X, FRAME_CENTER_Y]))
        if distance > 200:
            continue
        box = vertice.reshape(-1, 2) # Reshape to (4, 2)
        box = np.int32(box)
        box = ctrl.Reorder_Vertex_Pole(box)
        boxes.append(box)
    if len(boxes) == 1:
        boxes.append(boxes[0])
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
    x1, y1, x2, y2 = line1
    x3, y3, x4, y4 = line2
    dx1 = x2 - x1
    dy1 = y2 - y1
    dx2 = x4 - x3
    dy2 = y4 - y3

    denom = dy2 * dx1 - dx2 * dy1
    if abs(denom) < 1e-6:
        return None, None

    ua = ((x4 - x3) * (y1 - y3) - (y4 - y3) * (x1 - x3)) / denom
    ub = ((x2 - x1) * (y1 - y3) - (y2 - y1) * (x1 - x3)) / denom

    eps = 1e-6
    return_x = int(x1 + ua * (x2 - x1))
    return_y = int(y1 + ua * (y2 - y1))
    if return_x < 0 or return_x >= CAMERA_WIDTH or return_y < 0 or return_y >= CAMERA_HEIGHT:
        return None, None

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
    # 目标点方向向量
    direction_vector = np.array(end_vertice) - np.array(start_vertice)
    direction_vector = direction_vector / np.linalg.norm(direction_vector)
    # 移动步长
    step_size = 5
    # 得到目标像素,当没有算出交点时停留在当前像素
    if intersectionx is None or intersectiony is None:
        return (current_pixel[0], current_pixel[1]), ending
    next_pixel = np.array([intersectionx, intersectiony]) + direction_vector * step_size
    # 计算移动向量
    move_vector = next_pixel - np.array(current_pixel)
    if np.linalg.norm(move_vector) < step_size:
        ending = True
        return (np.array(end_vertice) -np.array(current_pixel)).astype(int).tolist(), ending
    elif np.linalg.norm(move_vector) > 100:
        return np.array([0, 0]).astype(int).tolist(), ending
    else:
        return np.array(move_vector).astype(int).tolist(), ending

def get_target_pixel_pole(laser_point, center_point, binary, step=0.05): # step为弧度制
    lx, ly = laser_point
    if lx == 0 and ly == 0:
        return (0, 0)
    cx, cy = center_point
    h, w = binary.shape
    theta = math.atan2(cy - ly, cx - lx)
    theta += step
    dx = math.cos(theta)
    dy = math.sin(theta)
    max_distance = int(math.hypot(w, h)) + 2
    for i in range(1, max_distance):
        x = int(cx - i * dx)
        y = int(cy - i * dy)
        if x < 0 or x >= w or y < 0 or y >= h:
            return (0, 0)
        if binary[y, x] > 0:
            return (x, y)
    return (0, 0)
    

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

def get_laser_point_via_white(white_frame, red_frame, green_frame, red, green, last_red, last_green, roi_start, roi_end):
    # 寻找并筛选轮廓
    probable_contours = find_contours(white_frame, min_area=MIN_LASER_AREA, max_area=MAX_LASER_AREA)
    probable_centers = []
    probable_rois = []
    probable_reds = []
    probable_greens = []

    # 得到轮廓中心
    for contour in probable_contours:
        contourx, contoury = lb._cal_single_center(contour)
        if contourx > 0 and contoury > 0:
            probable_centers.append((contourx, contoury))
    # 根据轮廓中心在红色和绿色图像中提取ROI，并计算非零像素数量，找出红色和绿色激光点
    for center in probable_centers:
        x, y = center
        roi_size = 40
        red_roi = red_frame[max(0, y - roi_size):min(white_frame.shape[0], y + roi_size),
                            max(0, x - roi_size):min(white_frame.shape[1], x + roi_size)]
        green_roi = green_frame[max(0, y - roi_size):min(white_frame.shape[0], y + roi_size),
                                max(0, x - roi_size):min(white_frame.shape[1], x + roi_size)]
        red_count = cv2.countNonZero(red_roi)
        # red_count = 100
        green_count = cv2.countNonZero(green_roi)
        if red_count > 0 and green_count > 0:
            probable_reds.append((center, red_count))
            probable_greens.append((center, green_count))
        elif red_count > 0 and green_count == 0:
            probable_reds.append((center, red_count))
        elif green_count > 0 and red_count == 0:
            probable_greens.append((center, green_count))
    
    # 根据非零像素数量排序，选择数量最多的作为激光点
    probable_reds.sort(key=lambda x: x[1], reverse=True)
    probable_greens.sort(key=lambda x: x[1], reverse=True)

    red_center = (probable_reds[0][0][0] + roi_start[0], probable_reds[0][0][1] + roi_start[1]) if probable_reds else (0, 0)
    green_center = (probable_greens[0][0][0] + roi_start[0], probable_greens[0][0][1] + roi_start[1]) if probable_greens else (0, 0)
    roi_size = 40
    if red_center[0] == 0 and red_center[1] == 0:
        roi_red = red[max(0, last_red[1] - roi_size):min(red.shape[0], last_red[1] + roi_size),
                        max(0, last_red[0] - roi_size):min(red.shape[1], last_red[0] + roi_size)]
        if cv2.countNonZero(roi_red) > 0:
            M = cv2.moments(roi_red)
            if M["m00"] != 0:
                cX = int(M["m10"] / M["m00"]) + max(0, last_red[0] - roi_size)
                cY = int(M["m01"] / M["m00"]) + max(0, last_red[1] - roi_size)
                red_center = (cX, cY)
    if green_center[0] == 0 and green_center[1] == 0:
        roi_green = green[max(0, last_green[1] - roi_size):min(green.shape[0], last_green[1] + roi_size),
                                max(0, last_green[0] - roi_size):min(green.shape[1], last_green[0] + roi_size)]
        if cv2.countNonZero(roi_green) > 0:
            M = cv2.moments(roi_green)
            if M["m00"] != 0:
                cX = int(M["m10"] / M["m00"]) + max(0, last_green[0] - roi_size)
                cY = int(M["m01"] / M["m00"]) + max(0, last_green[1] - roi_size)
                green_center = (cX, cY)
    return red_center, green_center

def prune_skeleton(binary_img, min_length=50):
    # 提取骨架轮廓
    skeleton_contours, _ = cv2.findContours(binary_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    pruned_skeleton = np.zeros_like(binary_img)
    for contour in skeleton_contours:
        if cv2.arcLength(contour, closed=False) >= min_length:
            cv2.drawContours(pruned_skeleton, [contour], -1, 255, 1)
    return pruned_skeleton

def get_roi_boundary(bounding_rect, roi_size=20):
    x, y, w, h = bounding_rect
    x1 = max(0, x - roi_size)
    y1 = max(0, y - roi_size)
    x2 = min(CAMERA_WIDTH, x + w + roi_size)
    y2 = min(CAMERA_HEIGHT, y + h + roi_size)
    return (x1, y1, x2, y2)

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
    warped = frame
    if not ret:
        print("Failed to grab initial frame")
        cap.release()
        return
    phase = 0
    try:
        while True:
            route_vertices = []
            target_pixel = (0, 0)
            bounding_rect = None
            roi_center = (0, 0)
            redx, redy = 0, 0
            greenx, greeny = 0, 0
            last_valid_vertices = []
            # 获取图像
            _, frame = cap.read()
            # frame = cv2.imread("black.jpg")
            # frame = cv2.resize(frame, (CAMERA_WIDTH, CAMERA_HEIGHT))
            # frame = frame[400:700, 500:800]
            # frame = cv2.resize(frame, (CAMERA_WIDTH, CAMERA_HEIGHT))
            # 预处理
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # 寻找轮廓(包含尺寸筛选)
            white_frame, black_frame, red_frame, green_frame= preprocess_frame(frame)
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            white_frame = cv2.morphologyEx(white_frame, cv2.MORPH_CLOSE, kernel, iterations=1)
            black_frame = cv2.morphologyEx(black_frame, cv2.MORPH_CLOSE, kernel, iterations=5)
            black_roi = black_frame
            red_roi = red_frame
            green_roi = green_frame
            black_contours = find_contours(black_frame)
            # 筛选轮廓
            valid_vertices = lb.Find_Poly(black_contours, shape=4, factor=0.1)
            # 将轮廓顶点转换为矩形框，此处包含了对轮廓中心点距离图像中心的筛选
            valid_vertices = vertice_to_box(valid_vertices)
            last_valid_vertices = valid_vertices
            red_center, green_center = get_laser_point_via_white(white_frame, red_frame, green_frame, red_frame, green_frame, last_red, last_green, (0, 0), (CAMERA_WIDTH, CAMERA_HEIGHT))
            redx, redy = red_center
            greenx, greeny = green_center
            last_red = red_center
            last_green = green_center
            if redx != 0 and redy != 0:
                cv2.circle(frame, (redx, redy), 5, (0, 0, 255), -1)
            # 进行数据发送
            if conn is not None and redx != 0 and redy != 0:
                msg = [0, redx, redy]
                conn.send(msg)

            # 显示图像
            # frame = cv2.resize(frame, (640, 360))  # Resize for better display
            # cv2.imshow('Original Frame', frame)
            # if cv2.waitKey(1) & 0xFF == ord('q'):
            #     break
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
