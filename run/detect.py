import cv2
import numpy as np
import process_lib.image_lib as lb
import process_lib.control_lib as ctrl
from multiprocessing import shared_memory, Value, Pipe
from typing import List, Tuple, Optional
from scipy.interpolate import splprep, splev
import heapq
import time
import math
import os

CAMERA_FPS = 30
CAMERA_WIDTH = 1280 # 1080p 1920*1080
CAMERA_HEIGHT = 720 # 1080p 1920*1080
FRAME_CENTER_X = CAMERA_WIDTH // 2
FRAME_CENTER_Y = CAMERA_HEIGHT // 2
TARGET_X, TARGET_Y = 640, 560
AVG_SLOPE_FILTER_THRESHOLD = 1
white = np.full((CAMERA_HEIGHT, CAMERA_WIDTH), 255, dtype=np.uint8)
frame_share = ctrl.MemoryShare(name='shared_frame', shape=(CAMERA_HEIGHT,CAMERA_WIDTH,3), dtype='uint8')
frame_share2 = ctrl.MemoryShare(name='shared_frame2', shape=(CAMERA_HEIGHT,CAMERA_WIDTH,3), dtype='uint8')

# 打开摄像头
def open_camera(camera_index=0):
    try:
        cap = cv2.VideoCapture(camera_index)
        cap.set(cv2.CAP_PROP_FPS, CAMERA_FPS)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M','J','P','G'))
        cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)
        cap.set(cv2.CAP_PROP_EXPOSURE, 30)
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
    red_mask1 = cv2.inRange(img_hsv, (0, 25, 50), (15, 255, 255))
    red_mask2 = cv2.inRange(img_hsv, (165, 25, 50), (180, 255, 255))
    red_frame = cv2.bitwise_and(white, white, mask=cv2.bitwise_or(red_mask1, red_mask2))
    pink_mask = cv2.inRange(img_hsv, (130, 20, 100), (180, 150, 255))
    black_mask = cv2.inRange(img_hsv, (0, 0, 0), (180, 255, 150))
    black_frame = cv2.bitwise_and(white, white, mask=black_mask)
    img_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    pink_frame = cv2.bitwise_and(white, white, mask=pink_mask)

    return red_frame, black_frame, pink_frame, img_gray

class AStar:
    """A*寻路算法实现（二维网格）"""
    
    def __init__(self, grid: List[List[int]]):
        """
        初始化地图
        :param grid: 二维数组，0表示可通行，1表示障碍物
        """
        self.grid = grid
        self.rows = len(grid)
        self.cols = len(grid[0]) if self.rows > 0 else 0
    
    def heuristic(self, a: Tuple[int, int], b: Tuple[int, int]) -> float:
        """
        启发式函数：使用欧几里得距离
        """
        return math.sqrt((a[0] - b[0]) * (a[0] - b[0]) + (a[1] - b[1]) * (a[1] - b[1]))
    
    def get_neighbors(self, node: Tuple[int, int]) -> List[Tuple[Tuple[int, int], float]]:
        x, y = node
        neighbors = []
        # 8个方向：(dx, dy)
        directions = [
            (-1, 0), (1, 0), (0, -1), (0, 1),          # 上下左右
            (-1, -1), (-1, 1), (1, -1), (1, 1)         # 对角线
        ]
        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            if not (0 <= nx < self.rows and 0 <= ny < self.cols):
                continue
            if self.grid[nx][ny] == 1:
                continue
            
            # 计算步长代价
            if dx != 0 and dy != 0:   # 对角线
                # 防穿墙检查：如果相邻的正交格子有障碍物，禁止斜穿
                if self.grid[x][ny] == 1 or self.grid[nx][y] == 1:
                    continue
                step_cost = math.sqrt(2)
            else:
                step_cost = 1.0
            
            neighbors.append(((nx, ny), step_cost))
        return neighbors
    
    def find_path(self, start: Tuple[int, int], goal: Tuple[int, int]) -> Optional[List[Tuple[int, int]]]:
        """
        A*寻路主函数
        :param start: 起点坐标 (row, col)
        :param goal: 终点坐标 (row, col)
        :return: 路径列表（从起点到终点），若无解则返回None
        """
        # 开放列表：存储待探索的节点 (f值, g值, 节点坐标)
        # 使用heapq实现优先队列
        open_list = []
        heapq.heappush(open_list, (0, 0, start))
        
        # 记录每个节点的父节点（用于重建路径）
        parent = {start: None}
        
        # g_cost: 从起点到每个节点的实际代价
        g_cost = {start: 0}
        
        # 已探索集合（关闭列表）
        closed_set = set()
        
        while open_list:
            # 弹出f值最小的节点
            current_f, current_g, current = heapq.heappop(open_list)
            
            # 如果已探索过，跳过
            if current in closed_set:
                continue
            
            # 加入已探索集合
            closed_set.add(current)
            
            # 到达目标节点，重建路径
            if current == goal:
                return self._reconstruct_path(parent, goal)
            
            # 扩展邻居节点
            for neighbor, step_cost in self.get_neighbors(current):
                if neighbor in closed_set:
                    continue
                
                # 计算经过当前节点到达邻居的代价
                tentative_g = g_cost[current] + step_cost  # 每步代价为1
                
                # 如果邻居不在g_cost中，或者找到了更优路径
                if neighbor not in g_cost or tentative_g < g_cost[neighbor]:
                    # 更新最优路径
                    parent[neighbor] = current
                    g_cost[neighbor] = tentative_g
                    f_value = tentative_g + self.heuristic(neighbor, goal)
                    heapq.heappush(open_list, (f_value, tentative_g, neighbor))
        
        # 开放列表为空，无解
        return None
    
    def _reconstruct_path(self, parent: dict, goal: Tuple[int, int]) -> List[Tuple[int, int]]:
        """
        从父节点字典重建路径
        """
        path = []
        current = goal
        while current is not None:
            path.append(current)
            current = parent[current]
        path.reverse()
        return path

def get_passable_grid(binary_image, grid_count):
    img_height, img_width = binary_image.shape
    grid_height = img_height // grid_count
    grid_width = img_width // grid_count
    passable_grid = []
    for i in range (grid_count):
        passable_row = []
        for j in range (grid_count):
            grid = binary_image[i * grid_height:(i + 1) * grid_height, j * grid_width:(j + 1) * grid_width]
            white_count = cv2.countNonZero(grid)
            if white_count < 1000:
                passable_row.append(0)  # 可通行
            else:
                passable_row.append(1)  # 障碍物
        passable_grid.append(passable_row)
    return passable_grid

def draw_path_on_grid(grid: List[List[int]], 
                      path: List[Tuple[int, int]],
                      start: Tuple[int, int],
                      goal: Tuple[int, int],
                      cell_size: int = 10) -> np.ndarray:
    """
    在网格地图上绘制路径，返回绘制好的图像 (BGR)
    """
    rows, cols = len(grid), len(grid[0])
    height, width = rows * cell_size, cols * cell_size
    img = np.ones((height, width, 3), dtype=np.uint8) * 255

    # 绘制网格线
    for i in range(rows + 1):
        cv2.line(img, (0, i * cell_size), (width, i * cell_size), (200, 200, 200), 1)
    for j in range(cols + 1):
        cv2.line(img, (j * cell_size, 0), (j * cell_size, height), (200, 200, 200), 1)

    # 障碍物
    for i in range(rows):
        for j in range(cols):
            if grid[i][j] == 1:
                cv2.rectangle(img, (j * cell_size, i * cell_size),
                              ((j + 1) * cell_size, (i + 1) * cell_size), (0, 0, 0), -1)

    def to_pixel(row, col):
        return (int(col * cell_size + cell_size / 2),
                int(row * cell_size + cell_size / 2))

    # 路径
    if path:
        pts = [to_pixel(r, c) for r, c in path]
        for i in range(len(pts) - 1):
            cv2.line(img, pts[i], pts[i + 1], (255, 0, 0), 3, cv2.LINE_AA)
        for p in pts:
            cv2.circle(img, p, 3, (255, 0, 0), -1)

    # 起点终点
    cv2.circle(img, to_pixel(start[0], start[1]), 8, (0, 255, 0), -1)
    cv2.circle(img, to_pixel(goal[0], goal[1]), 8, (255, 0, 255), -1)

    return img   # 返回图像数组

def smooth_path_bspline(path: List[Tuple[int, int]], num_points: int = 300) -> np.ndarray:
    """
    对路径进行 B 样条平滑，返回平滑后的点集 (N×2)，每行为 (row, col)
    """
    if len(path) < 4:
        return np.array(path, dtype=np.float32)
    path_np = np.array(path, dtype=np.float32)
    x = path_np[:, 1]   # col
    y = path_np[:, 0]   # row
    tck, u = splprep([x, y], s=0, k=3)
    u_new = np.linspace(0, 1, num_points)
    x_s, y_s = splev(u_new, tck)
    return np.vstack([y_s, x_s]).T   # 恢复为 (row, col)

def draw_smooth_path_on_grid(
    grid: List[List[int]],
    raw_path: List[Tuple[int, int]],          # A* 原始路径点
    start: Tuple[int, int],
    goal: Tuple[int, int],
    cell_size: int = 20,
    smooth_points: int = 400,                 # 平滑插值点数
    show_raw: bool = True                     # 是否同时显示原始折线（蓝色）
) -> np.ndarray:
    """
    绘制网格、障碍物、原始路径（可选）和平滑曲线（红色）
    返回 BGR 图像数组，不阻塞
    """
    rows, cols = len(grid), len(grid[0])
    height, width = rows * cell_size, cols * cell_size
    img = np.ones((height, width, 3), dtype=np.uint8) * 255

    # 绘制网格线
    for i in range(rows + 1):
        cv2.line(img, (0, i * cell_size), (width, i * cell_size), (200, 200, 200), 1)
    for j in range(cols + 1):
        cv2.line(img, (j * cell_size, 0), (j * cell_size, height), (200, 200, 200), 1)

    # 障碍物
    for i in range(rows):
        for j in range(cols):
            if grid[i][j] == 1:
                cv2.rectangle(img, (j * cell_size, i * cell_size),
                              ((j + 1) * cell_size, (i + 1) * cell_size), (0, 0, 0), -1)

    def grid_to_pixel(row, col):
        return (int(col * cell_size + cell_size / 2),
                int(row * cell_size + cell_size / 2))

    # ---- 原始折线（蓝色） ----
    if show_raw and raw_path:
        pts = [grid_to_pixel(r, c) for r, c in raw_path]
        for i in range(len(pts) - 1):
            cv2.line(img, pts[i], pts[i + 1], (255, 0, 0), 2, cv2.LINE_AA)

    # ---- 平滑曲线（红色） ----
    if raw_path and len(raw_path) >= 4:
        smoothed = smooth_path_bspline(raw_path, num_points=smooth_points)
        # 将平滑点转为像素坐标（浮点映射，保证连续）
        pts_smooth = []
        for r, c in smoothed:
            px = c * cell_size + cell_size / 2.0
            py = r * cell_size + cell_size / 2.0
            pts_smooth.append((int(px), int(py)))
        if len(pts_smooth) > 1:
            pts_array = np.array(pts_smooth, dtype=np.int32).reshape((-1, 1, 2))
            cv2.polylines(img, [pts_array], False, (0, 0, 255), 3, cv2.LINE_AA)
    else:
        # 路径太短，直接画原始点作为折线（降级）
        if raw_path:
            pts = [grid_to_pixel(r, c) for r, c in raw_path]
            for i in range(len(pts) - 1):
                cv2.line(img, pts[i], pts[i + 1], (0, 0, 255), 2, cv2.LINE_AA)

    # ---- 起点终点 ----
    cv2.circle(img, grid_to_pixel(start[0], start[1]), 8, (0, 255, 0), -1)
    cv2.circle(img, grid_to_pixel(goal[0], goal[1]), 8, (255, 0, 255), -1)

    return img

def hl(image):

    # 取ROI
    height, width = image.shape
    # 边缘检测与霍夫直线检测
    edges = cv2.Canny(image, 50, 150)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=50, minLineLength=30, maxLineGap=10)

    # 用于存储所有有效线段的参数 (x1, y1, x2, y2, slope)
    valid_lines = []
    # 创建输出图片
    output_image = image.copy()

    # 遍历所有直线
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line
            # 绘制所有直线
            cv2.line(output_image, (x2, y2), (x1, y1), (255, 0, 255), 4)
            # 计算线段的斜率,垂直直线给一个大值
            if x2 != x1:
                slope = (y2 - y1) / (x2 - x1)
            else:
                slope = 999.0
            # 计算线段长度,过滤掉过短的线段
            length = np.hypot(x2 - x1, y2 - y1)
            if length < 30:
                continue
            # 存储有效线段
            valid_lines.append((x1, y1, x2, y2, slope, length))

    if valid_lines:
        # 过滤过小斜率
        filtered_lines = [l for l in valid_lines if abs(l[4]) >= AVG_SLOPE_FILTER_THRESHOLD]
        if not filtered_lines:
            filtered_lines = valid_lines

        # 计算平均斜率与中心线在ROI底部的x坐标
        avg_slope = np.mean([l[4] for l in filtered_lines])
        all_x = [l[0] for l in filtered_lines] + [l[2] for l in filtered_lines]
        all_y = [l[1] for l in filtered_lines] + [l[3] for l in filtered_lines]
        p_start_x = int(np.mean(all_x))
        p_start_y = int(np.mean(all_y))

        # 计算中心线在ROI顶部的X坐标,斜率很小时认为delta_x = 0
        delta_y = height
        if abs(avg_slope) < 0.001:
            delta_x = 0
        else:
            delta_x = int(delta_y / avg_slope)
        p_end_x = p_start_x - delta_x

        # 使用p_start_x计算中心偏移
        offset_x = p_start_x - (width // 2)
        offset_y = p_start_y - (height // 2)

        # 计算角度
        center_angle_rad = math.atan(avg_slope)
        angle_horiz = math.degrees(center_angle_rad)

        # 转换为与垂直方向的夹角 (垂直向前为 0 度。左转为负，右转为正)
        # 确保斜率接近垂直时角度接近 0
        if abs(avg_slope) > 999:
            center_angle = 0.0
        else:
            center_angle = 90 - angle_horiz if angle_horiz > 0 else -90 - angle_horiz

        # 绘制中心线及其他可视化
        cv2.line(output_image, (p_start_x, height), (p_end_x, 0), (0, 255, 255), 4)
        cv2.putText(output_image, f'Angle: {center_angle:.2f} degrees', (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        cv2.putText(output_image, f'Avg X: {offset_x}', (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        
    else:
        # 未检测到有效轨道
        offset_x = 0
        offset_y = 0
        center_angle = 0.0

    return offset_x, offset_y, center_angle, output_image

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

def get_grab_state(target_center, pink_center, distance_threshold=100):
    if target_center is None or pink_center is None:
        return False

    distance = np.linalg.norm(np.array(target_center) - np.array(pink_center))
    return distance < distance_threshold

def get_closest_center(centers, target_center):
    if not centers or target_center is None:
        return None

    closest_center = min(centers, key=lambda c: np.linalg.norm(np.array(c) - np.array(target_center)))
    return closest_center

def find_lines_via_hough(black_frame, min_line_length=50, max_line_gap=20):
    edges = cv2.Canny(black_frame, 50, 150)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=80, minLineLength=min_line_length, maxLineGap=max_line_gap)

    return lines

def main(conn=None, frame_ready1=None, frame_ready2=None):
    # 显示FPS
    last_time = time.time()
    current_time = time.time()
    fps = 0
    frame_count = 0
    last_red = (0, 0)
    last_green = (0, 0)
    # 打开摄像头
    cap = None
    cap2 = None
    if frame_ready1 is None:
        cap = open_camera(0)
        if cap is None:
            return
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab initial frame")
            cap.release()
            return
    if frame_ready2 is None:
        cap2 = open_camera(1)
        if cap2 is None:
            print("Camera 2 could not be opened.")
            frame_ready2 = None
        ret2, frame2 = cap2.read() if cap2 is not None else (False, None)
        if not ret2:
            print("Failed to grab initial frame from camera 2")
            cap2.release()
    
    try:
        while True:
            # 获取图像
            frame_not_ready1 = False
            frame_not_ready2 = False
            frame = None
            frame2 = None
            if frame_ready1 is not None:
                if frame_ready1.value:
                    frame = frame_share.read()
                    frame_ready1.value = False
                else:
                    frame_not_ready1 = True
            else:
                ret, frame = cap.read()
                if not ret:
                    print("Failed to grab frame from camera 1")
                    break

            if frame_ready2 is not None:
                if frame_ready2.value:
                    frame2 = frame_share2.read()
                    frame_ready2.value = False
                else:
                    frame_not_ready2 = True
            else:
                ret2, frame2 = cap2.read() if cap2 is not None else (False, None)
                if not ret2 and cap2 is not None:
                    print("Failed to grab frame from camera 2")
                    break
            
            if frame_not_ready1 and frame_not_ready2:
                time.sleep(0.01)
                continue

            if not frame_not_ready1:
                # 预处理
                red_frame, black_frame, pink_frame, gray= preprocess_frame(frame)
                # 对预处理得到的图进行后处理
                # gray = lb.Sigmoid_Curve_Transform_LUT(gray, k=15.0, threshold=150)
                # gray = lb.Adaptive_Sigmoid_Transform(gray, grid_size=(8, 8), k_base=10.0, k_range=(5.0, 15.0))
                # gray = lb.Adaptive_Sigmoid_Transform_Fast(gray, k_base=10.0, k_range=(5.0, 15.0), filter_size=31)
                # gray = lb.Adaptive_Sigmoid_Transform_UMat(gray, grid_size=(8, 8), k_base=10.0, k_range=(5.0, 15.0))
                # gray = cv2.createCLAHE(clipLimit=8.0, tileGridSize=(8, 8)).apply(gray)
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
                binary = cv2.threshold(gray, 120, 255, cv2.THRESH_BINARY_INV)[1]
                binary = cv2.dilate(binary, kernel, iterations=2)
                pink_frame = cv2.erode(pink_frame, kernel, iterations=1)
                pink_frame = cv2.dilate(pink_frame, kernel, iterations=3)

                # lines = find_lines_via_hough(black_frame, min_line_length=50, max_line_gap=20)
                # if lines is not None:
                #     for line in lines:
                #         x1, y1, x2, y2 = line
                #         cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                # 找出可以通行的网格
                passable_grid = get_passable_grid(binary, grid_count=10)
                start = (8, 5)  # 起点坐标 (row, col)
                goal = (0, 9)  # 终点坐标 (row, col)
                astar = AStar(passable_grid)
                path = astar.find_path(start, goal)
                path_frame = frame.copy()
                # 绘制平滑路径
                path_frame = draw_smooth_path_on_grid(passable_grid, path, start, goal, cell_size=20, smooth_points=400, show_raw=True)
                # 寻找目标（粉色纸片）
                centers = find_object(pink_frame, min_area=2000, max_area=None)
                closest_center = get_closest_center(centers, (TARGET_X, TARGET_Y))
                offsetx, offsety = 0, 0
                if closest_center is not None:
                    offsetx = closest_center[0] - TARGET_X
                    offsety = closest_center[1] - TARGET_Y
                else:
                    offsetx = 0
                    offsety = 0
                offsetx += 1000
                offsety += 1000
                # 与MCU进行通信
                if conn is not None:
                    conn.send([0, offsetx, offsety])
                grab_state = get_grab_state(closest_center, (TARGET_X, TARGET_Y), distance_threshold=50)
                cv2.circle(frame, (TARGET_X, TARGET_Y), 5, (0, 255, 255), -1)
                if grab_state:
                    cv2.putText(frame, "Grab State: True", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                else:
                    cv2.putText(frame, "Grab State: False", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

                if centers is not None:
                    for center in centers:
                        cv2.circle(frame, center, 5, (0, 255, 0), -1)
                        cv2.putText(frame, f"({center[0]}, {center[1]})", (center[0] + 10, center[1] - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

                if path is not None:
                    for (row, col) in path:
                        grid_height = binary.shape[0] // 10
                        grid_width = binary.shape[1] // 10
                        center_x = col * grid_width + grid_width // 2
                        center_y = row * grid_height + grid_height // 2
                        cv2.circle(frame, (center_x, center_y), 5, (255, 0, 0), -1)

                # if centers is not None:
                #     for center in centers:
                #         cv2.circle(frame, center, 5, (0, 255, 0), -1)
                #         cv2.putText(frame, f"({center[0]}, {center[1]})", (center[0] + 10, center[1] - 10),
                #                     cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

            # 显示图像
            if not frame_not_ready1:
                frame = cv2.resize(frame, (640, 360))
                gray = cv2.resize(gray, (640, 360))
                binary = cv2.resize(binary, (640, 360))
                # pink_frame = cv2.resize(pink_frame, (640, 360))
                # path_frame = cv2.resize(path_frame, (640, 360))
                cv2.imshow('Original Frame', frame)
                cv2.imshow('Gray Frame', gray)
                cv2.imshow("Binary", binary)
                # cv2.imshow("Pink Frame", pink_frame)
                # cv2.imshow("Path Frame", path_frame)
            if not frame_not_ready2:
                frame2 = cv2.resize(frame2, (640, 360))
                cv2.imshow('Camera 2 Frame', frame2)
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
        cap.release() if cap is not None else None
        cv2.destroyAllWindows()
        frame_share.close() if frame_share is not None else None
        frame_share2.close() if frame_share2 is not None else None

if __name__ == "__main__":
    main()
