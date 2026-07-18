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
MIN_AREA_SMALL = 5000
MAX_AREA_SMALL = 20000
MIN_AREA_LARGE = 30000
MAX_AREA_LARGE = 10000000
MIN_CHESS_RADIUS = 5
MAX_CHESS_RADIUS = 50
MODEL_PATH = "/home/ubuntu/Project/Project/run/best.rknn"
NUM_CLASSES = 2
white = np.full((CAMERA_HEIGHT, CAMERA_WIDTH), 255, dtype=np.uint8)
sorted_chess_board_position = []
isInitialized = False
chess_color = None
chess_index = None
board_index = None
phase = 0

class ObjectTracker:
    def __init__(self, num_objects):
        self.num_objects = num_objects
        # 核心列表：索引即 ID，存储的是 (cx, cy) 元组
        self.tracked_centers = [None] * num_objects  
        self.initialized = False

    def init_first_frame(self, curr_centers):
        """第一帧：将YOLO传进来的乱序中心点，按顺序分配给 ID 0, 1, 2..."""
        # 如果你希望按从左到右分配ID，可以在这里先排序：
        # sorted_centers = sorted(curr_centers, key=lambda p: p[0])
        # for i in range(self.num_objects): self.tracked_centers[i] = sorted_centers[i]
        
        for i in range(min(self.num_objects, len(curr_centers))):
            self.tracked_centers[i] = curr_centers[i]
        self.initialized = True

    def update(self, curr_centers):
        """
        输入: curr_centers (当前帧YOLO输出的乱序中心点列表，如 [(x1,y1), (x2,y2), ...])
        输出: (moved_id, new_position, distance)
              moved_id: 移动目标的ID，-1表示无移动
        """
        if not self.initialized:
            self.init_first_frame(curr_centers)
            return -1, None, 0.0

        # 1. 提取上一帧存储的固定顺序中心点
        prev_centers = self.tracked_centers  # 按 ID 0,1,2... 排列

        # 2. 匹配：建立 上一帧ID 与 当前帧乱序下标 的映射
        pairs = self._match(prev_centers, curr_centers)  # [(tracked_id, curr_index), ...]

        # 3. 构建新的跟踪列表（按ID顺序更新）
        new_tracked = [None] * self.num_objects
        moved_id = -1
        max_dist = 0.0

        for tracked_id, curr_idx in pairs:
            curr_point = curr_centers[curr_idx]
            new_tracked[tracked_id] = curr_point
            
            # 计算位移（找出移动最大的那个）
            prev_point = prev_centers[tracked_id]
            if prev_point is not None:
                dist = np.hypot(prev_point[0] - curr_point[0],
                                prev_point[1] - curr_point[1])
                if dist > max_dist:
                    max_dist = dist
                    moved_id = tracked_id

        # 4. 更新内部存储
        self.tracked_centers = new_tracked

        # 5. 阈值判断（防止抖动误报）
        if max_dist < 5.0:  # 像素阈值，可根据实际调整
            return -1, None, 0.0
        
        return moved_id, self.tracked_centers[moved_id], max_dist

    def _match(self, prev_centers, curr_centers):
        """
        贪心匹配 (数量固定且少时足够快)
        返回: [(tracked_id, curr_index), ...]
        """
        n = len(prev_centers)
        used_curr = [False] * n
        pairs = []
        
        # 计算 N x N 距离矩阵
        dist_matrix = np.zeros((n, n))
        for i in range(n):
            if prev_centers[i] is None:  # 针对漏检预留
                continue
            for j in range(n):
                dist_matrix[i][j] = np.hypot(prev_centers[i][0] - curr_centers[j][0],
                                             prev_centers[i][1] - curr_centers[j][1])
        
        # 逐轮找最小距离匹配
        for _ in range(n):
            min_dist = float('inf')
            min_i, min_j = -1, -1
            for i in range(n):
                if any(p[0] == i for p in pairs):  # ID i 已经匹配过了
                    continue
                if prev_centers[i] is None:
                    continue
                for j in range(n):
                    if used_curr[j]:
                        continue
                    if dist_matrix[i][j] < min_dist:
                        min_dist = dist_matrix[i][j]
                        min_i, min_j = i, j
            if min_i != -1:
                pairs.append((min_i, min_j))
                used_curr[min_j] = True
                
        return pairs

def initialization(black_chess_position, white_chess_position, contour_vertex, last_black_chess_position, last_white_chess_position):
    sorted_chess_board_position = []
    chess_board_position = []
    chess_board_distances = []
    if len(black_chess_position) == 5 and len(white_chess_position) == 5 and len(contour_vertex) == 9:
        last_black_chess_position = black_chess_position.copy()
        last_white_chess_position = white_chess_position.copy()
        # 得出黑棋的拟合直线和垂线
        k, b = np.polyfit([p[0] for p in black_chess_position], [p[1] for p in black_chess_position], 1) if len(black_chess_position) >= 2 else (0, 0)
        x0, y0 = black_chess_position[0] if len(black_chess_position) > 0 else (0, 0)
        k_perp = -1 / k if k != 0 else 0
        b_perp = y0 - k_perp * x0 if k != 0 else 0

        if contour_vertex is not None and len(contour_vertex) > 1:
            for vertex in contour_vertex:
                if vertex is None:
                    continue
                try:
                    center = lb._cal_single_center(vertex)
                    contour = np.array(vertex, dtype=np.int32).reshape(-1, 1, 2)
                    chess_board_position.append(center)
                except Exception as e:
                    print(f"Error drawing contour: {e}")
                    continue
            for i in range(min(len(contour_vertex), 9)):
                center = lb._cal_single_center(contour_vertex[i])
        
        if k_perp != 0 and len(chess_board_position) == 9:
            for center in chess_board_position:
                distance = (k_perp * center[0] - center[1] + b_perp) / math.sqrt(k_perp**2 + 1)
                distance = -distance #if k_perp > 0 else distance
                chess_board_distances.append(distance)
            combined = list(zip(chess_board_position, chess_board_distances))
            combined_sorted = sorted(combined, key=lambda x: x[1])
            sorted_positions = [item[0] for item in combined_sorted]
            sorted_distances = [item[1] for item in combined_sorted]
            for i in range (3):
                chess_board_centers = sorted_positions[i*3:(i+1)*3]
                chess_board_centers.sort(key=lambda x: x[1])  # 按y坐标排序
                for j in range(3):
                    sorted_chess_board_position.append(chess_board_centers[j]) if k_perp > 0 else sorted_chess_board_position.append(chess_board_centers[2-j])
        return sorted_chess_board_position, last_black_chess_position, last_white_chess_position
    return None, None, None

def get_chess_state(black_chess_position, white_chess_position, sorted_chess_board_position):
    chess_state = []
    for i in range(9):
        if i < len(sorted_chess_board_position):
            board_center = sorted_chess_board_position[i]
            found_black = any(np.hypot(board_center[0] - bx, board_center[1] - by) < 30 for bx, by in black_chess_position)
            found_white = any(np.hypot(board_center[0] - wx, board_center[1] - wy) < 30 for wx, wy in white_chess_position)
            if found_black:
                chess_state.append('B')
            elif found_white:
                chess_state.append('W')
            else:
                chess_state.append('E')  # Empty
        else:
            chess_state.append('E')  # Empty
    return chess_state

# 各题目对应函数
# 题目1：将任意黑棋子移动到5号方格
def task_1(black_chess_position, white_chess_position, last_black_chess_position, last_white_chess_position, valid_contour_vertex_small, conn):
    global isInitialized
    global chess_color, chess_index
    global phase
    global sorted_chess_board_position
    if not isInitialized:
        sorted_chess_board_position, last_black_chess_position, last_white_chess_position = initialization(black_chess_position, white_chess_position, valid_contour_vertex_small, last_black_chess_position, last_white_chess_position)
        if sorted_chess_board_position is not None:
            isInitialized = True
            print("Initialization complete.")
    else:
        if phase == 0:
            bx, by = black_chess_position[chess_index]
            if conn is not None:
                conn.send((bx, by))
        elif phase == 1:
            cx, cy = sorted_chess_board_position[4]  # 5号方格的中心点
            if conn is not None:
                conn.send((cx, cy))
    # 将当前的黑白棋子位置和轮廓返回，以便在主循环中继续使用
    return last_black_chess_position, last_white_chess_position, valid_contour_vertex_small

# 题目2：能将任意两颗黑棋子和两颗白棋子放入指定方格中
# 此处的chess_color和chess_index、chess_board_index均需要为列表
def task_2(black_chess_position, white_chess_position, last_black_chess_position, last_white_chess_position, valid_contour_vertex_small, conn):
    global isInitialized
    global chess_color, chess_index, board_index
    global sorted_chess_board_position
    if not isInitialized:
        sorted_chess_board_position, last_black_chess_position, last_white_chess_position = initialization(black_chess_position, white_chess_position, valid_contour_vertex_small, last_black_chess_position, last_white_chess_position)
        if sorted_chess_board_position is not None:
            isInitialized = True
            print("Initialization complete.")
    else:
        if phase == 0:
            cx, cy = black_chess_position[chess_index[0]] if chess_color == "black" else white_chess_position[chess_index[0]]
            if conn is not None:
                conn.send((cx, cy))
        elif phase == 1:
            cx, cy = sorted_chess_board_position[board_index[0]]  # 第一个目标方格的中心点
            if conn is not None:
                conn.send((cx, cy))
        elif phase == 2:
            cx, cy = black_chess_position[chess_index[1]] if chess_color == "black" else white_chess_position[chess_index[1]]
            if conn is not None:
                conn.send((cx, cy))
        elif phase == 3:
            cx, cy = sorted_chess_board_position[board_index[1]]  # 第二个目标方格的中心点
            if conn is not None:
                conn.send((cx, cy))
    # 将当前的黑白棋子位置和轮廓返回，以便在主循环中继续使用
    return last_black_chess_position, last_white_chess_position, valid_contour_vertex_small

# 题目3：将棋盘在45度范围内旋转后，能将任意两颗黑棋子和两颗白棋子放入指定方格中
# 因棋子移动逻辑和task_2相同，故直接调用task_2即可
def task_3(black_chess_position, white_chess_position, last_black_chess_position, last_white_chess_position, valid_contour_vertex_small, conn):
    global isInitialized
    global chess_color, chess_index, board_index
    global sorted_chess_board_position
    if not isInitialized:
        sorted_chess_board_position, last_black_chess_position, last_white_chess_position = initialization(black_chess_position, white_chess_position, valid_contour_vertex_small, last_black_chess_position, last_white_chess_position)
        if sorted_chess_board_position is not None:
            isInitialized = True
            print("Initialization complete.")
    else:
        if phase == 0:
            cx, cy = black_chess_position[chess_index[0]] if chess_color == "black" else white_chess_position[chess_index[0]]
            if conn is not None:
                conn.send((cx, cy))
        elif phase == 1:
            cx, cy = sorted_chess_board_position[board_index[0]]  # 第一个目标方格的中心点
            if conn is not None:
                conn.send((cx, cy))
        elif phase == 2:
            cx, cy = black_chess_position[chess_index[1]] if chess_color == "black" else white_chess_position[chess_index[1]]
            if conn is not None:
                conn.send((cx, cy))
        elif phase == 3:
            cx, cy = sorted_chess_board_position[board_index[1]]  # 第二个目标方格的中心点
            if conn is not None:
                conn.send((cx, cy))
    # 将当前的黑白棋子位置和轮廓返回，以便在主循环中继续使用
    return last_black_chess_position, last_white_chess_position, valid_contour_vertex_small

# 题目4：装置执黑棋与人对弈（第一步方格可设置），若人应对的第一步白棋有错误，装置能获胜
def task_4(black_chess_position, white_chess_position, last_black_chess_position, last_white_chess_position, valid_contour_vertex_small, conn):
    global isInitialized
    global chess_color, chess_index, board_index
    global sorted_chess_board_position
    if not isInitialized:
        sorted_chess_board_position, last_black_chess_position, last_white_chess_position = initialization(black_chess_position, white_chess_position, valid_contour_vertex_small, last_black_chess_position, last_white_chess_position)
        if sorted_chess_board_position is not None:
            isInitialized = True
            print("Initialization complete.")
    
    return last_black_chess_position, last_white_chess_position, valid_contour_vertex_small

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

def chess_position_init(black_chess_position, white_chess_position):
    for i in range(4):
        black_chess_position.append((0, 0))
        white_chess_position.append((0, 0))
    return black_chess_position, white_chess_position

# 对图像进行预处理
def preprocess_frame(frame):
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    pink_mask = cv2.inRange(frame, (130, 20, 100), (160, 150, 255))
    black_mask = cv2.inRange(frame, (0, 0, 0), (180, 255, 100))
    white_mask = cv2.inRange(frame, (20, 0, 170), (160, 50, 255))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    pink_mask = cv2.morphologyEx(pink_mask, cv2.MORPH_CLOSE, kernel)
    black_mask = cv2.morphologyEx(black_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    pink_frame = cv2.bitwise_and(white, white, mask=pink_mask)
    black_frame = cv2.bitwise_and(white, white, mask=black_mask)
    white_frame = cv2.bitwise_and(white, white, mask=white_mask)

    return pink_frame, black_frame, white_frame

def find_contours(binary):
    contours, _ = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, None
    
    valid_contour_vertex_small = lb.Find_Poly(contours, shape=4, min_area=MIN_AREA_SMALL, max_area=MAX_AREA_SMALL, factor=0.1)

    return valid_contour_vertex_small

def find_chess(gray):
    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, dp=1, minDist=80, param1=150, param2=30, minRadius=20, maxRadius=50)
    if circles is not None:
        circles = np.uint16(np.around(circles))
    return circles

def find_chess_via_yolo(img, detector):
    boxes, _, cls_ids = detector.detect(img)
    black_centers = []
    white_centers = []
    for box, cls_id in zip(boxes, cls_ids):
        x1, y1, x2, y2 = box
        center_x = (x1 + x2) // 2
        center_y = (y1 + y2) // 2
        if cls_id == 0:  # 假设0是黑棋
            black_centers.append((center_x, center_y))
        elif cls_id == 1:  # 假设1是白棋
            white_centers.append((center_x, center_y))
    return black_centers, white_centers

def distinguish_chess_color(roi):
    roi_hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    black_mask = cv2.inRange(roi_hsv, (0, 0, 0), (180, 255, 100))
    white_mask = cv2.inRange(roi_hsv, (20, 0, 170), (160, 50, 255))
    black_count = cv2.countNonZero(black_mask)
    white_count = cv2.countNonZero(white_mask)
    if black_count > white_count:
        return "black"
    else:
        return "white"

def main(conn=None):
    global isInitialized
    global sorted_chess_board_position
    global phase
    global chess_color, chess_index, board_index
    with lb.YOLODetector(MODEL_PATH, NUM_CLASSES, method="rknn", conf_thresh=0.5, iou_thresh=0.60, imgsz=(224,224)) as detector:
        # 显示FPS
        last_time = time.time()
        current_time = time.time()
        fps = 0
        frame_count = 0
        target_point = (640, 360)  # 目标点坐标，位于图像中心
        current_point = (640, 360)  # 当前点坐标，初始化为图像中心
        # 历史棋子和棋盘位置
        last_black_chess_position = []
        last_white_chess_position = []
        sorted_chess_board_position = []

        black_tracker = ObjectTracker(num_objects=5)  # 假设最多有5个黑棋
        white_tracker = ObjectTracker(num_objects=5)  # 假设最多有5个白棋

        task = "task_1"  # 当前任务
        isInitialized = False  # 标记是否已初始化棋子位置

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
                # 当前棋子位置(这里是每次循环更新的)
                black_chess_position = []
                white_chess_position = []
                # 获取图像并进行预处理
                _, frame = cap.read()
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                gray = cv2.GaussianBlur(gray, (15, 15), 0)
                gray = cv2.createCLAHE(clipLimit=6.0, tileGridSize=(8, 8)).apply(gray)
                gray_bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
                pink_frame, black_frame, white_frame = preprocess_frame(frame)
                # 找出黑白轮廓
                black_contours, _ = cv2.findContours(black_frame, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                white_contours, _ = cv2.findContours(white_frame, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                # 通过YOLO检测棋子位置
                black_chess_position, white_chess_position = find_chess_via_yolo(gray_bgr, detector)
                # 若漏检，跳过此帧
                if len(black_chess_position) != 5 or len(white_chess_position) != 5:
                    continue
                # 检测棋盘格
                valid_contour_vertex_small = find_contours(black_frame)
                cv2.drawContours(frame, valid_contour_vertex_small, -1, (0, 255, 0), 2) if valid_contour_vertex_small is not None else None
                # 绘制出棋子位置并编号
                black_chess_position.sort(key=lambda x: (x[1], x[0]))  # 按y坐标排序，y相同按x坐标排序
                white_chess_position.sort(key=lambda x: (x[1], x[0]))

                moved_id_black, new_position_black, distance_black = black_tracker.update(black_chess_position)
                moved_id_white, new_position_white, distance_white = white_tracker.update(white_chess_position)
                # 根据移动的棋子ID更新位置
                if moved_id_black != -1:
                    print(f"Black chess ID {moved_id_black} moved to {new_position_black} with distance {distance_black:.2f}")
                    black_chess_position[moved_id_black] = new_position_black
                    last_black_chess_position = black_chess_position.copy()
                if moved_id_white != -1:
                    print(f"White chess ID {moved_id_white} moved to {new_position_white} with distance {distance_white:.2f}")
                    white_chess_position[moved_id_white] = new_position_white
                    last_white_chess_position = white_chess_position.copy()

                for i, (bx, by) in enumerate(black_tracker.tracked_centers):
                    cv2.putText(frame, f"B{i+1}", (bx, by), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    cv2.circle(frame, (bx, by), 5, (0, 0, 255), -1)
                for i, (wx, wy) in enumerate(white_tracker.tracked_centers):
                    cv2.putText(frame, f"W{i+1}", (wx, wy), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    cv2.circle(frame, (wx, wy), 5, (255, 255, 255), -1)

                # 获取当前任务的目标点
                if task == "task_1":
                    last_black_chess_position, last_white_chess_position, valid_contour_vertex_small = task_1((black_chess_position, white_chess_position, last_black_chess_position, last_white_chess_position, valid_contour_vertex_small, conn))
                elif task == "task_2":
                    last_black_chess_position, last_white_chess_position, valid_contour_vertex_small = task_2((black_chess_position, white_chess_position, last_black_chess_position, last_white_chess_position, valid_contour_vertex_small, conn))
                elif task == "task_3":
                    last_black_chess_position, last_white_chess_position, valid_contour_vertex_small = task_3((black_chess_position, white_chess_position, last_black_chess_position, last_white_chess_position, valid_contour_vertex_small, conn))
                elif task == "task_4":
                    last_black_chess_position, last_white_chess_position, valid_contour_vertex_small = task_4((black_chess_position, white_chess_position, last_black_chess_position, last_white_chess_position, valid_contour_vertex_small, conn))

                if not isInitialized:
                    sorted_chess_board_position, last_black_chess_position, last_white_chess_position = initialization(black_chess_position, white_chess_position, valid_contour_vertex_small, last_black_chess_position, last_white_chess_position)
                    if sorted_chess_board_position is not None:
                        isInitialized = True
                        print("Initialization complete.")
                        # for i, pos in enumerate(sorted_chess_board_position):
                        #     cv2.putText(frame, f"S{i+1}", pos, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                        #     cv2.circle(frame, pos, 5, (0, 255, 0), -1)
                        # cv2.imshow("Original Frame", frame)
                        # while True:
                        #     cv2.waitKey(1)
                        #     time.sleep(0.1)

                black_frame = cv2.resize(black_frame, (640, 480))
                # gray = cv2.resize(gray, (640, 480))
                frame = cv2.resize(frame, (640, 480))
                # white_frame = cv2.resize(white_frame, (640, 480))
                gray_bgr = cv2.resize(gray_bgr, (640, 480))
                cv2.imshow("Gray BGR Frame", gray_bgr)
                # cv2.imshow("Gray Frame", gray)
                # cv2.imshow("White Frame", white_frame)
                cv2.imshow("Original Frame", frame)
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
