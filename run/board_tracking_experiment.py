import numpy as np
from scipy.spatial import KDTree
from scipy.optimize import linear_sum_assignment
from collections import Counter
import cv2
import process_lib.image_lib as lb
import process_lib.control_lib as ctrl
from multiprocessing import shared_memory, Value, Pipe
import time
import math
import os

class BoardTrackerVoting:
    def __init__(self, piece_ids, square_ids):
        """
        piece_ids: 棋子固定编号列表，如 ['P0','P1',...]
        square_ids: 棋盘格固定编号列表，如 ['S0','S1',...]
        """
        self.piece_ids = list(piece_ids)
        self.square_ids = list(square_ids)
        self.all_ids = self.piece_ids + self.square_ids   # 所有对象的统一ID顺序
        self.history_coords = {}      # ID -> np.array([x, y])
        self.history_array = None     # 按 all_ids 顺序排列的历史坐标数组，形状 (N,2)
        self.initialized = False

    def initialize(self, piece_coords, square_coords):
        """
        在第一帧稳定、所有对象都被检测到且棋子未移动时调用。
        piece_coords: 棋子坐标列表，顺序与 piece_ids 一一对应
        square_coords: 棋盘格坐标列表，顺序与 square_ids 一一对应
        """
        for pid, coord in zip(self.piece_ids, piece_coords):
            self.history_coords[pid] = np.array(coord, dtype=float)
        for sid, coord in zip(self.square_ids, square_coords):
            self.history_coords[sid] = np.array(coord, dtype=float)

        # 把所有历史坐标按统一顺序堆成一个数组，方便矩阵运算
        self.history_array = np.array([self.history_coords[i] for i in self.all_ids])
        self.initialized = True

    def update(self, det_piece_coords, det_square_coords, distance_thresh=30.0, move_thresh=50.0):
        """
        每一帧调用，处理当前检测结果。
        """
        det_points = []
        det_info = []
        for i, c in enumerate(det_piece_coords):
            det_points.append(c)
            det_info.append(('piece', i))
        for i, c in enumerate(det_square_coords):
            det_points.append(c)
            det_info.append(('square', i))
        
        det_points = np.array(det_points, dtype=float)
        
        # 防崩溃：如果当前帧什么都没检测到，直接返回
        if len(det_points) == 0:
            return {}, None, list(self.square_ids), {}

        # 2. 投票找出全局平移向量 T (增加距离截断，防止跨区域乱配对)
        translations = []
        for det_pt in det_points:
            diffs = det_pt - self.history_array
            dists = np.linalg.norm(diffs, axis=1)
            # 限制：只在距离80像素以内的点对进行投票，认为物体不可能瞬间瞬移太远
            valid_indices = np.where(dists < 80)[0]
            for idx in valid_indices:
                translations.append(diffs[idx])
        
        # 防崩溃：如果没有产生任何合理的位移对，默认物体没动
        if len(translations) == 0:
            T = np.array([0.0, 0.0])
            best_votes = 0
        else:
            translations = np.array(translations)
            quantized = np.round(translations, 1)
            trans_tuples = [tuple(t) for t in quantized]
            counter = Counter(trans_tuples)
            best_trans_tuple, best_votes = counter.most_common(1)[0]
            T = np.array(best_trans_tuple)
            
            # 异常拦截：如果算出的位移大得离谱（比如>60像素），说明投票被噪点带偏了，强制归零
            if np.linalg.norm(T) > 60:
                print(f"警告: 位移 {T} 过大，判定为异常，强制归零")
                T = np.array([0.0, 0.0])
                
        print(f"估计的全局平移向量: {T}，得票 {best_votes}")

        # 3. 根据 T 预测所有历史对象在当前帧的位置
        predicted_array = self.history_array + T

        # 4. 使用匈牙利算法进行全局最优匹配 (解决冲突跳变问题)
        dist_matrix = np.linalg.norm(det_points[:, np.newaxis] - predicted_array, axis=2)
        row_ind, col_ind = linear_sum_assignment(dist_matrix)

        id_map = {}
        matched_pairs = {} # hist_id -> det_idx
        for r, c in zip(row_ind, col_ind):
            if dist_matrix[r, c] <= distance_thresh:
                hist_id = self.all_ids[c]
                id_map[r] = hist_id
                matched_pairs[hist_id] = r

        # 5. 找出被移动的棋子
        moved_piece = None
        piece_set = set(self.piece_ids)
        matched_piece_ids = {hid for hid in id_map.values() if hid in piece_set}

        for det_idx, info in enumerate(det_info):
            if info[0] == 'piece' and det_idx not in id_map:
                unmatched_ids = piece_set - matched_piece_ids
                if len(unmatched_ids) == 1:
                    moved_id = list(unmatched_ids)[0]
                    # 判断是否真的发生了移动（距离要足够远），而不是单纯的检测抖动
                    hist_idx = self.all_ids.index(moved_id)
                    pred_dist = np.linalg.norm(det_points[det_idx] - predicted_array[hist_idx])
                    if pred_dist > move_thresh:
                        moved_piece = (moved_id, det_points[det_idx])
                        id_map[det_idx] = moved_id
                        matched_piece_ids.add(moved_id)
                        matched_pairs[moved_id] = det_idx
                        print(f"检测到棋子移动: {moved_id} -> {det_points[det_idx]}")

        # 如果还没找到，但历史棋子少了一个，说明被移出画面了或漏检了
        if moved_piece is None:
            unmatched_ids = piece_set - matched_piece_ids
            if len(unmatched_ids) == 1:
                moved_piece = (list(unmatched_ids)[0], None)

        # 6. 找出被遮挡的棋盘格
        square_set = set(self.square_ids)
        matched_square_ids = {hid for hid in id_map.values() if hid in square_set}
        missing_squares = list(square_set - matched_square_ids)

        # 7. 生成所有对象的推断位置，并平滑更新 history_array (防止雪崩)
        inferred_coords = {}
        new_history = []
        for i, oid in enumerate(self.all_ids):
            if oid in matched_pairs:
                det_idx = matched_pairs[oid]
                coord = det_points[det_idx]
            else:
                coord = predicted_array[i]
            
            inferred_coords[oid] = tuple(coord.tolist())
            new_history.append(coord)
        
        # 平滑更新历史坐标：50%旧状态 + 50%新状态。防止单帧噪点导致整体飞掉
        self.history_array = 0.5 * self.history_array + 0.5 * np.array(new_history)

        return id_map, moved_piece, missing_squares, inferred_coords


"""
使用方法：
# 1. 定义你所有棋子和棋盘格的固定编号
tracker = BoardTrackerVoting(
    piece_ids=['P0','P1','P2','P3','P4','P5','P6','P7'],
    square_ids=['S0','S1','S2','S3','S4','S5','S6','S7','S8']
)

# 2. 第一帧稳定时初始化（所有对象都检测到了，棋子未移动）
init_pieces = [(120,200), (180,210), ...]   # 你的8个棋子坐标（顺序按你的编号）
init_squares = [(50,50), (150,50), ...]     # 你的9个格子坐标
tracker.initialize(init_pieces, init_squares)

# 3. 以后每一帧，传入当前检测结果
curr_pieces = [(121,201), (182,208), ...]   # 检测到的所有棋子（顺序随意）
curr_squares = [(51,51), (152,49)]          # 检测到的所有格子（可能不全）
id_map, moved, missing, inferred = tracker.update(curr_pieces, curr_squares)

print("匹配关系:", id_map)
print("被移动的棋子:", moved)
print("被遮挡的格子:", missing)
print("所有对象推断位置:", inferred)
"""


CAMERA_FPS = 30
CAMERA_WIDTH = 1280 # 1080p 1920*1080
CAMERA_HEIGHT = 720 # 1080p 1920*1080
FRAME_CENTER_X = CAMERA_WIDTH // 2
FRAME_CENTER_Y = CAMERA_HEIGHT // 2
MIN_AREA_SMALL = 9000
MAX_AREA_SMALL = 70000
MIN_AREA_LARGE = 30000
MAX_AREA_LARGE = 10000000
MIN_CHESS_RADIUS = 5
MAX_CHESS_RADIUS = 50
white = np.full((CAMERA_HEIGHT, CAMERA_WIDTH), 255, dtype=np.uint8)
clicked = False

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

# 对图像进行预处理
def preprocess_frame(frame):
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    pink_mask = cv2.inRange(frame, (130, 20, 100), (160, 150, 255))
    black_mask = cv2.inRange(frame, (0, 0, 0), (180, 255, 100))
    white_mask = cv2.inRange(frame, (20, 0, 170), (160, 50, 255))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    pink_mask = cv2.morphologyEx(pink_mask, cv2.MORPH_CLOSE, kernel)
    # black_mask = cv2.morphologyEx(black_mask, cv2.MORPH_CLOSE, kernel, iterations=3)
    pink_frame = cv2.bitwise_and(white, white, mask=pink_mask)
    black_frame = cv2.bitwise_and(white, white, mask=black_mask)
    white_frame = cv2.bitwise_and(white, white, mask=white_mask)

    return pink_frame, black_frame, white_frame

def find_contours(binary):
    contours, _ = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, None
    
    valid_contour_vertex_small = lb.Find_Poly(contours, shape=4, min_area=MIN_AREA_SMALL, max_area=MAX_AREA_SMALL, factor=0.1)
    valid_contour_vertex_large = lb.Find_Poly(contours, shape=4, min_area=MIN_AREA_LARGE, max_area=MAX_AREA_LARGE, factor=0.1)

    return valid_contour_vertex_small, valid_contour_vertex_large

def find_chess(gray):
    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, dp=1, minDist=80, param1=150, param2=30, minRadius=20, maxRadius=50)
    if circles is not None:
        circles = np.uint16(np.around(circles))
    return circles

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

def mouse_callback(event, x, y, flags, param):
    """鼠标回调函数：记录左键点击的坐标"""
    global clicked
    if event == cv2.EVENT_LBUTTONDOWN:
        clicked = True
        print("鼠标点击")

def main(conn=None):
    # 显示FPS
    last_time = time.time()
    current_time = time.time()
    fps = 0
    frame_count = 0
    target_point = (640, 360)  # 目标点坐标，位于图像中心
    current_point = (640, 360)  # 当前点坐标，初始化为图像中心
    global clicked
    clicked = False
    cv2.namedWindow("Original Frame")
    cv2.setMouseCallback("Original Frame", mouse_callback)

    tracker = BoardTrackerVoting(
        piece_ids=['W1', 'W2', 'W3', 'W4', 'B1', 'B2', 'B3', 'B4'],
        square_ids=['S1','S2','S3','S4','S5','S6','S7','S8', 'S9']
    )

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
            # 当前棋子位置
            black_chess_position = []
            white_chess_position = []
            # 获取图像
            _, frame = cap.read()
            # frame = frame[CAMERA_HEIGHT//4:3*CAMERA_HEIGHT//4, CAMERA_WIDTH//4:3*CAMERA_WIDTH//4]
            # frame = cv2.resize(frame, (CAMERA_WIDTH, CAMERA_HEIGHT), interpolation=cv2.INTER_NEAREST)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
            pink_frame, black_frame, white_frame = preprocess_frame(frame)
            contours, _ = cv2.findContours(black_frame, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            circles = find_chess(gray)
            valid_contour_vertex_small, valid_contour_vertex_large = find_contours(black_frame)
            if circles is not None:
                for (x, y, r) in circles[0, :]:
                    x, y, r = int(x), int(y), int(r)
                    roi = frame[max(0, y - r):min(frame.shape[0], y + r), max(0, x - r):min(frame.shape[1], x + r)]
                    chess_color = distinguish_chess_color(roi)
                    cv2.circle(frame, (x, y), r, (0, 255, 0), 2)
                    cv2.circle(frame, (x, y), 2, (0, 0, 255), 3)
                    if chess_color == "black":
                        black_chess_position.append((x, y))
                    else:
                        white_chess_position.append((x, y))

                    if len(black_chess_position) == 4 and len(white_chess_position) == 4 and len(valid_contour_vertex_small) >= 5:
                        if not tracker.initialized and len(valid_contour_vertex_small) >= 9:
                            # 修复1：顺序必须与 piece_ids=['W1','W2','W3','W4','B1','B2','B3','B4'] 一致
                            # 修复2：排序不能仅按Y，如果同一行有两个棋子，Y的微小误差会导致左右互换。应先按Y再按X排序
                            white_chess_position.sort(key=lambda p: (p[1], p[0]))
                            black_chess_position.sort(key=lambda p: (p[1], p[0]))
                            
                            init_pieces = white_chess_position + black_chess_position
                            init_squares = [lb._cal_single_center(v) for v in valid_contour_vertex_small[:9]]
                            
                            tracker.initialize(init_pieces, init_squares)
                            print("Tracker initialized with initial positions.")
                        elif not tracker.initialized and len(valid_contour_vertex_small) < 9:
                            print("Waiting for all squares to be detected for initialization.")
                        else:
                            while not clicked:
                                time.sleep(0.01)  # 等待用户点击
                                cv2.waitKey(1)
                            clicked = False  # 重置点击状态
                            
                            all_det_pieces = black_chess_position + white_chess_position
                            id_map, moved_piece, missing_squares, inferred_coords = tracker.update(all_det_pieces, [lb._cal_single_center(v) for v in valid_contour_vertex_small[:9]])
                            
                            # 修复3：显示算法处理后的真实ID，而不是按原始检测顺序显示
                            for det_idx, hid in id_map.items():
                                if hid.startswith('B') or hid.startswith('W'):
                                    coord = all_det_pieces[det_idx]
                                    color = (0, 0, 255) if hid.startswith('B') else (255, 255, 255)
                                    cv2.putText(frame, hid, coord, cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                                    
            else:
                cv2.putText(frame, "Chess Point: Not Found", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            

            if valid_contour_vertex_small is not None and len(valid_contour_vertex_small) > 1:
                for vertex in valid_contour_vertex_small:
                    if vertex is None:
                        continue
                    try:
                        center = lb._cal_single_center(vertex)
                        contour = np.array(vertex, dtype=np.int32).reshape(-1, 1, 2)
                        cv2.drawContours(frame, [contour], -1, (0, 255, 0), 3)
                        cv2.circle(frame, center, 5, (255, 0, 0), -1)
                    except Exception as e:
                        print(f"Error drawing contour: {e}")
                        continue
            if valid_contour_vertex_large is not None and len(valid_contour_vertex_large) > 1:
                for vertex in valid_contour_vertex_large:
                    if vertex is None:
                        continue
                    try:
                        center = lb._cal_single_center(vertex)
                        contour = np.array(vertex, dtype=np.int32).reshape(-1, 1, 2)
                        cv2.drawContours(frame, [contour], -1, (0, 255, 255), 3)
                        cv2.circle(frame, center, 5, (255, 0, 0), -1)
                    except Exception as e:
                        print(f"Error drawing contour: {e}")
                        continue
            # pink_frame = cv2.resize(pink_frame, (640, 480))
            # black_frame = cv2.resize(black_frame, (640, 480))
            gray = cv2.resize(gray, (640, 480))
            frame = cv2.resize(frame, (640, 480))
            white_frame = cv2.resize(white_frame, (640, 480))
            cv2.imshow("Gray Frame", gray)
            cv2.imshow("White Frame", white_frame)
            cv2.imshow("Original Frame", frame)
            # cv2.imshow("Pink Frame", pink_frame)
            # cv2.imshow("Black Frame", black_frame)
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
