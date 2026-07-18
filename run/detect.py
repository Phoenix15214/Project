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
                distance = -distance if k_perp > 0 else distance
                chess_board_distances.append(distance)
            combined = list(zip(chess_board_position, chess_board_distances))
            combined_sorted = sorted(combined, key=lambda x: x[1])
            sorted_positions = [item[0] for item in combined_sorted]
            sorted_distances = [item[1] for item in combined_sorted]
            for i in range (3):
                chess_board_centers = sorted_positions[i*3:(i+1)*3]
                chess_board_centers.sort(key=lambda x: x[1])  # 按y坐标排序
                for j in range(3):
                    sorted_chess_board_position.append(chess_board_centers[j])
        return sorted_chess_board_position, last_black_chess_position, last_white_chess_position
    return None, None, None

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
    with lb.YOLODetector(MODEL_PATH, NUM_CLASSES, method="rknn", conf_thresh=0.5, iou_thresh=0.30, imgsz=(640,640)) as detector:
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
        # 当前棋子位置
        current_black_chess_position = []
        current_white_chess_position = []

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
        phase = 0
        try:
            while True:
                # 当前棋子位置
                black_chess_position = []
                white_chess_position = []
                chess_board_position = []
                chess_board_distances = []
                # 获取图像并进行预处理
                _, frame = cap.read()
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                gray = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
                pink_frame, black_frame, white_frame = preprocess_frame(frame)
                # 找出黑白轮廓
                black_contours, _ = cv2.findContours(black_frame, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                white_contours, _ = cv2.findContours(white_frame, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                # 通过YOLO检测棋子位置
                black_chess_position, white_chess_position = find_chess_via_yolo(frame, detector)
                valid_contour_vertex_small = find_contours(black_frame)
                # 绘制出棋子位置并编号
                if len(black_chess_position) > 0 and len(white_chess_position) > 0:
                    black_chess_position.sort(key=lambda x: (x[1], x[0]))  # 按y坐标排序，y相同按x坐标排序
                    white_chess_position.sort(key=lambda x: (x[1], x[0]))
                    for i, (bx, by) in enumerate(black_chess_position):
                        cv2.putText(frame, f"B{i+1}", (bx, by), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                        cv2.circle(frame, (bx, by), 5, (0, 0, 255), -1)
                    for i, (wx, wy) in enumerate(white_chess_position):
                        cv2.putText(frame, f"W{i+1}", (wx, wy), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                        cv2.circle(frame, (wx, wy), 5, (255, 255, 255), -1)

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
                white_frame = cv2.resize(white_frame, (640, 480))
                # cv2.imshow("Gray Frame", gray)
                cv2.imshow("White Frame", white_frame)
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
