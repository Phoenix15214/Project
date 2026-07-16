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
MIN_AREA_SMALL = 9000
MAX_AREA_SMALL = 70000
MIN_AREA_LARGE = 30000
MAX_AREA_LARGE = 10000000
MIN_CHESS_RADIUS = 5
MAX_CHESS_RADIUS = 50
white = np.full((CAMERA_HEIGHT, CAMERA_WIDTH), 255, dtype=np.uint8)

# 储存各对象坐标，前四个为黑棋，5-8为白棋，最后九个为棋盘格中心点
positions = []
# 初始化各对象位置
for i in range(17):
    positions.append((0, 0))



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
    white_mask = cv2.inRange(frame, (0, 0, 200), (180, 30, 255))
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
    white_mask = cv2.inRange(roi_hsv, (0, 0, 200), (180, 30, 255))
    black_count = cv2.countNonZero(black_mask)
    white_count = cv2.countNonZero(white_mask)
    if black_count > white_count:
        return "black"
    else:
        return "white"


def main(conn=None):
    # 显示FPS
    last_time = time.time()
    current_time = time.time()
    fps = 0
    frame_count = 0
    target_point = (640, 360)  # 目标点坐标，位于图像中心
    current_point = (640, 360)  # 当前点坐标，初始化为图像中心
    # 初始棋子位置，用于记住返回位置
    initial_black_chess_position = []
    initial_white_chess_position = []
    # 初始化棋子位置
    # black_chess_position, white_chess_position = chess_position_init(black_chess_position, white_chess_position)
    # initial_black_chess_position, initial_white_chess_position = chess_position_init(initial_black_chess_position, initial_white_chess_position)
    # 棋盘格中心位置
    chess_board_position = []

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
                black_chess_position.sort(key=lambda x: x[1])
                white_chess_position.sort(key=lambda x: x[1])
                for i in range(min(len(black_chess_position), 4)):
                    cv2.putText(frame, f"B{i+1}", black_chess_position[i], cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                for i in range(min(len(white_chess_position), 4)):
                    cv2.putText(frame, f"W{i+1}", white_chess_position[i], cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
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
