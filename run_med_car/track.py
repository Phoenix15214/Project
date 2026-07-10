import cv2
import numpy as np
import process_lib.image_lib as lb
import process_lib.control_lib as ctrl
from multiprocessing import shared_memory, Value
import time
import math
import os

CAMERA_FPS = 120
AVG_SLOPE_FILTER_THRESHOLD = 0.8
MAX_CONTOUR_AREA = 2000
MIN_CONTOUR_AREA = 100
white_frame = np.full((240, 320), 255, dtype=np.uint8)
last_track_time = 0

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
    # if -eps <= ua <= 1 + eps and -eps <= ub <= 1 + eps:
    #     return int(x1 + ua * (x2 - x1)), int(y1 + ua * (y2 - y1))
    # return None

def _find_vertical_lines(lines, angle_threshold=10):
    segs = []
    isVertical = False
    intersection = None
    if lines is None:
        return False, None
    for line in lines:
        x1, y1, x2, y2, _, length = line
        dx, dy = x2 - x1, y2 - y1
        segs.append((x1, y1, x2, y2, dx, dy, length))

    cos_thresh = np.sin(np.deg2rad(angle_threshold))
    n = len(segs)
    for i in range(n):
        for j in range(i + 1, n):
            s1, s2 = segs[i], segs[j]
            dot_product = s1[4] * s2[4] + s1[5] * s2[5] # 向量点积
            cos_val = abs(dot_product) / (s1[6] * s2[6] + 1e-6)
            if cos_val < cos_thresh:
                isVertical = True
                intersection = _find_intersection(s1, s2)
                break
        if isVertical:
            break
    return isVertical, intersection

def hl(image):

    # 取ROI
    height, width = image.shape
    # 边缘检测与霍夫直线检测
    edges = cv2.Canny(image, 50, 150)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=50, minLineLength=30, maxLineGap=10)

    # 用于存储所有有效线段的参数 (x1, y1, x2, y2, slope)
    valid_lines = []
    # 用于储存路口检测的参数
    isVertical = False
    intersection = None
    # 创建输出图片
    output_image = image.copy()

    # 遍历所有直线
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
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
        # 检测路口
        isVertical, intersection = _find_vertical_lines(valid_lines)

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
            # center_angle = (90 - angle_horiz) if angle_horiz > 0 else (90 + angle_horiz)
            center_angle = 90 - angle_horiz if angle_horiz > 0 else -90 - angle_horiz

        # 绘制中心线及其他可视化
        cv2.line(output_image, (p_start_x, height), (p_end_x, 0), (0, 255, 255), 4)
        cv2.putText(output_image, f'Angle: {center_angle:.2f} degrees', (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        cv2.putText(output_image, f'Avg X: {offset_x}', (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        cv2.putText(output_image, f'Vertical: {isVertical}', (10, 110),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        
    else:
        # 未检测到有效轨道
        offset_x = 0
        offset_y = 0
        center_angle = 0.0

    return offset_x, offset_y, center_angle, output_image, isVertical, intersection

def main(shm_name, frame_ready, yolo_start=None, conn=None, stop_event=None, core=None):
    if core is not None:
        os.sched_setaffinity(0, {core})
    # cv2.setNumThreads(2)
    global last_track_time
    last_time = time.time()
    last_imshow_time = time.time()
    current_time = time.time()
    last_angle = 0
    isJunction = 0
    frame_count = 0
    fps = 0
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FPS, CAMERA_FPS)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture video")
        cap.release()
        exit()
    cv2.namedWindow('red', cv2.WINDOW_NORMAL)
    cv2.createTrackbar("Threshold", "red", 15, 255, lambda x: None)
    pack = ctrl.SerialPacket(port="/dev/ttyUSB0", baudrate=38400, timeout=0.1)if conn is None else None
    
    if shm_name is not None:
        try:
            shm = shared_memory.SharedMemory(name=shm_name, create=False, size=640*480*3)
        except:
            shm = shared_memory.SharedMemory(create=False, name=shm_name)
    frame_view = np.ndarray((480, 640, 3), dtype=np.uint8, buffer=shm.buf)if shm_name is not None else None
    stop_event = stop_event or type("StopEvent", (), {"is_set": staticmethod(lambda: False)})()

    try:
        while not stop_event.is_set():
            cx, cy = 0, 0
            ret, frame = cap.read()
            if not ret:
                print("Failed to capture video")
                raise OSError("Failed to capture video")
                break
            if frame_view is not None:
                frame_view[:] = frame
                frame_ready.value = True

            frame = cv2.resize(frame, (320, 240))
            frame = cv2.GaussianBlur(frame, (5, 5), 0)
            # 颜色提取
            red = lb.Color_Extraction(frame, color = lb.RED)
            black_mask = cv2.inRange(frame, (0, 0, 0), (180, 255, 100))
            binary_black = cv2.bitwise_and(white_frame, white_frame, mask=black_mask)
            gray = cv2.cvtColor(red, cv2.COLOR_BGR2GRAY)
            binary = cv2.threshold(gray, cv2.getTrackbarPos("Threshold", "red"), 255, cv2.THRESH_BINARY_INV)[1]
            contours = cv2.findContours(binary_black, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0]
            valid_contours = []

            for contour in contours:
                area = cv2.contourArea(contour)
                if MIN_CONTOUR_AREA < area < MAX_CONTOUR_AREA:
                    valid_contours.append(contour)
            if len(valid_contours) > 9:
                cx, cy = lb.Get_Center_Point(valid_contours, mode=lb.CENTER_ALL)

            # kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
            # binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
            edges = cv2.Canny(binary, 50, 150)
            lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=50, minLineLength=10, maxLineGap=5)
            valid_lines = []
            if lines is not None:
                for line in lines:
                    x1, y1, x2, y2 = line[0]
                    # cv2.line(output_image, (x2, y2), (x1, y1), (255, 0, 255), 4)
                    # 计算线段的斜率,垂直直线给一个大值
                    if x2 != x1:
                        slope = (y2 - y1) / (x2 - x1)
                    else:
                        slope = 999.0
                    # 过滤掉太平坦的线段
                    if abs(slope) < 0.2:
                        continue
                    # 存储有效线段
                    valid_lines.append((x1, y1, x2, y2, slope))
            # 霍夫直线检测
            offset_x, offset_y, angle, output_image, isVertical, intersection = hl(binary)
            # 修改yolo_start的值
            if yolo_start is not None:
                if isVertical:
                    yolo_start.value = True
                else:
                    yolo_start.value = True
            # 复位isJunction状态
            isJunction = 0
            if abs(angle - last_angle) > 75:
                if offset_y > 80:
                    isJunction = 100
                    # print("Junction detected!")
                    turning_standby = False
                else:
                    isJunction = 0
            if intersection is not None:
                inter_x, inter_y = intersection
            else:
                inter_x, inter_y = 0, 0
            angle = int(angle + 180)
            offset_x = int(offset_x + 1000)
            offset_y = int(offset_y + 1000)
            inter_y = inter_y + 5 if inter_y != 0 else 0
            cy = cy + 10 if cy != 0 else 0
            if pack is not None:
                pack.insert_byte(0x0E)  # 包头
                pack.insert_two_bytes(pack.num_to_bytes(angle))
                pack.insert_two_bytes(pack.num_to_bytes(offset_x))
                pack.insert_two_bytes(pack.num_to_bytes(inter_x))
                pack.insert_two_bytes(pack.num_to_bytes(inter_y))
                pack.insert_two_bytes(pack.num_to_bytes(cx))
                pack.insert_two_bytes(pack.num_to_bytes(cy))
                pack.insert_two_bytes(pack.num_to_bytes(isJunction))
            msg = [0, angle, offset_x, inter_x, inter_y, cx, cy, isJunction] # 0表示来自track.py的消息
            if conn is not None:
                try:
                    conn.send(msg)
                except (BrokenPipeError, EOFError, OSError):
                    break
            if pack is not None:
                pack.send_packet() if pack is not None else None
            current_time = time.time()
            if current_time - last_imshow_time >= 0.05:
                cv2.imshow("red", output_image)
                cv2.imshow("black", binary_black)
                last_imshow_time = current_time
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            frame_count += 1
            current_time = time.time()
            if current_time - last_time >= 1.0:
                fps = frame_count / (current_time - last_time)
                last_time = current_time
                frame_count = 0
                print(f"FPS: {fps:.2f}")
            # print(f"一次循环经过{time.time() - last_track_time:.4f}秒")
            # last_track_time = time.time()

    finally:
        if shm_name is not None:
            shm.close()
        cap.release()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main(None, Value('b', False), None, None, None, None)