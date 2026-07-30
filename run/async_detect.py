import time
import numpy as np
import cv2
import threading
import math
from queue import Queue, Empty, Full
from rknnlite.api import RKNNLite
from multiprocessing import shared_memory, Value, Pipe
from filterpy.kalman import KalmanFilter
import process_lib.control_lib as ctrl
import supervision as sv


# ========================= 配置 =========================
MODEL_PATH = "/home/ubuntu/Project/Project/run/best.rknn"
CAMERA_SOURCE = 0

CAP_WIDTH = 640
CAP_HEIGHT = 480

INFER_SIZE = 256
CONF_THRESH = 0.1
IOU_THRESH = 0.45

SHOW_DISPLAY = True
DISPLAY_WIDTH = 640
DISPLAY_HEIGHT = 480

line_start = (110, 305)
line_end = (490, 270)
# =======================================================


infer_q = Queue(maxsize=2)
raw_q = Queue(maxsize=2)
result_q = Queue(maxsize=2)
display_q = Queue(maxsize=2)
frame_share = ctrl.MemoryShare(name='shared_frame', shape=(CAP_HEIGHT, CAP_WIDTH,3), dtype='uint8')
white = np.full((CAP_HEIGHT, CAP_WIDTH), 255, dtype=np.uint8)

running = True


def _should_run(stop_event=None):
    return stop_event is None or not stop_event.is_set()


def letterbox_rknn(img, new_shape=256, color=(114, 114, 114)):
    h, w = img.shape[:2]
    scale = min(new_shape / h, new_shape / w)
    nw, nh = int(w * scale), int(h * scale)

    canvas = np.full((new_shape, new_shape, 3), color, dtype=np.uint8)
    pad_w, pad_h = (new_shape - nw) // 2, (new_shape - nh) // 2

    canvas[pad_h:pad_h + nh, pad_w:pad_w + nw] = cv2.resize(img, (nw, nh))
    return canvas, scale, pad_w, pad_h


def nms_rknn(boxes, scores, iou_thresh):
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = (x2 - x1).clip(min=0) * (y2 - y1).clip(min=0)
    order = scores.argsort()[::-1]
    keep = []

    while order.size > 0:
        i = order[0]
        keep.append(i)

        if order.size == 1:
            break

        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        inter = (xx2 - xx1).clip(min=0) * (yy2 - yy1).clip(min=0)
        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-6)

        order = order[1:][iou < iou_thresh]

    return keep


def postprocess_rknn(outputs, scale, pad_w, pad_h, conf_thresh, iou_thresh):
    pred = outputs[0][0].T

    boxes_xywh = pred[:, :4]
    class_scores = pred[:, 4:]

    scores = class_scores.max(axis=1)
    class_ids = class_scores.argmax(axis=1)

    mask = scores > conf_thresh
    boxes_xywh, scores, class_ids = boxes_xywh[mask], scores[mask], class_ids[mask]

    if len(boxes_xywh) == 0:
        return [], [], []

    boxes_xyxy = np.zeros_like(boxes_xywh)

    boxes_xyxy[:, 0] = boxes_xywh[:, 0] - boxes_xywh[:, 2] / 2
    boxes_xyxy[:, 1] = boxes_xywh[:, 1] - boxes_xywh[:, 3] / 2
    boxes_xyxy[:, 2] = boxes_xywh[:, 0] + boxes_xywh[:, 2] / 2
    boxes_xyxy[:, 3] = boxes_xywh[:, 1] + boxes_xywh[:, 3] / 2

    boxes_xyxy[:, [0, 2]] = (boxes_xyxy[:, [0, 2]] - pad_w) / scale
    boxes_xyxy[:, [1, 3]] = (boxes_xyxy[:, [1, 3]] - pad_h) / scale

    keep = nms_rknn(boxes_xyxy, scores, iou_thresh)

    return boxes_xyxy[keep].astype(np.int32), scores[keep], class_ids[keep]

def get_pink_center(frame):
    frame_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    pink_mask = cv2.inRange(frame_hsv, (130, 40, 100), (180, 80, 255))
    pink_frame = cv2.bitwise_and(white, white, mask=pink_mask)
    
    pink_frame = cv2.medianBlur(pink_frame, 5)
    pink_frame = cv2.dilate(pink_frame, None, iterations=2)
    contours, _ = cv2.findContours(pink_frame, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        sorted_contours = sorted(contours, key=cv2.contourArea, reverse=True)
        for contour in sorted_contours:
            area = cv2.contourArea(contour)
            if area > 100:
                M = cv2.moments(contour)
                if M["m00"] != 0:
                    cX = int(M["m10"] / M["m00"])
                    cY = int(M["m01"] / M["m00"])
                    if cX < 350:
                        continue
                    if cY < 250 or cY > 400:
                        continue
                    return (cX, cY)
    return None

def get_on_line_position(ball_center, line_start, line_end):
    x1, y1 = line_start
    x2, y2 = line_end
    bx, by = ball_center

    if x1 == x2 and y1 == y2:
        return (x1, y1), 0.0

    dx = x2 - x1
    dy = y2 - y1
    numerator = (bx - x1) * dx + (by - y1) * dy
    denominator = dx * dx + dy * dy
    if denominator == 0:
        return (x1, y1), 0.0

    t = numerator / denominator
    px = x1 + t * dx
    py = y1 + t * dy

    if t < 0:
        return (x1, y1), 0.0
    if t > 1:
        return (x2, y2), 1.0

    return (int(px), int(py)), t

def get_projection(point, line_start, line_end):
    """
    计算点到线段的投影参数 t 和垂足坐标（不截断）。
    返回: (垂足坐标 (x, y), 参数 t)
    """
    x1, y1 = line_start
    x2, y2 = line_end
    bx, by = point

    dx = x2 - x1
    dy = y2 - y1
    denom = dx * dx + dy * dy
    if denom == 0:
        return (x1, y1), 0.0

    t = ((bx - x1) * dx + (by - y1) * dy) / denom
    px = x1 + t * dx
    py = y1 + t * dy
    return (px, py), t

def frame_enhancement(frame):
    frame_lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(frame_lab)
    enhanced_l = cv2.createCLAHE(clipLimit=5.0, tileGridSize=(8, 8)).apply(l)
    # enhanced_l = cv2.bilateralFilter(l, d=9, sigmaColor=75, sigmaSpace=75)
    enhanced_lab = cv2.merge((enhanced_l, a, b))
    enhanced_frame = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
    return enhanced_frame

def sharpen_edges(frame):
    kernel = np.array([[0, -1, 0],
                       [-1, 5,-1],
                       [0, -1, 0]])
    sharpened_frame = cv2.filter2D(frame, -1, kernel)
    return sharpened_frame

def unsharp_mask(frame, kernel_size=(9, 9), sigma=10.0, strength=1.5):
    frame_float = frame.astype(np.float32)
    blurred = cv2.GaussianBlur(frame_float, kernel_size, sigma)
    detail = frame_float - blurred
    sharpened = frame_float + strength * detail
    sharpened = np.clip(sharpened, 0, 255).astype(np.uint8)
    return sharpened

def point_to_line_distance(point, line_point1, line_point2):
    p = np.array(point)
    n1 = np.array(line_point1)
    n2 = np.array(line_point2)
    if np.linalg.norm(n2 - n1) == 0:
        return np.linalg.norm(p - n1)
    distance = np.abs(np.cross(n2 - n1, n1 - p)) / np.linalg.norm(n2 - n1)
    return distance

def process_detection_result(latest, line_start, line_end, state, kf):
    if latest is None:
        return None

    boxes, scores, cls_ids, res_w, res_h = latest
    centers = []
    boxes_tmp, scores_tmp, cls_ids_tmp = [], [], []
    for box, score, cls_id in zip(boxes, scores, cls_ids):
        box_area = (box[2] - box[0]) * (box[3] - box[1])
        if box_area > 500:
            continue
        x1, y1, x2, y2 = box
        center = ((x1 + x2) // 2, (y1 + y2) // 2)
        centers.append(center)
        boxes_tmp.append(box)
        scores_tmp.append(score)
        cls_ids_tmp.append(cls_id)
    
    boxes, scores, cls_ids = boxes_tmp, scores_tmp, cls_ids_tmp

    # 筛选投影落在线段内的球
    valid_candidates = []
    # ---- 改动开始：在循环中记录对应的 box, score, cls_id ----
    for idx, center in enumerate(centers):
        proj_point, t = get_projection(center, line_start, line_end)
        if 0.0 <= t <= 1.0:   # 垂足在线段内部（包含端点）
            dist = point_to_line_distance(center, line_start, line_end)
            if dist > 20:
                continue
            valid_candidates.append((center, dist, proj_point, t, boxes[idx], scores[idx], cls_ids[idx]))
    # ---- 改动结束 ----

    if valid_candidates:
        # 按垂直距离升序排序
        valid_candidates.sort(key=lambda x: x[1])
        # 取最近的一个
        # ---- 改动开始：从第一个候选中提取 box, score, cls_id ----
        valid_center, _, proj_point, t, box, score, cls_id = valid_candidates[0]
        state["last_position"] = valid_center
        state["lost_frame"] = 0
        boxes = [box]
        scores = [score]
        cls_ids = [cls_id]
        # ---- 改动结束 ----
    else:
        # 无有效球，使用历史位置（或可考虑其他策略）
        valid_center = state["last_position"]
        state["lost_frame"] += 1
        # 对于历史位置，重新计算投影（可能在线段外）
        proj_point, t = get_projection(valid_center, line_start, line_end)
        # ---- 改动开始：无有效候选时返回空列表 ----
        boxes = []
        scores = []
        cls_ids = []
        # ---- 改动结束 ----

    # line_length = math.hypot(line_end[0] - line_start[0], line_end[1] - line_start[1])
    line_length = 1000
    offset_distance = line_length * (t - 0.5)   # 使用真实 t（即使超出范围）
    send_distance = offset_distance

    current_time_kf = time.time()
    dt = current_time_kf - state["last_speed_time"] if current_time_kf - state["last_speed_time"] > 0 else 0.01
    state["last_speed_time"] = current_time_kf

    # 1. 计算瞬时速度 (基于原始位移差分，延迟最低)
    raw_speed = 0.0
    if dt > 0:
        raw_speed = (offset_distance - state["last_raw_offset"]) / dt
    state["last_raw_offset"] = offset_distance  # 更新历史位移

    # 2. 中值滤波 (窗口大小为5，去除由检测框抖动引起的脉冲噪声)
    MEDIAN_WINDOW = 13
    state["speed_buffer"].append(raw_speed)
    if len(state["speed_buffer"]) > MEDIAN_WINDOW:
        state["speed_buffer"].pop(0)
    
    # 计算中值
    median_speed = np.median(state["speed_buffer"])

    # 3. EWMA 滤波 (平滑剩余噪声)
    # Alpha 越大(接近1)，延迟越低但滤波效果越弱；越小越平滑但延迟略增。
    # 建议 0.4~0.6 之间，您可以根据效果调整
    EWMA_ALPHA = 0.15
    state["smoothed_speed"] = EWMA_ALPHA * median_speed + (1 - EWMA_ALPHA) * state["smoothed_speed"]
    
    speed = state["smoothed_speed"]
    
    # Kalman 更新
    # current_time_kf = time.time()
    # dt = current_time_kf - state["last_speed_time"] if current_time_kf - state["last_speed_time"] > 0 else 0.01
    # state["last_speed_time"] = current_time_kf

    # kf.F[0, 1] = dt
    # kf.predict()
    # kf.update(np.array([[offset_distance]]))
    offset_distance_send = int(offset_distance) + 1000
    
    # speed = float(kf.x[1, 0])
    # # speed = (offset_distance - state["last_offset_distance"]) / dt if dt > 0 else 0.0
    # state["last_offset_distance"] = float(kf.x[0, 0])
    # # state["last_offset_distance"] = offset_distance
    # # speed = speed / 10

    
    speed_send = int(speed) + 1000
    state["last_speed"] = speed_send

    # 返回时，proj_point 是真实垂足（可能带浮点，可转为 int 用于显示）
    on_line_position = (int(round(proj_point[0])), int(round(proj_point[1])))

    # ---- 改动开始：调整 sorted_centers 的提取方式 ----
    sorted_centers = [c for c, _, _, _, _, _, _ in valid_candidates] if valid_candidates else []
    # ---- 改动结束 ----

    return boxes, scores, cls_ids, res_w, res_h, valid_center, on_line_position, offset_distance_send, speed_send, sorted_centers

def put_latest(q, item):
    try:
        q.put_nowait(item)
    except Full:
        try:
            q.get_nowait()
        except Empty:
            pass
        try:
            q.put_nowait(item)
        except Full:
            pass


def capture_thread(stop_event=None):
    global line_start, line_end
    if isinstance(CAMERA_SOURCE, str) and "!" in CAMERA_SOURCE:
        cap = cv2.VideoCapture(CAMERA_SOURCE, cv2.CAP_GSTREAMER)
    else:
        cap = cv2.VideoCapture(CAMERA_SOURCE)

    try:
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open camera {CAMERA_SOURCE}")

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAP_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAP_HEIGHT)
        # cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        cap.set(cv2.CAP_PROP_FPS, 120)
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
        cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 3)
        cap.set(cv2.CAP_PROP_EXPOSURE, 30)

        while _should_run(stop_event):
            ok, frame = cap.read()
            if not ok:
                raise RuntimeError("Failed to grab frame")

            pink_center = get_pink_center(frame)
            if pink_center is not None:
                line_end = pink_center

            # 进行对比度增强和锐化处理
            # frame = frame_enhancement(frame)
            # frame = sharpen_edges(frame)
            # frame = unsharp_mask(frame)
            h, w = frame.shape[:2]
            # frame = frame[2 * h // 5: 3 * h // 4, : ]  # 裁剪为中心区域
            # frame = cv2.resize(frame, (CAP_WIDTH, CAP_HEIGHT))  # 调整为指定分辨率

            canvas, scale, pad_w, pad_h = letterbox_rknn(frame, INFER_SIZE)
            rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
            input_tensor = np.expand_dims(rgb, axis=0)

            if SHOW_DISPLAY:
                put_latest(display_q, (frame, w, h))

            put_latest(infer_q, (input_tensor, scale, pad_w, pad_h, w, h))
    finally:
        cap.release()


def infer_thread(stop_event=None):
    rknn = RKNNLite()

    try:
        if rknn.load_rknn(MODEL_PATH) != 0:
            raise RuntimeError("load rknn failed")

        if rknn.init_runtime(core_mask=RKNNLite.NPU_CORE_0) != 0:
            raise RuntimeError("init rknn runtime failed")

        fps_count = 0
        fps_time = time.time()

        while _should_run(stop_event):
            try:
                input_tensor, scale, pad_w, pad_h, w, h = infer_q.get(timeout=0.1)
            except Empty:
                continue

            outputs = rknn.inference(inputs=[input_tensor])

            # 这里统计的是拿到 NPU 推理输出的帧率
            fps_count += 1
            now = time.time()
            if now - fps_time >= 1.0:
                print(f"NPU result fps: {fps_count}")
                fps_count = 0
                fps_time = now

            put_latest(raw_q, (outputs, scale, pad_w, pad_h, w, h))
    finally:
        rknn.release()


def post_thread(stop_event=None):
    while _should_run(stop_event):
        try:
            outputs, scale, pad_w, pad_h, w, h = raw_q.get(timeout=0.1)
        except Empty:
            continue

        boxes, scores, cls_ids = postprocess_rknn(
            outputs,
            scale,
            pad_w,
            pad_h,
            CONF_THRESH,
            IOU_THRESH
        )

        put_latest(result_q, (boxes, scores, cls_ids, w, h))


def draw_boxes(img, boxes, scores, cls_ids, cap_w, cap_h):
    if boxes is None or len(boxes) == 0:
        return img

    h, w = img.shape[:2]
    sx = w / float(cap_w)
    sy = h / float(cap_h)

    for box, score, cls_id in zip(boxes, scores, cls_ids):
        x1, y1, x2, y2 = box

        x1 = int(x1 * sx)
        y1 = int(y1 * sy)
        x2 = int(x2 * sx)
        y2 = int(y2 * sy)

        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

    return img


def _ensure_threads_alive(thread_items, stop_event):
    for name, thread in thread_items:
        if thread.is_alive():
            continue

        stop_event.set()
        raise RuntimeError(f"{name} thread exited unexpectedly")

def main(frame_ready: Value, conn=None, stop_event=None):
    global running

    global line_start, line_end
    last_send_time = 0.0
    frame_ready.value = False

    if stop_event is None:
        stop_event = threading.Event()

    state = {
        "last_position": (0, 0),
        "last_offset_distance": 0.0,
        "last_speed": 0.0,
        "last_speed_time": time.time(),
        "lost_frame": 0,
        "speed_buffer": [],          # 中值滤波历史窗口
        "smoothed_speed": 0.0,       # EWMA平滑后的速度
        "last_raw_offset": 0.0,      # 上一帧的原始位移(用于计算瞬时速度)
    }

    kf = KalmanFilter(dim_x=2, dim_z=1)
    kf.x = np.array([[0.], [0.]])
    kf.F = np.array([[1., 0.01], [0., 1.]])
    kf.H = np.array([[1., 0.]])
    kf.P *= 1000.0
    kf.R = 15
    kf.Q = np.array([[0.1, 0.0], [0.0, 5.0]])

    ct = threading.Thread(target=capture_thread, args=(stop_event,), daemon=True)
    it = threading.Thread(target=infer_thread, args=(stop_event,), daemon=True)
    pt = threading.Thread(target=post_thread, args=(stop_event,), daemon=True)
    worker_threads = [
        ("capture", ct),
        ("infer", it),
        ("post", pt),
    ]

    ct.start()
    it.start()
    pt.start()

    try:
        if SHOW_DISPLAY:
            frame_count = 0
            latest = None
            while _should_run(stop_event):
                _ensure_threads_alive(worker_threads, stop_event)

                try:
                    frame, cap_w, cap_h = display_q.get(timeout=0.03)
                except Empty:
                    continue

                while True:
                    try:
                        latest = result_q.get_nowait()
                    except Empty:
                        break

                disp = cv2.resize(frame, (DISPLAY_WIDTH, DISPLAY_HEIGHT))
                result = process_detection_result(latest, line_start, line_end, state, kf)

                if result is not None:
                    boxes, scores, cls_ids, res_w, res_h, valid_center, on_line_position, offset_distance, speed, centers = result
                    draw_frame = frame.copy()
                    if len(centers) > 0:
                        cv2.circle(draw_frame, valid_center, 5, (0, 255, 0), -1)
                    cv2.circle(draw_frame, on_line_position, 5, (0, 0, 255), -1)
                    cv2.line(draw_frame, line_start, line_end, (255, 255, 0), 2)
                    cv2.circle(draw_frame, line_start, 5, (255, 0, 0), -1)
                    cv2.circle(draw_frame, line_end, 5, (255, 0, 0), -1)

                    if conn is not None:
                        send_message = [0, offset_distance, speed]
                        now = time.time()
                        if now - last_send_time >= 0.01:
                            try:
                                conn.send(send_message)
                                last_send_time = now
                            except (BrokenPipeError, EOFError, OSError):
                                conn = None

                    disp = cv2.resize(draw_frame, (DISPLAY_WIDTH, DISPLAY_HEIGHT))
                    disp = draw_boxes(disp, boxes, scores, cls_ids, res_w, res_h)

                # cv2.imshow("RKNN Async Test", disp)
                # if frame_count > 3:
                #     frame_count = 0
                #     frame_share.write(disp)
                #     frame_ready.value = True
                # else:
                #     frame_count += 1
                frame_share.write(disp)
                frame_ready.value = True

                if cv2.waitKey(1) == 27:
                    stop_event.set()
                    break
        else:
            latest = None
            while _should_run(stop_event):
                _ensure_threads_alive(worker_threads, stop_event)

                while _should_run(stop_event):
                    try:
                        latest = result_q.get_nowait()
                    except Empty:
                        break

                result = process_detection_result(latest, line_start, line_end, state, kf)
                if result is not None and conn is not None:
                    _, _, _, _, _, _, _, offset_distance, speed, _ = result
                    send_message = [0, offset_distance, speed]
                    now = time.time()
                    if now - last_send_time >= 0.002:
                        try:
                            conn.send(send_message)
                            last_send_time = now
                        except (BrokenPipeError, EOFError, OSError):
                            conn = None

                time.sleep(0.002)

    except KeyboardInterrupt:
        frame_share.close()
        stop_event.set()
        pass
    except Exception:
        frame_share.close()
        stop_event.set()
        raise
    finally:
        running = False
        frame_share.close()
        stop_event.set()
        time.sleep(0.3)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    frame_ready = Value('b', False)
    main(frame_ready=frame_ready)
