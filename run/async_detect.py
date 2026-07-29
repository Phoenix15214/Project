import time
import numpy as np
import cv2
import threading
import math
from queue import Queue, Empty, Full
from rknnlite.api import RKNNLite
from multiprocessing import shared_memory, Value, Pipe
from filterpy.kalman import KalmanFilter


# ========================= 配置 =========================
MODEL_PATH = "/home/ubuntu/Project/Project/run/best.rknn"
CAMERA_SOURCE = 0

CAP_WIDTH = 640
CAP_HEIGHT = 480

INFER_SIZE = 256
CONF_THRESH = 0.5
IOU_THRESH = 0.45

SHOW_DISPLAY = True
DISPLAY_WIDTH = 640
DISPLAY_HEIGHT = 480
# =======================================================


infer_q = Queue(maxsize=2)
raw_q = Queue(maxsize=2)
result_q = Queue(maxsize=2)
display_q = Queue(maxsize=2)

running = True


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


def process_detection_result(latest, line_start, line_end, state, kf):
    if latest is None:
        return None

    boxes, scores, cls_ids, res_w, res_h = latest
    centers = []
    for box in boxes:
        x1, y1, x2, y2 = box
        center_x = (x1 + x2) // 2
        center_y = (y1 + y2) // 2
        centers.append((center_x, center_y))

    centers = sorted(centers, key=lambda c: c[1])
    if len(centers) > 0:
        valid_center = centers[0]
        state["last_position"] = valid_center
    else:
        valid_center = state["last_position"]
        state["lost_frame"] += 1

    on_line_position, t = get_on_line_position(valid_center, line_start, line_end)
    line_length = math.hypot(line_end[0] - line_start[0], line_end[1] - line_start[1])
    offset_distance = line_length * (t - 0.5)

    current_time_kf = time.time()
    dt = current_time_kf - state["last_speed_time"] if current_time_kf - state["last_speed_time"] > 0 else 0.01
    state["last_speed_time"] = current_time_kf

    kf.F[0, 1] = dt
    kf.predict()
    kf.update(np.array([[offset_distance]]))

    speed = float(kf.x[1, 0])
    state["last_offset_distance"] = float(kf.x[0, 0])

    offset_distance = int(offset_distance) + 1000
    speed = int(speed) + 1000
    state["last_speed"] = speed

    return boxes, scores, cls_ids, res_w, res_h, valid_center, on_line_position, offset_distance, speed, centers


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


def capture_thread():
    if isinstance(CAMERA_SOURCE, str) and "!" in CAMERA_SOURCE:
        cap = cv2.VideoCapture(CAMERA_SOURCE, cv2.CAP_GSTREAMER)
    else:
        cap = cv2.VideoCapture(CAMERA_SOURCE)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAP_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAP_HEIGHT)
    # cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    cap.set(cv2.CAP_PROP_FPS, 120)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))

    while running:
        ok, frame = cap.read()
        if not ok:
            time.sleep(0.001)
            continue

        h, w = frame.shape[:2]

        canvas, scale, pad_w, pad_h = letterbox_rknn(frame, INFER_SIZE)
        rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
        input_tensor = np.expand_dims(rgb, axis=0)

        if SHOW_DISPLAY:
            put_latest(display_q, (frame, w, h))

        put_latest(infer_q, (input_tensor, scale, pad_w, pad_h, w, h))

    cap.release()


def infer_thread():
    rknn = RKNNLite()

    if rknn.load_rknn(MODEL_PATH) != 0:
        print("load rknn failed")
        return

    if rknn.init_runtime(core_mask=RKNNLite.NPU_CORE_0) != 0:
        print("init rknn runtime failed")
        rknn.release()
        return

    fps_count = 0
    fps_time = time.time()

    while running:
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

    rknn.release()


def post_thread():
    while running:
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

        if SHOW_DISPLAY:
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

def main(conn=None, stop_event=None):
    line_start = (0, 100)
    line_end = (640, 300)
    last_send_time = 0.0

    state = {
        "last_position": (0, 0),
        "last_offset_distance": 0.0,
        "last_speed": 0.0,
        "last_speed_time": time.time(),
        "lost_frame": 0,
    }

    kf = KalmanFilter(dim_x=2, dim_z=1)
    kf.x = np.array([[0.], [0.]])
    kf.F = np.array([[1., 0.01], [0., 1.]])
    kf.H = np.array([[1., 0.]])
    kf.P *= 1000.
    kf.R = 15.0
    kf.Q = np.array([[0.1, 0.0], [0.0, 5.0]])

    ct = threading.Thread(target=capture_thread, daemon=True)
    it = threading.Thread(target=infer_thread, daemon=True)
    pt = threading.Thread(target=post_thread, daemon=True)

    ct.start()
    it.start()
    pt.start()

    try:
        if SHOW_DISPLAY:
            latest = None
            while True:
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

                cv2.imshow("RKNN Async Test", disp)

                if cv2.waitKey(1) == 27:
                    break
        else:
            latest = None
            while True:
                while True:
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
        pass
    finally:
        running = False
        time.sleep(0.3)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
