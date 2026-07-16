"""
YOLO11n-Pose rk3588 NPU inference (RKNN Lite)
"""
import cv2
import time
import argparse
import numpy as np
from rknnlite.api import RKNNLite


# ── 模型 / 类别 / 可视化配置 ──────────────────────────────────────────
MODEL_PATH = "best.rknn"
INPUT_SIZE = 640

CLASS_NAMES = {0: 'board', 1: 'white', 2: 'black'}
NUM_CLASSES = len(CLASS_NAMES)
NUM_KPTS = 13
KPT_DIM = NUM_KPTS * 3
STRIDE = 4 + 1 + NUM_CLASSES + KPT_DIM  # cx,cy,w,h + obj + cls * 3 + kpt * 13 * 3

CLASS_COLORS = {0: (255, 0, 0), 1: (0, 255, 0), 2: (0, 0, 255)}
KPT_COLORS = {0: (0, 165, 255), 1: (0, 255, 255), 2: (255, 0, 255)}
BOARD_KPT_NAMES = [
    'zuoshang', 'zuoxia', 'youshang', 'youxia',
    '1', '2', '3', '4', '5', '6', '7', '8', '9'
]


# ── 1. RKNN 模型加载 ─────────────────────────────────────────────────
def load_rknn_model(model_path):
    """
    加载 RKNN 模型并初始化 NPU 运行时
    返回: rknn
    """
    rknn = RKNNLite()
    ret = rknn.load_rknn(model_path)
    if ret != 0:
        raise RuntimeError(f"load_rknn failed, ret={ret}")

    ret = rknn.init_runtime(core_mask=RKNNLite.NPU_CORE_AUTO)
    if ret != 0:
        raise RuntimeError(f"init_runtime failed, ret={ret}")

    print(f"[RKNN] model loaded: {model_path}")
    # 若想打印固定尺寸，可直接写死：
    # print(f"[RKNN] input size: {INPUT_SIZE} x {INPUT_SIZE}")
    return rknn

# ── 2. 图像预处理 ────────────────────────────────────────────────────
def preprocess(frame, input_size):
    """
    将 BGR 帧 resize + 归一化, 转为模型输入 format (NHWC, uint8/float32)
    返回: (input_data, pad_left, pad_top, scale)
    """
    h0, w0 = frame.shape[:2]

    scale = min(input_size / w0, input_size / h0)
    new_w, new_h = int(w0 * scale), int(h0 * scale)
    resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    pad_w = input_size - new_w
    pad_h = input_size - new_h
    pad_left = pad_w // 2
    pad_top = pad_h // 2

    padded = cv2.copyMakeBorder(
        resized, pad_top, pad_h - pad_top,
        pad_left, pad_w - pad_left,
        cv2.BORDER_CONSTANT, value=(114, 114, 114))

    input_data = padded.astype(np.float32) / 255.0
    input_data = np.expand_dims(input_data, axis=0)  # (1, H, W, 3)

    return input_data, pad_left, pad_top, scale


# ── 3. NPU 推理 ──────────────────────────────────────────────────────
def inference(rknn, input_data):
    """
    执行一次 NPU 推理
    返回: outputs (list of np.ndarray)
    """
    outputs = rknn.inference(inputs=[input_data])
    return outputs


# ── 4. 后处理 ────────────────────────────────────────────────────────
def nms(boxes, scores, iou_thres):
    """单类 NMS, 返回保留的索引"""
    order = np.argsort(scores)[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        if order.size == 1:
            break
        xx1 = np.maximum(boxes[i, 0], boxes[order[1:], 0])
        yy1 = np.maximum(boxes[i, 1], boxes[order[1:], 1])
        xx2 = np.minimum(boxes[i, 2], boxes[order[1:], 2])
        yy2 = np.minimum(boxes[i, 3], boxes[order[1:], 3])
        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h
        area_i = (boxes[i, 2] - boxes[i, 0]) * (boxes[i, 3] - boxes[i, 1])
        area_o = (boxes[order[1:], 2] - boxes[order[1:], 0]) * (boxes[order[1:], 3] - boxes[order[1:], 1])
        iou = inter / np.maximum(area_i + area_o - inter, 1e-6)
        order = order[1 + np.where(iou <= iou_thres)[0]]
    return np.array(keep, dtype=np.int32)


def postprocess(outputs, frame_shape, pad_left, pad_top, scale,
                input_size, conf_thres=0.5, iou_thres=0.6):
    """
    解析 RKNN 输出 → 检测结果列表
    返回: detections = [{'bbox': [x1,y1,x2,y2], 'cls': int, 'conf': float, 'kpts': [[x,y,v], ...]}, ...]
    """
    h0, w0 = frame_shape[:2]
    pred = outputs[0]  # shape: (1, STRIDE, num_proposals) 或 (1, num_proposals, STRIDE)

    if pred.shape[1] == STRIDE:
        pred = pred[0].T  # → (num_proposals, STRIDE)
    else:
        pred = pred[0]

    all_boxes_raw = []  # xyxy (在 640 空间)
    all_confs = []
    all_cls = []
    all_kpts_raw = []

    for row in pred:
        cx, cy, bw, bh = row[0:4]
        obj_conf = row[4]
        cls_scores = row[5:5 + NUM_CLASSES]

        max_cls_conf = cls_scores.max()
        cls_id = int(cls_scores.argmax())
        conf = obj_conf * max_cls_conf

        if conf < conf_thres:
            continue

        x1 = cx - bw / 2
        y1 = cy - bh / 2
        x2 = cx + bw / 2
        y2 = cy + bh / 2

        all_boxes_raw.append([x1, y1, x2, y2])
        all_confs.append(conf)
        all_cls.append(cls_id)

        # 关键点: (NUM_KPTS * 3) 个值
        kpt_start = 5 + NUM_CLASSES
        kpt_vals = row[kpt_start:kpt_start + KPT_DIM].reshape(NUM_KPTS, 3)
        all_kpts_raw.append(kpt_vals)

    if not all_boxes_raw:
        return []

    all_boxes_raw = np.array(all_boxes_raw, dtype=np.float32)
    all_confs = np.array(all_confs, dtype=np.float32)
    all_cls = np.array(all_cls, dtype=np.int32)
    all_kpts_raw = np.array(all_kpts_raw, dtype=np.float32)

    detections = []
    for cls_id in np.unique(all_cls):
        idx = np.where(all_cls == cls_id)[0]
        keep = nms(all_boxes_raw[idx], all_confs[idx], iou_thres)
        for k in keep:
            i = idx[k]
            bbox_raw = all_boxes_raw[i]

            # 映射回原图坐标
            x1 = (bbox_raw[0] - pad_left) / scale
            y1 = (bbox_raw[1] - pad_top) / scale
            x2 = (bbox_raw[2] - pad_left) / scale
            y2 = (bbox_raw[3] - pad_top) / scale

            x1 = max(0, min(w0, x1))
            y1 = max(0, min(h0, y1))
            x2 = max(0, min(w0, x2))
            y2 = max(0, min(h0, y2))

            # 关键点映射
            kpts = []
            for kp in all_kpts_raw[i]:
                kx = (kp[0] - pad_left) / scale
                ky = (kp[1] - pad_top) / scale
                kv = kp[2]
                kpts.append([float(kx), float(ky), float(kv)])

            detections.append({
                'bbox': [int(x1), int(y1), int(x2), int(y2)],
                'cls': int(cls_id),
                'conf': float(all_confs[i]),
                'kpts': kpts,
            })

    return detections


# ── 5. 可视化绘制 ────────────────────────────────────────────────────
def draw_pose_results(img, detections):
    """在图像上绘制检测框、标签和关键点"""
    h, w = img.shape[:2]

    for det in detections:
        cls_id = det['cls']
        conf = det['conf']
        x1, y1, x2, y2 = det['bbox']
        kpts = det['kpts']

        color = CLASS_COLORS.get(cls_id, (255, 255, 255))
        name = CLASS_NAMES.get(cls_id, str(cls_id))

        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

        label = f"{name} {conf:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(img, (x1, y1 - th - 4), (x1 + tw + 4, y1), color, -1)
        cv2.putText(img, label, (x1 + 2, y1 - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        kpt_color = KPT_COLORS.get(cls_id, (255, 255, 255))
        for k, (kx, ky, kv) in enumerate(kpts):
            if 0 < kx < w and 0 < ky < h:
                if kv > 0:
                    cv2.circle(img, (int(kx), int(ky)), 6, kpt_color, -1)
                    cv2.circle(img, (int(kx), int(ky)), 7, (255, 255, 255), 2)
                else:
                    cv2.circle(img, (int(kx), int(ky)), 5, (128, 128, 128), -1)
                    cv2.circle(img, (int(kx), int(ky)), 6, (200, 200, 200), 1)

                if cls_id == 0 and k < len(BOARD_KPT_NAMES):
                    cv2.putText(img, BOARD_KPT_NAMES[k],
                                (int(kx) + 8, int(ky) - 8),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45, kpt_color, 1, cv2.LINE_AA)

    return img


# ── 6. 摄像头初始化 ──────────────────────────────────────────────────
def setup_camera(camera_id, width, height):
    """打开摄像头并设置分辨率, 返回 cv2.VideoCapture"""
    cap = cv2.VideoCapture(camera_id)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M','J','P','G'))

    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera {camera_id}")

    real_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    real_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"[Camera] device={camera_id}, resolution={real_w}x{real_h}")
    return cap, real_w, real_h


# ── 7. 主循环 ────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description='YOLO11n-Pose rk3588 NPU Inference')
    parser.add_argument('--model', default=MODEL_PATH, help='RKNN model path')
    parser.add_argument('--camera', type=int, default=0, help='Camera device ID')
    parser.add_argument('--conf', type=float, default=0.5, help='Confidence threshold')
    parser.add_argument('--iou', type=float, default=0.6, help='IoU threshold for NMS')
    parser.add_argument('--width', type=int, default=1280, help='Camera width')
    parser.add_argument('--height', type=int, default=720, help='Camera height')
    parser.add_argument('--input-size', type=int, default=INPUT_SIZE, help='Model input size')
    args = parser.parse_args()

    input_size = args.input_size

    rknn = load_rknn_model(args.model)
    cap, cam_w, cam_h = setup_camera(args.camera, args.width, args.height)

    cv2.namedWindow('YOLO11n-Pose RK3588', cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
    cv2.createTrackbar('Conf', 'YOLO11n-Pose RK3588', int(args.conf * 100), 100, lambda x: None)
    cv2.resizeWindow('YOLO11n-Pose RK3588', 1280, 720)

    fps = 0.0
    prev_time = time.time()

    print("[Main] press 'q' to quit")
    while True:
        ret, frame = cap.read()
        if not ret:
            print("[Main] camera read failed, exiting...")
            break

        frame_copy = frame.copy()
        conf_val = cv2.getTrackbarPos('Conf', 'YOLO11n-Pose RK3588') / 100.0

        start = time.time()

        input_data, pad_left, pad_top, scale = preprocess(frame_copy, input_size)
        outputs = inference(rknn, input_data)
        detections = postprocess(outputs, frame.shape,
                                 pad_left, pad_top, scale,
                                 input_size, conf_val, args.iou)
        annotated = draw_pose_results(frame_copy, detections)

        elapsed = time.time() - start
        fps = 0.9 * fps + 0.1 / elapsed if elapsed > 0 else fps

        cv2.putText(annotated, f"FPS: {fps:.1f}", (10, cam_h - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        cv2.imshow('YOLO11n-Pose RK3588', annotated)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    rknn.release()


if __name__ == "__main__":
    main()
