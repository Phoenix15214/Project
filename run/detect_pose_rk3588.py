"""
YOLO11n-Pose rk3588 NPU inference (RGB Fix Version)
修复：加入 BGR->RGB 转换，解决“全是框”和“分数偏低”问题。
"""
import cv2
import time
import argparse
import numpy as np
from rknnlite.api import RKNNLite

# ── 配置 ───────────────────────────────────────────────────────────────
MODEL_PATH = "best.rknn"
INPUT_SIZE = 640

CLASS_NAMES = {0: 'board', 1: 'white', 2: 'black'}
NUM_CLASSES = len(CLASS_NAMES)
NUM_KPTS = 13
KPT_DIM = NUM_KPTS * 3
BASE_STRIDE = 4 + NUM_CLASSES + KPT_DIM 

CLASS_COLORS = {0: (255, 0, 0), 1: (0, 255, 0), 2: (0, 0, 255)}
KPT_COLORS = {0: (0, 165, 255), 1: (0, 255, 255), 2: (255, 0, 255)}
BOARD_KPT_NAMES = [
    'zuoshang', 'zuoxia', 'youshang', 'youxia',
    '1', '2', '3', '4', '5', '6', '7', '8', '9'
]

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# ── 1. 加载模型 ───────────────────────────────────────────────────────
def load_rknn_model(model_path):
    rknn = RKNNLite()
    ret = rknn.load_rknn(model_path)
    if ret != 0: raise RuntimeError(f"load_rknn failed, ret={ret}")
    ret = rknn.init_runtime(core_mask=RKNNLite.NPU_CORE_AUTO)
    if ret != 0: raise RuntimeError(f"init_runtime failed, ret={ret}")
    print(f"[RKNN] model loaded: {model_path}")
    return rknn

# ── 2. 预处理 (关键修改：BGR -> RGB) ─────────────────────────────────────
def preprocess(frame, input_size):
    h0, w0 = frame.shape[:2]
    scale = min(input_size / w0, input_size / h0)
    new_w, new_h = int(w0 * scale), int(h0 * scale)
    
    # Resize
    resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    # Pad
    pad_w = input_size - new_w
    pad_h = input_size - new_h
    pad_left = pad_w // 2
    pad_top = pad_h // 2

    padded = cv2.copyMakeBorder(
        resized, pad_top, pad_h - pad_top,
        pad_left, pad_w - pad_left,
        cv2.BORDER_CONSTANT, value=(114, 114, 114))

    # --- [关键修改] 转换颜色空间 BGR -> RGB ---
    # YOLO 模型需要 RGB 输入，而 OpenCV 是 BGR
    input_rgb = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)
    
    # 转换为 uint8 (0-255) 传给 RKNN
    input_data = input_rgb.astype(np.uint8)
    input_data = np.expand_dims(input_data, axis=0)
    
    return input_data, pad_left, pad_top, scale

# ── 3. 推理 ───────────────────────────────────────────────────────────
def inference(rknn, input_data):
    return rknn.inference(inputs=[input_data])

# ── 4. 后处理 ───────────────────────────────────────────────────────────
def nms(boxes, scores, iou_thres):
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        if order.size == 1: break
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h
        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-6)
        inds = np.where(iou <= iou_thres)[0]
        order = order[inds + 1]
    return np.array(keep, dtype=np.int32)

def postprocess(outputs, frame_shape, pad_left, pad_top, scale,
                input_size, conf_thres=0.5, iou_thres=0.6):
    h0, w0 = frame_shape[:2]
    pred = outputs[0]

    if pred.shape[1] == BASE_STRIDE or pred.shape[1] == BASE_STRIDE + 1:
        pred = pred[0].T
    else:
        pred = pred[0]

    output_channels = pred.shape[1]
    has_objectness = (output_channels == BASE_STRIDE + 1)
    
    # 索引计算
    idx_cls_start = 4 + (1 if has_objectness else 0)
    idx_cls_end = idx_cls_start + NUM_CLASSES
    idx_kpt_start = idx_cls_end

    all_boxes_raw, all_confs, all_cls, all_kpts_raw = [], [], [], []

    for row in pred:
        box = row[0:4]
        cx, cy, bw, bh = box

        # Class scores
        cls_logits = row[idx_cls_start : idx_cls_end]
        cls_probs = sigmoid(cls_logits) # Logits -> Prob
        max_cls_conf = cls_probs.max()
        cls_id = int(cls_probs.argmax())

        conf = max_cls_conf
        if has_objectness:
            conf *= sigmoid(row[4])

        if conf < conf_thres:
            continue

        # Box Decode
        x1 = cx - bw / 2
        y1 = cy - bh / 2 
        x2 = cx + bw / 2
        y2 = cy + bh / 2

        all_boxes_raw.append([x1, y1, x2, y2])
        all_confs.append(conf)
        all_cls.append(cls_id)

        # Keypoints
        kpt_vals = row[idx_kpt_start : idx_kpt_start + KPT_DIM].reshape(NUM_KPTS, 3)
        kpt_vals[:, 2] = sigmoid(kpt_vals[:, 2]) 
        all_kpts_raw.append(kpt_vals)

    if not all_boxes_raw: return []

    # NMS
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
            
            # Map back to original image
            x1 = int(max(0, min(w0, (bbox_raw[0] - pad_left) / scale)))
            y1 = int(max(0, min(h0, (bbox_raw[1] - pad_top) / scale)))
            x2 = int(max(0, min(w0, (bbox_raw[2] - pad_left) / scale)))
            y2 = int(max(0, min(h0, (bbox_raw[3] - pad_top) / scale)))

            kpts = []
            for kp in all_kpts_raw[i]:
                kx = (kp[0] - pad_left) / scale
                ky = (kp[1] - pad_top) / scale
                kpts.append([float(kx), float(ky), float(kp[2])])

            detections.append({'bbox': [x1, y1, x2, y2], 'cls': int(cls_id), 'conf': float(all_confs[i]), 'kpts': kpts})
            
    return detections

# ── 5. 可视化 ─────────────────────────────────────────────────────────
def draw_pose_results(img, detections):
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
        cv2.putText(img, label, (x1 + 2, y1 - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        kpt_color = KPT_COLORS.get(cls_id, (255, 255, 255))
        for k, (kx, ky, kv) in enumerate(kpts):
            if 0 <= kx < w and 0 <= ky < h and kv > 0.5:
                cv2.circle(img, (int(kx), int(ky)), 6, kpt_color, -1)
                cv2.circle(img, (int(kx), int(ky)), 7, (255, 255, 255), 2)
                if cls_id == 0 and k < len(BOARD_KPT_NAMES):
                    cv2.putText(img, BOARD_KPT_NAMES[k], (int(kx)+8, int(ky)-8),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45, kpt_color, 1, cv2.LINE_AA)
    return img

# ── 6. 主循环 ─────────────────────────────────────────────────────────
def setup_camera(camera_id, width, height):
    cap = cv2.VideoCapture(camera_id)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M','J','P','G'))
    if not cap.isOpened(): raise RuntimeError(f"Cannot open camera {camera_id}")
    print(f"[Camera] {int(cap.get(3))}x{int(cap.get(4))}")
    return cap

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default=MODEL_PATH)
    parser.add_argument('--camera', type=int, default=0)
    parser.add_argument('--conf', type=float, default=0.5) # 建议在运行时调高到 0.6 或 0.7
    parser.add_argument('--iou', type=float, default=0.6)
    parser.add_argument('--input-size', type=int, default=INPUT_SIZE)
    args = parser.parse_args()

    rknn = load_rknn_model(args.model)
    cap = setup_camera(args.camera, 1280, 720)

    cv2.namedWindow('RKNN', cv2.WINDOW_NORMAL)
    # 初始阈值设高一点，防止噪点
    cv2.createTrackbar('Conf', 'RKNN', 60, 100, lambda x: None) 

    while True:
        ret, frame = cap.read()
        if not ret: break
        
        # 注意：这里传给 preprocess 的是原始 BGR frame
        input_data, pad_l, pad_t, scale = preprocess(frame, args.input_size)
        outputs = inference(rknn, input_data)
        
        # 动态获取阈值
        conf_val = cv2.getTrackbarPos('Conf', 'RKNN') / 100.0
        
        detections = postprocess(outputs, frame.shape, pad_l, pad_t, scale,
                                 args.input_size, conf_val, args.iou)
        
        annotated = draw_pose_results(frame, detections)
        cv2.putText(annotated, f"FPS: ... | Objs: {len(detections)}", (10, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
        
        cv2.imshow('RKNN', annotated)
        if cv2.waitKey(1) & 0xFF == ord('q'): break

    cap.release()
    cv2.destroyAllWindows()
    rknn.release()

if __name__ == "__main__":
    main()
