import cv2
import numpy as np
import onnxruntime as ort
import time

# ===================== Config =====================
MODEL_PATH = "/home/pi/Project/run/best_int8.onnx"
NUM_CLASSES = 8
CONF_THRESH = 0.4
IOU_THRESH = 0.45
TARGET_SIZE = (224, 224)
# ==================================================

def letterbox(img, target_size):
    """保持长宽比缩放，并填充灰边(114)"""
    h, w = img.shape[:2]
    tw, th = target_size
    scale = min(tw / w, th / h)
    nw, nh = int(w * scale), int(h * scale)
    img_resized = cv2.resize(img, (nw, nh))
    canvas = np.full((th, tw, 3), 114, dtype=np.uint8)
    top = (th - nh) // 2
    left = (tw - nw) // 2
    canvas[top:top+nh, left:left+nw] = img_resized
    return canvas, scale, (left, top)

def nms(boxes, scores, iou_thresh):
    """NMS算法"""
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        inter = np.maximum(0, xx2-xx1) * np.maximum(0, yy2-yy1)
        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-6)
        order = order[np.where(iou <= iou_thresh)[0] + 1]
    return keep

def postprocess(output, conf_thresh, iou_thresh, num_classes, scale, pad):
    pred = output.squeeze(0).T  # [1, 12, 8400] -> [8400, 12]
    
    boxes = pred[:, :4]          # xywh (已经是 0-224 的绝对坐标)
    class_scores = pred[:, 4:4+num_classes]
    scores = np.max(class_scores, axis=1)
    class_ids = np.argmax(class_scores, axis=1)

    mask = scores > conf_thresh
    boxes, scores, class_ids = boxes[mask], scores[mask], class_ids[mask]
    if len(boxes) == 0:
        return np.array([]), np.array([]), np.array([])

    # 【修复核心】xywh(绝对坐标) -> xyxy(绝对坐标)，不再乘以 640！
    x1 = boxes[:, 0] - boxes[:, 2] / 2
    y1 = boxes[:, 1] - boxes[:, 3] / 2
    x2 = boxes[:, 0] + boxes[:, 2] / 2
    y2 = boxes[:, 1] + boxes[:, 3] / 2
    boxes_xyxy = np.stack([x1, y1, x2, y2], axis=1)

    keep = nms(boxes_xyxy, scores, iou_thresh)
    boxes_xyxy, scores, class_ids = boxes_xyxy[keep], scores[keep], class_ids[keep]

    # 映射回原图坐标 (减去 pad，除以 scale)
    pad_left, pad_top = pad
    boxes_xyxy[:, [0, 2]] = (boxes_xyxy[:, [0, 2]] - pad_left) / scale
    boxes_xyxy[:, [1, 3]] = (boxes_xyxy[:, [1, 3]] - pad_top) / scale

    return boxes_xyxy.astype(int), scores, class_ids

def main():
    # Session配置优化
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 3   # 树莓派4核，留1核给系统
    opts.inter_op_num_threads = 3
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    
    print("Loading ONNX model...")
    session = ort.InferenceSession(MODEL_PATH, sess_options=opts, providers=['CPUExecutionProvider'])

    inp = session.get_inputs()[0]
    input_name = inp.name
    output_name = session.get_outputs()[0].name

    print("Initializing camera...")
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    # 预热
    dummy = np.zeros((1, 3, 224, 224), dtype=np.float32)
    session.run([output_name], {input_name: dummy})

    print("Starting inference... Press 'q' to quit.")
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        t0 = time.time()
        # 预处理 (必须 BGR2RGB + Letterbox)
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img_lb, scale, pad = letterbox(img_rgb, TARGET_SIZE)
        img_data = img_lb.astype(np.float32) / 255.0
        img_data = img_data.transpose(2, 0, 1)[np.newaxis, :]  # [1, 3, H, W]

        # 推理
        output = session.run([output_name], {input_name: img_data})[0]

        # 后处理
        boxes, scores, class_ids = postprocess(output, CONF_THRESH, IOU_THRESH, NUM_CLASSES, scale, pad)

        # 画框
        if len(boxes) > 0:
            for (x1, y1, x2, y2), score, cls_id in zip(boxes, scores, class_ids):
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                label = f"{cls_id}: {score:.2f}"
                print(f"Detected {label} at [{x1}, {y1}, {x2}, {y2}]")
                cv2.putText(frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        dt = time.time() - t0
        fps = 1.0 / dt if dt > 0 else 0
        cv2.putText(frame, f"FPS: {fps:.1f}  Infer: {dt*1000:.0f}ms",
                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        # cv2.imshow("YOLO11n INT8 - ONNX Runtime", frame)

        # if cv2.waitKey(1) & 0xFF == ord('q'):
        #     break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
