import os
os.environ.setdefault('QT_QPA_FONTDIR', '/usr/share/fonts/truetype/dejavu')

import cv2
import numpy as np
import process_lib.control_lib as ctrl
import process_lib.image_lib as lb
from multiprocessing import Process, Pipe, shared_memory, Value
from threading import Thread
import time
import onnxruntime as ort

# MODEL_PATH = "/home/pi/Project/run/best_int8.onnx"
MODEL_PATH = "/home/pi/Project/run/best.onnx"
NUM_CLASSES = 8
CONF_THRESH = 0.5
IOU_THRESH = 0.45
TARGET_SIZE = (224, 224)
IMG_SIZE = 640
CAMERA_FPS = 30

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

def open_camera(max_index=4, width=640, height=480, fps=CAMERA_FPS):
    for idx in range(max_index + 1):
        cap = cv2.VideoCapture(idx, cv2.CAP_V4L2)
        if not cap.isOpened(): continue
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        cap.set(cv2.CAP_PROP_FPS, fps)
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        if cap.read()[0]: return cap, idx
        cap.set(cv2.CAP_PROP_FOURCC, 0)
        if cap.read()[0]: return cap, idx
        cap.release()
    return None, -1

def main(shm_name, frame_ready, yolo_start=None, conn=None, stop_event=None, core=None):
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 2   # 树莓派4核，留1核给系统
    opts.inter_op_num_threads = 2
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    
    pack = ctrl.SerialPacket(port="/dev/ttyUSB0", baudrate=38400, timeout=0.1)if conn is None else None
    stop_event = stop_event or type("StopEvent", (), {"is_set": staticmethod(lambda: False)})()
    if shm_name is not None:
        try:
            shm = shared_memory.SharedMemory(name=shm_name, create=False)
        except Exception as e:
            print(f"Failed to access shared memory: {e}")
            return
    frame_view = np.ndarray((480, 640, 3), dtype=np.uint8, buffer=shm.buf)if shm_name is not None else None

    print("Loading ONNX model...")
    session = ort.InferenceSession(MODEL_PATH, sess_options=opts, providers=['CPUExecutionProvider'])
    inp = session.get_inputs()[0]
    input_name = inp.name
    output_name = session.get_outputs()[0].name
    dummy = np.zeros((1, 3, 224, 224), dtype=np.float32)
    session.run([output_name], {input_name: dummy})

    last_time = time.time()
    current_time = time.time()
    fps = 0
    frame_count = 0
    cap = None
    if frame_view is None:
        cap, _ = open_camera(fps=CAMERA_FPS)
    ret, frame = cap.read()if cap is not None else (True, frame_view.copy())
    if not ret:
        print('Failed to open camera.'); return

    try:
        while not stop_event.is_set():
            if frame_view is not None:
                if frame_ready.value:
                    frame = frame_view.copy()
                    frame_ready.value = False
                else:
                    time.sleep(0.02)
                    continue
            else:
                ok, frame = cap.read()
                if not ok:
                    break
            valid_targets = [(9999, 100, 0), (9999, 100, 0), (9999, 100, 0), (9999, 100, 0)]

            if yolo_start is None or yolo_start.value == True:
                # 预处理
                frame = cv2.GaussianBlur(frame, (5, 5), 0)
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                frame = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(frame)
                frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
                img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                # img_rgb = frame
                img_lb, scale, pad = letterbox(img_rgb, TARGET_SIZE)
                img_data = img_lb.astype(np.float32) / 255.0
                img_data = img_data.transpose(2, 0, 1)[np.newaxis, :]  # [1, 3, H, W]

                # 推理
                output = session.run([output_name], {input_name: img_data})[0]

                # 后处理
                boxes, scores, class_ids = postprocess(output, CONF_THRESH, IOU_THRESH, NUM_CLASSES, scale, pad)

                index = 0
                
                if len(boxes) > 0:
                    for (x1, y1, x2, y2), score, cls_id in zip(boxes, scores, class_ids):
                        if index < 4:
                            valid_targets[index] = (x1, cls_id, x1)
                            index += 1
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        label = f"{cls_id}: {score:.2f}"
                        # print(f"Detected {label} at [{x1}, {y1}, {x2}, {y2}]")
                        cv2.putText(frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

            if pack is not None:
                pack.insert_byte(0x08)  # 包头
            msg = [1, 0, 0, 0, 0, 0, 0, 0, 0]  # [包头, 4个目标类别, 4个目标位置（x1坐标）]
            send_index = 1
            if valid_targets:
                valid_targets = sorted(valid_targets, key=lambda x: x[0])
                for i in valid_targets:
                    if pack is not None:
                        if i[1] == 100:  # 无效目标
                            pack.insert_two_bytes(pack.num_to_bytes(0)) # 发送 0 表示无目标
                        else:
                            pack.insert_two_bytes(pack.num_to_bytes(i[1] + 1))
                    msg[send_index] = i[1] + 1 if i[1] != 100 else 0
                    msg[send_index + 4] = i[2] if i[1] != 100 else 0
                    send_index += 1

            if conn is not None:
                try:
                    conn.send(msg)
                except (BrokenPipeError, EOFError, OSError):
                    break
            if pack is not None:
                pack.send_packet()  # 发送数据包
            frame_count += 1
            current_time = time.time()
            if current_time - last_time > 1:
                fps = frame_count / (current_time - last_time)
                last_time = current_time
                frame_count = 0
                # print(f'FPS: {fps:.2f}')
            # cv2.imshow('Detect', frame)
            # cv2.imshow('IMG', img[7]) 
            # cv2.imshow('Black', black)
            # cv2.imshow('Warped', warped)
            # cv2.imshow('Contours', draw_img)
            # cv2.imshow("frame", frame)
            # if cv2.waitKey(1) & 0xFF in (27, ord('q')):
            #     break
    finally:
        if shm_name is not None:
            shm.close()
        if cap is not None:
            cap.release()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main(None, Value('b', False))
