import os
os.environ.setdefault('QT_QPA_FONTDIR', '/usr/share/fonts/truetype/dejavu')

import cv2
import numpy as np
from rknnlite.api import RKNNLite
import process_lib.control_lib as ctrl
from multiprocessing import Process, Pipe, shared_memory, Value
from threading import Thread
import time

RKNN_MODEL = 'best.rknn'
IMG_SIZE = 640
CONF_THRESH = 0.5
IOU_THRESH = 0.45
CAMERA_FPS = 30

def nms(boxes, scores, iou_thresh):
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = (x2 - x1).clip(min=0) * (y2 - y1).clip(min=0)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        if order.size == 1: break
        xx1, yy1 = np.maximum(x1[i], x1[order[1:]]), np.maximum(y1[i], y1[order[1:]])
        xx2, yy2 = np.minimum(x2[i], x2[order[1:]]), np.minimum(y2[i], y2[order[1:]])
        inter = (xx2 - xx1).clip(min=0) * (yy2 - yy1).clip(min=0)
        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-6)
        order = order[1:][iou < iou_thresh]
    return keep

def letterbox(img, new_shape=640, color=(114, 114, 114)):
    h, w = img.shape[:2]
    scale = min(new_shape / h, new_shape / w)
    nw, nh = int(w * scale), int(h * scale)
    canvas = np.full((new_shape, new_shape, 3), color, dtype=np.uint8)
    pad_w, pad_h = (new_shape - nw) // 2, (new_shape - nh) // 2
    canvas[pad_h:pad_h + nh, pad_w:pad_w + nw] = cv2.resize(img, (nw, nh))
    return canvas, scale, pad_w, pad_h

def postprocess(outputs, scale, pad_w, pad_h):
    # 1. 专门解析 YOLOv8 格式: (1, C+4, 8400) -> (8400, C+4)
    pred = outputs[0][0].T 
    
    boxes_xywh = pred[:, :4]
    class_scores = pred[:, 4:]
    
    # YOLOv8 没有独立的 obj 置信度，直接取最大类别分数
    scores = class_scores.max(axis=1)
    class_ids = class_scores.argmax(axis=1)
    
    # 2. 置信度初步过滤
    mask = scores > CONF_THRESH
    boxes_xywh, scores, class_ids = boxes_xywh[mask], scores[mask], class_ids[mask]
    if len(boxes_xywh) == 0: return [], [], []
    
    # 3. xywh 中心坐标 -> xyxy 左上右下坐标
    boxes_xyxy = np.zeros_like(boxes_xywh)
    boxes_xyxy[:, 0] = boxes_xywh[:, 0] - boxes_xywh[:, 2] / 2
    boxes_xyxy[:, 1] = boxes_xywh[:, 1] - boxes_xywh[:, 3] / 2
    boxes_xyxy[:, 2] = boxes_xywh[:, 0] + boxes_xywh[:, 2] / 2
    boxes_xyxy[:, 3] = boxes_xywh[:, 1] + boxes_xywh[:, 3] / 2
    
    # 4. 去掉 letterbox 的黑边填充和缩放，还原到原图坐标
    boxes_xyxy[:, [0, 2]] = (boxes_xyxy[:, [0, 2]] - pad_w) / scale
    boxes_xyxy[:, [1, 3]] = (boxes_xyxy[:, [1, 3]] - pad_h) / scale
    
    # 5. NMS 去重
    keep = nms(boxes_xyxy, scores, IOU_THRESH)
    return boxes_xyxy[keep].astype(np.int32), scores[keep], class_ids[keep]

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

def main(shm_name, frame_ready, conn=None):
    pack = ctrl.SerialPacket(port="/dev/ttyUSB0", baudrate=38400, timeout=0.1)if conn is None else None
    if shm_name is not None:
        try:
            shm = shared_memory.SharedMemory(name=shm_name, create=False)
        except Exception as e:
            print(f"Failed to access shared memory: {e}")
            return
    frame_view = np.ndarray((480, 640, 3), dtype=np.uint8, buffer=shm.buf)if shm_name is not None else None

    last_time = time.time()
    current_time = time.time()
    fps = 0
    rknn = RKNNLite()
    if rknn.load_rknn(RKNN_MODEL) != 0: return
    # 兼容不同版本的 rknn-lite API
    core_mask = getattr(RKNNLite, 'NPU_CORE_0_1_2', RKNNLite.NPU_CORE_0)
    if rknn.init_runtime(core_mask=core_mask) != 0:
        rknn.release(); return
    
    cap = None
    if frame_view is None:
        cap, _ = open_camera(fps=CAMERA_FPS)
    ret, frame = cap.read()if cap is not None else (True, frame_view.copy())
    if not ret:
        print('Failed to open camera.'); rknn.release(); return

    while True:
        if frame_view is not None:
            if frame_ready.value:
                frame = frame_view.copy()
                frame_ready.value = False
            else:
                time.sleep(0.003)
                continue
        else:
            ok, frame = cap.read()
            if not ok: break

        img = cv2.GaussianBlur(frame, (7, 7), 0)
        
        img, scale, pad_w, pad_h = letterbox(frame, IMG_SIZE)

        input_tensor = np.expand_dims(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), axis=0)
        
        outputs = rknn.inference(inputs=[input_tensor])
        
        boxes, scores, cls_ids = postprocess(outputs, scale, pad_w, pad_h)
        valid_targets = [(9999, 100), (9999, 100), (9999, 100), (9999, 100)]
        index = 0
        for box, score, cls_id in zip(boxes, scores, cls_ids):
            x1, y1, x2, y2 = box.tolist()
            if index < 4:
                valid_targets[index] = (x1, cls_id)
                index += 1
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f'id:{int(cls_id)} {score:.2f}', (x1, max(y1 - 6, 0)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)
        pack.insert_byte(0x08)if pack is not None else None  # 包头
        msg = [1, 0, 0, 0, 0]
        send_index = 1
        if valid_targets:
            valid_targets = sorted(valid_targets, key=lambda x: x[0])
            for i in valid_targets:
                pack.insert_two_bytes(pack.num_to_bytes(0 if i[1] == 100 else i[1] + 1))if pack is not None else None
                msg[send_index] = 0 if i[1] == 100 else i[1] + 1
                send_index += 1
            # print(msg)
        conn.send(msg) if conn is not None else None
        pack.send_packet() if pack is not None else None  # 发送数据包
        # current_time = time.time()
        # fps = 1 / (current_time - last_time)
        # last_time = current_time
        # print(f'FPS: {fps:.2f}')
        # cv2.imshow('RK3588 YOLO', frame)
        # if cv2.waitKey(1) & 0xFF in (27, ord('q')): break

    if shm_name is not None:
        shm.close()
    cap.release()
    cv2.destroyAllWindows()
    rknn.release()

if __name__ == '__main__':
    main(None, Value('b', False))
