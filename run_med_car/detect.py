import os
os.environ.setdefault('QT_QPA_FONTDIR', '/usr/share/fonts/truetype/dejavu')

import cv2
import numpy as np
import process_lib.control_lib as ctrl
import process_lib.image_lib as lb
from multiprocessing import Process, Pipe, shared_memory, Value
from threading import Thread
import time

IMG_SIZE = 640
CONF_THRESH = 0.5
IOU_THRESH = 0.45
CAMERA_FPS = 30

img = []

for i in range(8):
    path = f"/home/pi/Project/run/Templates/{i+1}.jpg"
    img.append(cv2.imread(path))
    img[i] = cv2.cvtColor(img[i], cv2.COLOR_BGR2GRAY)
    img[i] = cv2.resize(img[i], (100, 100), interpolation=cv2.INTER_NEAREST)

MIN_AREA = 20000
warped = img[7]
white = np.ones((480, 640), dtype=np.uint8) * 255

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

def main(shm_name, frame_ready, conn=None, stop_event=None, core=None):
    if core is not None:
        os.sched_setaffinity(0, {core})
    # cv2.setNumThreads(2)
    pack = ctrl.SerialPacket(port="/dev/ttyUSB0", baudrate=38400, timeout=0.1)if conn is None else None
    stop_event = stop_event or type("StopEvent", (), {"is_set": staticmethod(lambda: False)})()
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
    frame_count = 0
    cap = None
    if frame_view is None:
        cap, _ = open_camera(fps=CAMERA_FPS)
    ret, frame = cap.read()if cap is not None else (True, frame_view.copy())
    if not ret:
        print('Failed to open camera.'); return

    cv2.namedWindow('Contours', cv2.WINDOW_NORMAL)
    cv2.createTrackbar('Threshold', 'Contours', 140, 180, lambda x: None)

    try:
        while not stop_event.is_set():
            global warped
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
            # frame = cv2.GaussianBlur(frame, (5, 5), 0)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            threshold = cv2.getTrackbarPos('Threshold', 'Contours')
            black_mask = cv2.inRange(frame, (0, 0, 0), (180, 150, threshold))
            black = cv2.bitwise_and(white, white, mask=black_mask)
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            black = cv2.dilate(black, kernel, iterations=3)
            # black = cv2.morphologyEx(black, cv2.MORPH_CLOSE, kernel, iterations=5)
            contours = cv2.findContours(black, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0]
            draw_img = frame.copy()
            valid_targets = [(9999, 100), (9999, 100), (9999, 100), (9999, 100)]
            valid_boxes_cls_id = []
            for cnt in contours:
                epsilon = 0.1 * cv2.arcLength(cnt, True)
                approx = cv2.approxPolyDP(cnt, epsilon, True)
                contour_area = cv2.contourArea(cnt)
                contour_num = len(approx)
                if contour_area >= MIN_AREA and contour_num == 4:
                    rotating_box = cv2.minAreaRect(approx)
                    box = cv2.boxPoints(rotating_box)
                    box = np.int32(box)
                    box = ctrl.Reorder_Vertex(box)
                    warped = lb.Perspective_Transform(gray, box)
                    h, w = warped.shape
                    warped = warped[20:h-20, 20:w-20]
                    warped = cv2.GaussianBlur(warped, (7, 7), 0)
                    warped = cv2.resize(warped, (100, 100), interpolation=cv2.INTER_NEAREST)
                    index, location, match_scores = lb.Template_Matching(warped, img, threshold=0.4, min_scale=0.85, num_scale=2)
                    if index and location and match_scores:
                        valid_index = max(range(len(match_scores)), key=lambda i: match_scores[i])  # 获取最高匹配度的索引
                        valid_boxes_cls_id.append((box, valid_index))
                        print(f"Index: {index[valid_index]}, Location: {location[valid_index]}")
                        # if index[valid_index] == 5:
                        #     print(f"Match score: {match_scores[valid_index]:.4f} for template {index[valid_index]} at location {location[valid_index]}")
                        # print(index, match_scores)
                    draw_img = cv2.drawContours(draw_img, [box], -1, (0, 255, 0), 2)
            index = 0
            for box, cls_id in valid_boxes_cls_id:
                x1, y1, x2, y2 = box[0][0], box[0][1], box[2][0], box[2][1]
                if index < 4:
                    valid_targets[index] = (x1, cls_id)
                    index += 1
            if pack is not None:
                pack.insert_byte(0x08)  # 包头
            msg = [1, 0, 0, 0, 0]
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
