#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""香橙派发送端：摄像头 → JPEG → ZMQ。依赖：pip install imagezmq opencv-python"""
import cv2
import imagezmq
import time
import socket
import zmq
import process_lib.control_lib as ctrl
from multiprocessing import shared_memory, Value, Pipe

# ================= 配置 =================
TARGET  = "tcp://BAMBOO.local:5555"   # 电脑地址，连不上就换成 tcp://192.168.x.x:5555
CAMERA  = 0                            # 摄像头编号
WIDTH   = 640
HEIGHT  = 480
QUALITY = 70                           # JPEG 质量 1~100
SEND_TIMEOUT = 2000                    # 发送超时(ms)
SEND_TIMEOUT_MS = 2000
# ========================================

frame_ready = Value('b', False)

NAME = socket.gethostname()
frame_share = ctrl.MemoryShare(name='shared_frame', shape=(HEIGHT, WIDTH, 3), dtype='uint8')

def main(frame_ready: Value):
    try:
        while True:
            sender = None
            cam = None
            try:
                # cam = cv2.VideoCapture(CAMERA)
                # cam.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
                # cam.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)
                # if not cam.isOpened():
                #     raise RuntimeError(f"无法打开摄像头 {CAMERA}")

                sender = imagezmq.ImageSender(connect_to=TARGET)
                sender.zmq_socket.setsockopt(zmq.SNDTIMEO, SEND_TIMEOUT)
                sender.zmq_socket.setsockopt(zmq.RCVTIMEO, SEND_TIMEOUT_MS)
                sender.zmq_socket.setsockopt(zmq.LINGER, 0)
                print(f"[{NAME}] 已连接 {TARGET}，开始发送…")

                while True:
                    if not frame_ready.value:
                        time.sleep(0.005)
                        continue

                    frame = frame_share.read()
                    frame_ready.value = False
                    ok, jpg = cv2.imencode(
                        ".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, QUALITY])
                    if not ok:
                        raise RuntimeError("JPEG 编码失败")

                    try:
                        sender.send_jpg(NAME, jpg)      # 同步等回复，保证 REQ-REP
                    except (zmq.Again, zmq.ZMQError, BrokenPipeError, OSError):
                        break

            except KeyboardInterrupt:
                print("\n退出")
                raise
            except Exception:
                # 非致命问题：释放本轮资源后从头重试，等待重连。
                pass
            finally:
                if cam is not None:
                    cam.release()
                if sender is not None:
                    try:
                        sender.zmq_socket.close()
                    except Exception:
                        pass

            time.sleep(2)

    finally:
        try:
            frame_share.close()
        except Exception:
            pass


if __name__ == "__main__":
    frame_ready = Value('b', False)
    main(frame_ready=frame_ready)