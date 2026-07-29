#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""香橙派发送端：摄像头 → JPEG → ZMQ。依赖：pip install imagezmq opencv-python"""
import cv2
import imagezmq
import time
import socket
import zmq

# ================= 配置 =================
TARGET  = "tcp://BAMBOO.local:5555"   # 电脑地址，连不上就换成 tcp://192.168.x.x:5555
CAMERA  = 0                            # 摄像头编号
WIDTH   = 640
HEIGHT  = 480
QUALITY = 70                           # JPEG 质量 1~100
SEND_TIMEOUT = 2000                    # 发送超时(ms)
SEND_TIMEOUT_MS = 2000
# ========================================

NAME = socket.gethostname()


def main():
    while True:
        sender = None
        cam = None
        try:
            cam = cv2.VideoCapture(CAMERA)
            cam.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
            cam.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)
            if not cam.isOpened():
                raise RuntimeError(f"无法打开摄像头 {CAMERA}")

            sender = imagezmq.ImageSender(connect_to=TARGET)
            sender.zmq_socket.setsockopt(zmq.SNDTIMEO, SEND_TIMEOUT)
            sender.zmq_socket.setsockopt(zmq.RCVTIMEO, SEND_TIMEOUT_MS)
            sender.zmq_socket.setsockopt(zmq.LINGER, 0)
            print(f"[{NAME}] 已连接 {TARGET}，开始发送…")

            while True:
                ret, frame = cam.read()
                if not ret:
                    raise RuntimeError("摄像头读帧失败")
                ok, jpg = cv2.imencode(
                    ".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, QUALITY])
                if not ok:
                    continue
                sender.send_jpg(NAME, jpg)      # 同步等回复，保证 REQ-REP

        except zmq.Again:
            print("发送超时（电脑端未响应），2 秒后重试…")
        except KeyboardInterrupt:
            print("\n退出")
            break
        except Exception as e:
            print(f"错误：{e}，2 秒后重试…")
        finally:
            if cam is not None:
                cam.release()
            if sender is not None:
                try:
                    sender.zmq_socket.close()
                except Exception:
                    pass
            time.sleep(2)


if __name__ == "__main__":
    main()