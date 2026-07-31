#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
香橙派发送端：共享内存/摄像头 → JPEG → UDP 分片发送

修复：
1. 真正限制发送帧率，不再被 120fps 源冲垮
2. 捕获线程按目标帧率丢帧，减少无效读取/复制
3. 自适应码率更合理，避免一开始就糊
"""

import cv2
import time
import socket
import struct
import threading
import random
import select
import json

from multiprocessing import Value

try:
    import process_lib.control_lib as ctrl
except Exception:
    ctrl = None


# ================= 配置 =================
TARGET = "tcp://BAMBOO.local:5555"   # 兼容旧配置，会自动解析成 UDP 目标
CAMERA = 0
WIDTH = 640
HEIGHT = 480

# 如果你想固定帧率，把 INIT_FPS 和 MAX_FPS 改成一样，例如都 30
INIT_QUALITY = 78
MIN_QUALITY = 75
MAX_QUALITY = 92

INIT_FPS = 30
MIN_FPS = 10
MAX_FPS = 30        # 热点图传建议 30；有线局域网可以 60

INIT_SCALE = 1.0
MIN_SCALE = 1.0

MTU = 1100
MAX_FRAME_BYTES = 1_200_000

MAGIC = 0x4F505631
HEADER = struct.Struct("!IIIHHd")
CTRL_MAGIC = b"OPVC"

NAME = socket.gethostname()
# ========================================


def parse_target(target: str):
    """
    兼容 tcp://BAMBOO.local:5555 这种旧配置。
    返回 (host, port)
    """
    t = target.strip()

    if "://" in t:
        t = t.split("://", 1)[1]

    t = t.split("/", 1)[0]

    if ":" in t:
        host, port = t.rsplit(":", 1)
        port = int(port)
    else:
        host = t
        port = 5555

    return host, port


def resolve_target(host: str, port: int):
    """
    解析 UDP 目标地址。
    优先 IPv4。
    """
    last_err = None

    for family in (socket.AF_INET, socket.AF_INET6):
        try:
            infos = socket.getaddrinfo(host, port, family, socket.SOCK_DGRAM)
            if infos:
                return infos[0][0], infos[0][4]
        except Exception as e:
            last_err = e

    raise RuntimeError(f"无法解析目标地址 {host}:{port}, {last_err}")


def main(frame_ready: Value):
    # ---------- 图像来源 ----------
    frame_share = None
    cam = None
    use_camera = False

    if ctrl is not None:
        try:
            frame_share = ctrl.MemoryShare(
                name="shared_frame",
                shape=(HEIGHT, WIDTH, 3),
                dtype="uint8"
            )
            print("[Pi] 使用共享内存 shared_frame")
        except Exception as e:
            print(f"[Pi] 共享内存初始化失败: {e}")
            frame_share = None

    if frame_share is None:
        use_camera = True
        cam = cv2.VideoCapture(CAMERA)
        cam.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
        cam.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)

        if not cam.isOpened():
            raise RuntimeError(f"无法打开摄像头 {CAMERA}")

        print(f"[Pi] 使用摄像头 {CAMERA}")

    stop_event = threading.Event()

    latest_lock = threading.Lock()
    latest = {
        "frame": None,
        "ts": 0.0,
        "new": False,
    }

    frame_event = threading.Event()

    state_lock = threading.Lock()
    state = {
        "quality": float(INIT_QUALITY),
        "fps": float(INIT_FPS),
        "scale": float(INIT_SCALE),
        "last_adj": 0.0,
        "start_t": time.time(),
    }

    # ---------- 捕获线程 ----------
    def capture_loop():
        """
        从共享内存或摄像头取帧。
        这里会按目标帧率主动丢帧，避免 120fps 全部进入编码/发送路径。
        """
        next_cap_t = 0.0

        while not stop_event.is_set():
            with state_lock:
                fps = max(5.0, state["fps"])

            interval = 1.0 / fps
            now = time.time()

            if not use_camera:
                # ---------- 共享内存来源 ----------
                if not frame_ready.value:
                    stop_event.wait(0.001)
                    continue

                # 如果还没到允许捕获的时间，就丢掉这一帧
                if now < next_cap_t:
                    frame_ready.value = False
                    stop_event.wait(min(0.002, max(0.0, next_cap_t - now)))
                    continue

                try:
                    frame = frame_share.read()
                except Exception:
                    stop_event.wait(0.05)
                    continue

                frame_ready.value = False

                if frame is None:
                    continue

                try:
                    frame = frame.copy()
                except Exception:
                    pass

                ts = time.time()
                next_cap_t = now + interval

            else:
                # ---------- 摄像头来源 ----------
                ok, frame = cam.read()

                if not ok or frame is None:
                    stop_event.wait(0.03)
                    continue

                # 摄像头需要持续 read，否则内部缓冲会旧帧堆积；
                # 但我们可以只把符合目标帧率的帧放进 latest。
                if now < next_cap_t:
                    continue

                ts = time.time()
                next_cap_t = now + interval

            with latest_lock:
                latest["frame"] = frame
                latest["ts"] = ts
                latest["new"] = True

            frame_event.set()

    # ---------- 网络目标 ----------
    host, port = parse_target(TARGET)

    family = socket.AF_INET
    target_addr = (host, port)

    while not stop_event.is_set():
        try:
            family, target_addr = resolve_target(host, port)
            break
        except Exception as e:
            print(f"[Pi] 解析 {host}:{port} 失败: {e}, 2s 后重试")
            time.sleep(2)

    if stop_event.is_set():
        return

    sock = socket.socket(family, socket.SOCK_DGRAM)

    try:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 2 * 1024 * 1024)
    except Exception:
        pass

    sock.setblocking(False)

    boot_id = random.randint(0, 0xFFFFFFFF)
    seq = 0

    print(f"[Pi] 目标: {host}:{port}")
    print(f"[Pi] boot_id: {boot_id}")

    # ---------- 控制反馈线程 ----------
    def control_loop():
        """
        接收电脑端反馈，根据 fps / loss / kbps 调整质量、帧率、分辨率。
        """
        while not stop_event.is_set():
            try:
                r, _, _ = select.select([sock], [], [], 0.25)
                if not r:
                    continue

                data, addr = sock.recvfrom(2048)
            except Exception:
                continue

            if not data.startswith(CTRL_MAGIC):
                continue

            try:
                msg = json.loads(data[len(CTRL_MAGIC):].decode("utf-8"))
            except Exception:
                continue

            with state_lock:
                now = time.time()

                # 避免调节过于频繁
                if now - state["last_adj"] < 1.0:
                    continue

                loss = float(msg.get("loss", 0.0))
                rfps = float(msg.get("fps", 0.0))
                kbps = float(msg.get("kbps", 0.0))

                q = state["quality"]
                fps = state["fps"]
                sc = state["scale"]

                startup = (now - state["start_t"]) < 5.0

                # 严重恶化：大幅降
                if loss > 15.0 or ((not startup) and rfps < max(1.0, fps * 0.55)):
                    q -= 10
                    fps = max(MIN_FPS, fps - 3)
                    sc = max(MIN_SCALE, sc - 0.06)

                # 中度恶化：小幅降
                elif loss > 7.0 or ((not startup) and rfps < max(1.0, fps * 0.82)):
                    q -= 4
                    fps = max(MIN_FPS, fps - 1)
                    sc = max(MIN_SCALE, sc - 0.02)

                # 网络良好：缓慢升
                # 只有接收帧率接近目标帧率时才升，避免异常高帧率误判
                elif loss < 2.0 and (0.88 * fps) <= rfps <= (1.25 * fps):
                    if kbps < 6000 and q < MAX_QUALITY:
                        q += 4

                    if kbps < 4500 and q > 70 and fps < MAX_FPS:
                        fps = min(MAX_FPS, fps + 1)

                    if kbps < 5000:
                        sc = min(1.0, sc + 0.02)

                # 码率过高保护
                if kbps > 8000:
                    q = max(MIN_QUALITY, q - 8)
                    fps = max(MIN_FPS, fps - 2)

                state["quality"] = min(MAX_QUALITY, max(MIN_QUALITY, q))
                state["fps"] = min(MAX_FPS, max(MIN_FPS, fps))
                state["scale"] = min(1.0, max(MIN_SCALE, sc))
                state["last_adj"] = now

    threading.Thread(target=capture_loop, daemon=True).start()
    threading.Thread(target=control_loop, daemon=True).start()

    # ---------- 主发送循环 ----------
    last_send = 0.0

    try:
        while not stop_event.is_set():
            with state_lock:
                fps = max(5.0, state["fps"])
                quality = int(state["quality"])
                scale = float(state["scale"])

            interval = 1.0 / fps
            now = time.time()

            # ========== 真正的帧率限制 ==========
            # 只有距离上次发送超过 interval，才允许进入发送流程。
            # 即使 frame_event 一直被 120fps 源 set，也不会提前发。
            if last_send > 0.0 and now < last_send + interval:
                stop_event.wait(min(0.005, last_send + interval - now))
                continue

            # 到发送节拍后再取最新帧
            with latest_lock:
                if not latest["new"]:
                    frame = None
                    ts = 0.0
                else:
                    frame = latest["frame"]
                    ts = latest["ts"]
                    latest["new"] = False

            if frame is None:
                # 没有新帧时等待一下，避免空转
                frame_event.clear()
                frame_event.wait(0.002)
                continue

            # ---------- 编码 ----------
            try:
                h, w = frame.shape[:2]

                if scale < 0.99:
                    nw = max(64, int(w * scale))
                    nh = max(48, int(h * scale))
                    frame = cv2.resize(
                        frame,
                        (nw, nh),
                        interpolation=cv2.INTER_AREA
                    )

                ok, jpg = cv2.imencode(
                    ".jpg",
                    frame,
                    [cv2.IMWRITE_JPEG_QUALITY, quality]
                )

                if not ok:
                    continue

                payload = jpg.tobytes()

            except Exception:
                continue

            # 如果单帧过大，先降质量，必要时跳过
            if len(payload) > MAX_FRAME_BYTES:
                with state_lock:
                    state["quality"] = max(MIN_QUALITY, state["quality"] - 6)

                if len(payload) > MAX_FRAME_BYTES * 2:
                    continue

                try:
                    q2 = max(MIN_QUALITY, quality - 15)
                    ok2, jpg2 = cv2.imencode(
                        ".jpg",
                        frame,
                        [cv2.IMWRITE_JPEG_QUALITY, q2]
                    )
                    if ok2:
                        payload = jpg2.tobytes()
                except Exception:
                    pass

            # ---------- 分片发送 ----------
            seq = (seq + 1) & 0xFFFFFFFF
            total = (len(payload) + MTU - 1) // MTU

            if total <= 0 or total > 65535:
                continue

            sent_frags = 0

            for i in range(total):
                header = HEADER.pack(
                    MAGIC,
                    boot_id,
                    seq,
                    i,
                    total,
                    ts
                )

                frag = payload[i * MTU:(i + 1) * MTU]
                packet = header + frag

                try:
                    sock.sendto(packet, target_addr)
                    sent_frags += 1
                except BlockingIOError:
                    break
                except OSError:
                    time.sleep(0.01)
                    break

            if sent_frags < total:
                # 发送缓冲满或网络阻塞，轻微退让
                time.sleep(0.005)

            # 记录真实发送时间，作为下一帧限速基准
            last_send = time.time()

    except KeyboardInterrupt:
        print("\n[Pi] 手动退出")

    finally:
        stop_event.set()

        if cam is not None:
            try:
                cam.release()
            except Exception:
                pass

        if frame_share is not None:
            try:
                frame_share.close()
            except Exception:
                pass

        try:
            sock.close()
        except Exception:
            pass


if __name__ == "__main__":
    frame_ready = Value("b", False)
    main(frame_ready=frame_ready)