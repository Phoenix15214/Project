import os
os.environ.setdefault('QT_QPA_FONTDIR', '/usr/share/fonts/truetype/dejavu')

import cv2
import numpy as np
from rknnlite.api import RKNNLite
import process_lib.control_lib as ctrl
from multiprocessing import Process, Pipe, shared_memory, Value
from threading import Thread
import time
from detect import main as detect_main
from track import main as track_main
from transmit import Send_Process

shm_name = 'shared_frame'
# Separate pipes for track and detect to avoid producer contention on a single Pipe
pipe_track_recv, pipe_track_send = Pipe()
pipe_detect_recv, pipe_detect_send = Pipe()

def main():
    exit_code = 0
    frame_ready = Value('b', False)
    shm = shared_memory.SharedMemory(name=shm_name, create=True, size=640*480*3)
    shm.close()
    p_track = Process(target=track_main, args=(shm_name, frame_ready, pipe_track_send))
    p_detect = Process(target=detect_main, args=(shm_name, frame_ready, pipe_detect_send))
    # transmit process receives the two recv-ends so it can consume both independently
    p_transmission = Process(target=Send_Process, args=(([pipe_track_recv, pipe_detect_recv]), "justfloat"))
    p_transmission.start()
    p_track.start()
    p_detect.start()

    try:
        processes = [
            ("track", p_track),
            ("detect", p_detect),
            ("transmit", p_transmission),
        ]
        while True:
            for name, p in processes:
                if not p.is_alive():
                    code = p.exitcode
                    if code == 0:
                        raise RuntimeError(f"{name} 进程已退出，触发联动关闭")
                    raise RuntimeError(f"{name} 进程异常退出，exitcode={code}")
            time.sleep(0.5)
    except KeyboardInterrupt:
        print("KeyboardInterrupt received, terminating child processes...")
        exit_code = 0
        for p in (p_track, p_detect, p_transmission):
            try:
                if p.is_alive():
                    p.terminate()
            except Exception:
                pass
        for p in (p_track, p_detect, p_transmission):
            try:
                p.join(timeout=1)
            except Exception:
                pass
    except Exception as e:
        print(f"发生异常 ({e})，正在终止子进程...")
        exit_code = 1
        for p in (p_track, p_detect, p_transmission):
            try:
                if p.is_alive():
                    p.terminate()
            except Exception:
                pass
        for p in (p_track, p_detect, p_transmission):
            try:
                p.join(timeout=1)
            except Exception:
                pass

    try:
        shm = shared_memory.SharedMemory(name=shm_name)
        shm.unlink()  # 删除共享内存段
    except FileNotFoundError as e:
        print(f"共享内存段已被删除: {e}")
    except Exception as e:
        print(f"删除共享内存段时发生错误: {e}")

    if exit_code != 0:
        raise SystemExit(exit_code)

if __name__ == '__main__':
    main()