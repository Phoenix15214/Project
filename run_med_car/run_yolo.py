import os
os.environ.setdefault('QT_QPA_FONTDIR', '/usr/share/fonts/truetype/dejavu')

import cv2
import numpy as np
import process_lib.control_lib as ctrl
from multiprocessing import Process, Pipe, shared_memory, Value, Event
from threading import Thread
import signal
import time
from detect import main as detect_main
from track import main as track_main
from transmit import Send_Process
from detect_yolo import main as detect_yolo_main

shm_name = 'shared_frame'
pipe1, pipe2 = Pipe()


def _cleanup_processes(processes):
    for process in processes:
        if process is None:
            continue
        try:
            if process.is_alive():
                process.terminate()
        except AssertionError:
            continue
    for process in processes:
        if process is None:
            continue
        try:
            process.join(timeout=1)
        except AssertionError:
            continue


def _cleanup_shared_memory():
    try:
        shm = shared_memory.SharedMemory(name=shm_name)
    except FileNotFoundError:
        return
    try:
        shm.unlink()
    except FileNotFoundError:
        pass
    finally:
        shm.close()

def main():
    stop_event = Event()
    exit_code = 0
    frame_ready = Value('b', False)
    yolo_start = Value('b', False)
    p_track = None
    p_detect = None
    p_transmission = None
    try:
        try:
            shm = shared_memory.SharedMemory(name=shm_name, create=True, size=640*480*3)
        except FileExistsError:
            _cleanup_shared_memory()
            shm = shared_memory.SharedMemory(name=shm_name, create=True, size=640*480*3)
        shm.close()
        p_track = Process(target=track_main, args=(shm_name, frame_ready, yolo_start, pipe2, stop_event, 0))
        p_detect = Process(target=detect_yolo_main, args=(shm_name, frame_ready, yolo_start, pipe2, stop_event, 1))
        p_transmission = Process(target=Send_Process, args=(pipe1, "justfloat", stop_event, 2))

        p_track.start()
        p_detect.start()
        p_transmission.start()

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
        stop_event.set()
    except Exception as e:
        print(f"发生异常({e}),正在中止子进程...")
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
    finally:
        stop_event.set()
        _cleanup_processes((p_track, p_detect, p_transmission))
        _cleanup_shared_memory()

    if exit_code != 0:
        raise SystemExit(exit_code)
if __name__ == '__main__':
    main()