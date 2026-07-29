import os
import cv2
import numpy as np
import process_lib.control_lib as ctrl
from multiprocessing import Process, Pipe, shared_memory, Value, Event
from threading import Thread
import time
import traceback
# from detect import main as detect_main
import async_detect as detect_module
import transmit_experiment as transmit_experiment_module
from async_detect import main as detect_main
from transmit_asyncio import main as transmit_main
from transmit_experiment import main as transmit_experiment_main

pipe1, pipe2 = Pipe()


def _worker_entry(worker_name, target, *args, stop_event=None, error_event=None, **kwargs):
    try:
        target(*args, **kwargs)
    except KeyboardInterrupt:
        if stop_event is not None:
            stop_event.set()
        raise
    except Exception:
        print(f"[{worker_name}] crashed:")
        traceback.print_exc()
        if error_event is not None:
            error_event.set()
        if stop_event is not None:
            stop_event.set()
        raise


def _cleanup_shared_frames():
    for module in (detect_module, transmit_experiment_module):
        frame_share = getattr(module, "frame_share", None)
        if frame_share is None:
            continue
        try:
            frame_share.close()
        except Exception:
            pass

def main():
    p1 = None
    p2 = None
    p3 = None
    stop_event = Event()
    error_event = Event()
    frame_ready = Value('b', False)
    had_error = False

    try:
        p1 = Process(
            target=_worker_entry,
            args=("detect", detect_main, frame_ready),
            kwargs={"conn": pipe1, "stop_event": stop_event, "error_event": error_event},
        )
        p2 = Process(
            target=_worker_entry,
            args=("transmit", transmit_main),
            kwargs={"conn": pipe2, "stop_event": stop_event, "error_event": error_event},
        )
        p3 = Process(
            target=_worker_entry,
            args=("transmit_experiment", transmit_experiment_main, frame_ready),
            kwargs={"error_event": error_event},
        )

        p1.start()
        p2.start()
        p3.start()

        processes = [("detect", p1), ("transmit", p2), ("transmit_experiment", p3)]

        while True:
            if error_event.is_set():
                had_error = True
                stop_event.set()
                break

            if stop_event.is_set():
                break

            for name, proc in processes:
                if proc.is_alive():
                    continue

                if proc.exitcode is None:
                    continue

                print(f"[{name}] exited with code {proc.exitcode}")
                had_error = True
                stop_event.set()
                break

            if had_error:
                break

            if not any(proc.is_alive() for _, proc in processes):
                break

            time.sleep(0.2)
    except KeyboardInterrupt:
        print("KeyboardInterrupt received. Terminating processes...")
        stop_event.set()
    finally:
        for proc in (p1, p2, p3):
            if proc is not None and proc.is_alive():
                proc.terminate()

        for proc in (p1, p2, p3):
            if proc is not None:
                proc.join(timeout=2)

        _cleanup_shared_frames()

    if had_error:
        raise RuntimeError("At least one worker failed")

if __name__ == "__main__":
    main()
