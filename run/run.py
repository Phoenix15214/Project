import os
import cv2
import numpy as np
import process_lib.control_lib as ctrl
from multiprocessing import Process, Pipe, shared_memory, Value, Event
from threading import Thread
import time
import traceback
from detect import main as detect_main
from transmit_asyncio import main as transmit_main

pipe1, pipe2 = Pipe()


def _worker_entry(worker_name, target, conn, stop_event, error_event):
    try:
        target(conn=conn, stop_event=stop_event)
    except KeyboardInterrupt:
        stop_event.set()
        raise
    except Exception:
        print(f"[{worker_name}] crashed:")
        traceback.print_exc()
        error_event.set()
        stop_event.set()
        raise

def main():
    p1 = None
    p2 = None
    stop_event = Event()
    error_event = Event()

    try:
        p1 = Process(target=_worker_entry, args=("detect", detect_main, pipe1, stop_event, error_event))
        p2 = Process(target=_worker_entry, args=("transmit", transmit_main, pipe2, stop_event, error_event))

        p1.start()
        p2.start()

        processes = [("detect", p1), ("transmit", p2)]
        had_error = False

        while True:
            if error_event.is_set():
                had_error = True
                stop_event.set()
                break

            all_exited = True
            for name, proc in processes:
                if proc.is_alive():
                    all_exited = False
                    continue

                if proc.exitcode is None:
                    all_exited = False
                    continue

                if proc.exitcode != 0:
                    print(f"[{name}] exited with code {proc.exitcode}")
                    had_error = True
                    stop_event.set()
                    break

            if had_error:
                break

            if all_exited:
                break

            if (not p1.is_alive() and p1.exitcode == 0) or (not p2.is_alive() and p2.exitcode == 0):
                stop_event.set()

            time.sleep(0.2)

        if had_error:
            raise RuntimeError("At least one worker failed")
    except KeyboardInterrupt:
        print("KeyboardInterrupt received. Terminating processes...")
        stop_event.set()
    except Exception as e:
        print(f"An error occurred: {e}")
        raise
    finally:
        for proc in (p1, p2):
            if proc is not None and proc.is_alive():
                proc.terminate()

        for proc in (p1, p2):
            if proc is not None:
                proc.join(timeout=2)

if __name__ == "__main__":
    main()
