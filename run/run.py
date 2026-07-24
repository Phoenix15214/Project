import os
import cv2
import numpy as np
import process_lib.control_lib as ctrl
from multiprocessing import Process, Pipe, shared_memory, Value, Event
from threading import Thread
import time
from detect import main as detect_main
from transmit_asyncio import main as transmit_main
from track import main as track_main

pipe1, pipe2 = Pipe()

def main():
    frame_ready1 = Value('b', False)
    frame_ready2 = Value('b', False)
    try:
        p1 = Process(target=detect_main, args=(pipe1, frame_ready1, frame_ready2))
        p2 = Process(target=transmit_main, args=(pipe2,))
        p3 = Process(target=track_main, args=(frame_ready1, frame_ready2))

        p3.start()
        p2.start()
        p1.start()

        p3.join()
        p2.join()
        p1.join()
    except KeyboardInterrupt:
        print("KeyboardInterrupt received. Terminating processes...")
        p1.terminate()
        p2.terminate()
        p3.terminate()
    except Exception as e:
        print(f"An error occurred: {e}")
        p1.terminate()
        p2.terminate()
        p3.terminate()

if __name__ == "__main__":
    main()
