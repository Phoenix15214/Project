import os
import cv2
import numpy as np
import process_lib.control_lib as ctrl
from multiprocessing import Process, Pipe, shared_memory, Value, Event
from threading import Thread
import time
from detect import main as detect_main
from transmit_asyncio import main as transmit_main

pipe1, pipe2 = Pipe()

def main():
    try:
        p1 = Process(target=detect_main, args=(pipe1,))
        p2 = Process(target=transmit_main, args=(pipe2,))

        p1.start()
        p2.start()

        p1.join()
        p2.join()
    except KeyboardInterrupt:
        print("KeyboardInterrupt received. Terminating processes...")
        p1.terminate()
        p2.terminate()
    except Exception as e:
        print(f"An error occurred: {e}")
        p1.terminate()
        p2.terminate()

if __name__ == "__main__":
    main()
