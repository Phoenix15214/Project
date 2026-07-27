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
from steel_ball_detection import main as steel_ball_detection_main

pipe1, pipe2 = Pipe()
frame_share = ctrl.MemoryShare(name='shared_frame', shape=(CAMERA_HEIGHT,CAMERA_WIDTH,3), dtype='uint8')
frame_share2 = ctrl.MemoryShare(name='shared_frame2', shape=(CAMERA_HEIGHT,CAMERA_WIDTH,3), dtype='uint8')

def main():
    frame_ready1 = Value('b', False)
    frame_ready2 = Value('b', False)
    try:
        p1 = Process(target=detect_main, args=(pipe1, frame_ready1, frame_ready2))
        p2 = Process(target=transmit_main, args=(pipe2,))
        p3 = Process(target=track_main, args=(frame_ready1, frame_ready2))
        p4 = Process(target=steel_ball_detection_main, args=(pipe2, frame_ready2))

        p3.start()
        p2.start()
        p1.start()
        p4.start()

        p3.join()
        p2.join()
        p1.join()
        p4.join()
    except KeyboardInterrupt:
        print("KeyboardInterrupt received. Terminating processes...")
        p1.terminate()
        p2.terminate()
        p3.terminate()
        p4.terminate()
    except Exception as e:
        print(f"An error occurred: {e}")
        p1.terminate()
        p2.terminate()
        p3.terminate()
        p4.terminate()
    finally:
        p1.terminate() if p1.is_alive() else None
        p2.terminate() if p2.is_alive() else None
        p3.terminate() if p3.is_alive() else None
        p4.terminate() if p4.is_alive() else None
        frame_share.close()
        frame_share2.close()

if __name__ == "__main__":
    main()
