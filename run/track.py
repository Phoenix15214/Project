import cv2
import numpy as np
import process_lib.image_lib as lb
import process_lib.control_lib as ctrl
from multiprocessing import Process, Pipe, shared_memory, Value, Event
from threading import Thread

CAMERA_FPS = 30
CAMERA_WIDTH = 1280 # 1080p 1920*1080
CAMERA_HEIGHT = 720 # 1080p 1920*1080
FRAME_CENTER_X = CAMERA_WIDTH // 2
FRAME_CENTER_Y = CAMERA_HEIGHT // 2

frame_share = ctrl.MemoryShare(name='shared_frame', shape=(CAMERA_HEIGHT,CAMERA_WIDTH,3), dtype='uint8')
frame_share2 = ctrl.MemoryShare(name='shared_frame2', shape=(CAMERA_HEIGHT,CAMERA_WIDTH,3), dtype='uint8')

def open_camera(camera_index=0):
    try:
        cap = cv2.VideoCapture(camera_index)
        cap.set(cv2.CAP_PROP_FPS, CAMERA_FPS)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M','J','P','G'))
        cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)
        cap.set(cv2.CAP_PROP_EXPOSURE, 30)
        actual_auto_exp = cap.get(cv2.CAP_PROP_AUTO_EXPOSURE)
        actual_exp = cap.get(cv2.CAP_PROP_EXPOSURE)
        print(f"Camera settings: Auto Exposure={actual_auto_exp}, Exposure={actual_exp}")
        return cap
    except Exception as e:
        print(f"Error opening camera: {e}")
        raise RuntimeError("Failed to open camera.")
        return None

def video_capture_1(frame_ready: Value):
    cap = open_camera(0)
    if cap is None:
        print("Camera could not be opened. Exiting.")
        return
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to read frame from camera.")
                break
            frame_share.write(frame)
            frame_ready.value = True

    except KeyboardInterrupt:
        print("KeyboardInterrupt received. Exiting...")
    except Exception as e:
        print(f"An error occurred: {e}")

    finally:
        cap.release()
        frame_share.close()

def video_capture_2(frame_ready: Value):
    cap2 = open_camera(2)
    if cap2 is None:
        print("Camera 2 could not be opened. Exiting.")
        return
    try:
        while True:
            ret2, frame2 = cap2.read()
            if not ret2:
                print("Failed to read frame from camera 2.")
                return
            frame_share2.write(frame2)
            frame_ready.value = True

    except KeyboardInterrupt:
        print("KeyboardInterrupt received. Exiting...")
    except Exception as e:
        print(f"An error occurred: {e}")

    finally:
        cap2.release()
        frame_share2.close()

def main(frame_ready1: Value, frame_ready2: Value):
    try:
        thread1 = Thread(target=video_capture_1, args=(frame_ready1,))
        thread2 = Thread(target=video_capture_2, args=(frame_ready2,))
        thread1.start()
        thread2.start()
        thread1.join()
        thread2.join()
    except KeyboardInterrupt:
        print("KeyboardInterrupt received. Exiting...")
        thread1.terminate()
        thread2.terminate()
    except Exception as e:
        print(f"An error occurred in main: {e}")
        thread1.terminate()
        thread2.terminate()

if __name__ == "__main__":
    frame_ready1 = Value('b', False)
    frame_ready2 = Value('b', False)
    main(frame_ready1, frame_ready2)
