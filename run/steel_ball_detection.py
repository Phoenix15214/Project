import cv2
import numpy as np
import process_lib.image_lib as lb
import process_lib.control_lib as ctrl

frame_share2 = ctrl.MemoryShare(name='shared_frame2', shape=(720,1280,3), dtype='uint8')

def main(conn=None, frame_ready2=None):
    try:
        with lb.YOLODetector(method="rknn", model_path="/home/ubuntu/Project/Project/run/best.rknn", num_classes=1, conf_thresh=0.5, iou_thresh=0.5, imgsz=(640, 640), cores=2) as detector:
            while True:
                if frame_ready2 is not None and frame_ready2.value:
                    frame = frame_share2.read()
                    #这里不对frame_ready2.value进行重置，因为在detect.py中会在处理完后重置
                    boxes, scores, class_ids = detector.detect(frame)
                    frame = detector.draw_boxes(frame, boxes, scores, class_ids)
                    centers = []
                    for box in boxes:
                        x1, y1, x2, y2 = box
                        center_x = (x1 + x2) // 2
                        center_y = (y1 + y2) // 2
                        centers.append((center_x, center_y))
                    if conn is not None:
                        conn.send([2, centers])  # 发送检测结果
                    frame_ready2.value = False
    except KeyboardInterrupt:
        print("KeyboardInterrupt received. Exiting...")
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        cv2.destroyAllWindows()
        frame_share2.unlink()
