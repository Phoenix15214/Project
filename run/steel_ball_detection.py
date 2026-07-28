import cv2
import numpy as np
import process_lib.image_lib as lb
import process_lib.control_lib as ctrl
CAMERA_WIDTH = 1280 # 1080p 1920*1080
CAMERA_HEIGHT = 720 # 1080p 1920*1080

frame_share2 = ctrl.MemoryShare(name='shared_frame2', shape=(720,1280,3), dtype='uint8')
white = np.full((CAMERA_HEIGHT, CAMERA_WIDTH), 255, dtype=np.uint8)

def preprocess_frame(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 3)
    white_frame = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    contours, _ = cv2.findContours(white_frame, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    white_frame = cv2.medianBlur(white_frame, 5)
    min_area = 100
    mask = np.zeros_like(white_frame)
    for contour in contours:
        if cv2.contourArea(contour) > min_area:
            cv2.drawContours(mask, [contour], -1, 255, thickness=cv2.FILLED)

    return mask

def get_lost_object(last_objects, current_objects, white_frame, rio_size):
    valid_objects = []
    for obj in current_objects:
        # roi = white_frame[max(0, obj[1] - rio_size):min(white_frame.shape[0], obj[1] + rio_size),
        #                     max(0, obj[0] - rio_size):min(white_frame.shape[1], obj[0] + rio_size)]
        # if np.any(roi == 255):
        valid_objects.append(obj)

    for last_obj in last_objects:
        for current_obj in current_objects:
            distance = np.linalg.norm(np.array(last_obj) - np.array(current_obj))
            if distance < rio_size:
                break
        else:
            x, y = last_obj
            roi = white_frame[max(0, y - rio_size):min(white_frame.shape[0], y + rio_size),
                              max(0, x - rio_size):min(white_frame.shape[1], x + rio_size)]
            if np.any(roi == 255):
                M = cv2.moments(roi)
                if M["m00"] != 0:
                    cX = int(M["m10"] / M["m00"]) + max(0, x - rio_size)
                    cY = int(M["m01"] / M["m00"]) + max(0, y - rio_size)
                    valid_objects.append((cX, cY))

    return valid_objects

def main(conn=None):
    last_objects = []
    try:
        with lb.YOLODetector(method="rknn", model_path="/home/ubuntu/Project/Project/run/best.rknn", num_classes=1, conf_thresh=0.5, iou_thresh=0.5, imgsz=(640, 640), cores=2) as detector:
            while True:
                frame = frame_share2.read().copy()
                white_frame = preprocess_frame(frame)
                boxes, scores, class_ids = detector.detect(frame)
                # frame = detector.draw_boxes(frame, boxes, scores, class_ids)
                current_centers = [(int((x1+x2)/2), int((y1+y2)/2)) for (x1, y1, x2, y2) in boxes]
                valid_centers = get_lost_object(last_objects, current_centers, white_frame, rio_size=40)
                for centers in valid_centers:
                    cv2.circle(frame, centers, 5, (0, 255, 0), -1)
                current_centers = []
                # for box in boxes:
                #     x1, y1, x2, y2 = box
                #     center_x = (x1 + x2) // 2
                #     center_y = (y1 + y2) // 2
                #     centers.append((center_x, center_y))
                last_objects = valid_centers
                # if conn is not None:
                #     conn.send([2, valid_centers])  # 发送检测结果
                frame = cv2.resize(frame, (640, 480))
                white_frame = cv2.resize(white_frame, (640, 480))
                cv2.imshow("White Frame", white_frame)
                cv2.imshow("Steel Ball Detection", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                
    except KeyboardInterrupt:
        print("KeyboardInterrupt received. Exiting...")
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        cv2.destroyAllWindows()
        frame_share2.unlink()
