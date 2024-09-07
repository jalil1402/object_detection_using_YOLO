import cv2 as cv
from ultralytics import YOLO
import cvzone

# Define class names
class_names = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck',
    'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench',
    'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra',
    'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis',
    'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard',
    'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife',
    'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot',
    'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
    'dining table', ' toilet', 'TV', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase',
    'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]

cap = cv.VideoCapture("/home/jalil/learning_cv/videos/8279996-hd_1920_1080_24fps.mp4")
cap.set(4, 800)
cap.set(3, 800)
yolo = YOLO("/home/jalil/learning_cv/weights/yolov8l.pt")

while True:
    success, img = cap.read()
    if not success:
        break
    
    persons = yolo(img, stream=True)
    for p in persons:
        boxes = p.boxes
        for b in boxes:
            # Bounding box
            x1, y1, x2, y2 = b.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            bbox = int(x1), int(y1), int(x2-x1), int(y2-y1)
            cvzone.cornerRect(img, bbox)
            
            # Confidence and class ID
            conf = b.conf[0]
            cls = int(b.cls[0])
            
            # Get class name
            class_name = class_names[cls] if cls < len(class_names) else 'unknown'
            
            # Display the text
            cvzone.putTextRect(img, f'{class_name} {str(int(conf*100))}%', (max(0, int(x1)), max(35, int(y1))))
    
    cv.imshow("image", img)
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv.destroyAllWindows()
