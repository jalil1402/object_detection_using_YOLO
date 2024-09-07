import cv2 as cv
from ultralytics import YOLO
import cvzone
from sort import *
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
    'dining table', 'toilet', 'TV', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase',
    'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]

# Load the mask
mask = cv.imread("/home/jalil/learning_cv/images/mask.png")
# Initialize video capture
cap = cv.VideoCapture("/home/jalil/learning_cv/videos/aerial-view-of-asphalt-road-with-the-rural-area-on-both-sides-the-intercity-ro-SBV-347656697-preview.mp4")

# Set video frame width and height
cap.set(4, 800)
cap.set(3, 800)

# Load YOLO model
yolo = YOLO("/home/jalil/learning_cv/weights/yolov8l.pt")

car_counter = 0
detections=np.empty((0,5))
# tracking 
tracker=Sort(max_age=20,min_hits=3,iou_threshold=0.3)
while True:
    success,img= cap.read()
    
    if not success:
        break
    
    # Resize mask to match the size of the video frame
    # resized_mask = cv.resize(mask, (Imag.shape[1], Imag.shape[0]))
    
    # Apply bitwise_and operation
    # img = cv.bitwise_and(Imag, resized_mask)
    
    persons = yolo(img, stream=True)
    for p in persons:
        boxes = p.boxes
        for b in boxes:
            # Bounding box
            x1, y1, x2, y2 = b.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            bbox = int(x1), int(y1), int(x2-x1), int(y2-y1)
            
            # Confidence and class ID
            conf = b.conf[0]
            cls = int(b.cls[0])
            
            # Get class name
            class_name = class_names[cls] if cls < len(class_names) else 'unknown'
            
            if class_name == 'car':
                car_counter += 1
                cvzone.cornerRect(img, bbox, l=30, t=5, rt=1,
                              colorR=(255, 0, 255), colorC=(0, 255, 0))
                cvzone.putTextRect(img, f'{class_name} {car_counter} {str(int(conf*100))}%', (max(0, int(x1)), max(35, int(y1))))
                cvzone.putTextRect(img, f'{class_name} {str(int(conf*100))}%', (max(0, int(x1)), max(35, int(y1))))
                curant_aray=np.array((x1,y1,x2,y2,conf))
                detections=np.vstack((detections,curant_aray))
    

    results=tracker.update(detections)
    print("we reached here")
    for r in results:
     x1,y1,x2,y2,id=r
     print(x1,y1,x2,y2,id)
       
    cv.imshow("image", img)
    if cv.waitKey(0) & 0xFF == ord('q'):
        break
 

cap.release()
cv.destroyAllWindows()
