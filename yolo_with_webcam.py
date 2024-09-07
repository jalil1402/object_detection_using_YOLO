import cv2 as cv
from ultralytics import YOLO
import cvzone


cap=cv.VideoCapture(0)
cap.set(3,800)
cap.set(4,800)

yolo=YOLO("/home/jalil/learning_cv/weights/yolov8l.pt")

while True:
    success,img=cap.read()
    persons=yolo(img,stream=True)
    for p in persons:
        boxes=p.boxes
        for b in boxes:
            # bounding box
            x1,y1,x2,y2=b.xyxy[0]
            x1,y1,x2,y2=int(x1),int(y1),int(x2),int(y2)
            bbox=int(x1),int(y1),int(x2-x1),int(y2-y1)
            cvzone.cornerRect(img,bbox)
            # confidance
            conf=b.conf[0]
            cvzone.putTextRect(img,f'{str(int(conf*100))}% person',(max(0,int(x1)),max(35,int(y1))))
    cv.imshow("image", img)
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv.destroyAllWindows()


