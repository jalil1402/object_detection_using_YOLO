import cv2 as cv
import cvzone 
from ultralytics import YOLO


image=cv.imread("/home/jalil/learning_cv/imag/family.jpeg")
yolo=YOLO("/home/jalil/learning_cv/weights/yolov8n.pt")

face=yolo("/home/jalil/learning_cv/imag/family.jpeg", show=True)
cv.waitKey(0)






































# DISPLAYIG AN IMAGE
# image=cv.imread("/home/jalil/learning_cv/imag/mawgli.jpg")

# if image is not None:
#     cv.imshow('mawgli',image)
#     cv.waitKey(0)




# cap=cv.VideoCapture(0)

# while True:
#     sucsess, img=cap.read()
#     cv.imshow("image",img)
#     cv.waitKey(0)
#     cv.destroyAllWindows()

