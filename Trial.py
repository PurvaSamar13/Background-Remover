import cv2
import cvzone
from cvzone import stackImages
from cvzone.FPS import FPS
from cvzone.SelfiSegmentationModule import SelfiSegmentation
import os

cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)
segmentor = SelfiSegmentation()

fpsReader = FPS()
fpsReaderStart = False  # Variable to track whether fpsReader has started

while True:
    success, img = cap.read()
    imgOut = segmentor.removeBG(img, (255, 0, 0))

    imgStacked = stackImages([img, imgOut], 2, 1)

    if not fpsReaderStart:
        fpsReaderStart = True
        fpsReader.startCounter()  # Start the fpsReader when the first frame is processed

    fps, imgStacked = fpsReader.update(imgStacked)

    cv2.imshow("Image", imgStacked)
    cv2.waitKey(1)