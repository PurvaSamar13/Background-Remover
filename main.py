import cv2
import cvzone
from cvzone.SelfiSegmentationModule import SelfiSegmentation
import os

cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)
segmentor = SelfiSegmentation()
#imgBg = cv2.imread("images/Image1.jpg")

ListImg = os.listdir("images")
print(ListImg)
imgList = []
for imgPath in ListImg:
    img = cv2.imread(f'images/{imgPath}')
    imgList.append(img)
print(len(imgList))
IndexImg = 0
while True:
    success, img = cap.read()
    imgOut = segmentor.removeBG(img, imgList[IndexImg])

    imgStacked = cvzone.stackImages([img, imgOut], 2, 1)
    print(IndexImg)

    cv2.imshow("Image", imgStacked)
    key = cv2.waitKey(1)
    if key == ord('a'):
        if IndexImg > 0:
            IndexImg -= 1
    elif key == ord('d'):
        if IndexImg<len(imgList)-1:
            IndexImg += 1
    elif key == ord('q'):
        break


