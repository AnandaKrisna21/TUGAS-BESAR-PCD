#!/usr/bin/python3
import cv2
import numpy as np
import utlis

#############################
path = "1.jpg"
heightImg = 700
widthImg = 700
#############################


img = cv2.imread(path)

#Preprocessing
img = cv2.resize(img, (widthImg,heightImg))
imgContours = img.copy()
imgBiggestContours = img.copy()
imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
imgBlur = cv2.GaussianBlur(imgGray,(5,5),1)
imgCanny = cv2.Canny(imgBlur,10,50)

#Finding All Contours
contours, hierarchy = cv2.findContours(imgCanny,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
cv2.drawContours(imgContours,contours,-1,(0,255,0),10)

#Find Rectangles
rectCon = utlis.rectContour(contours)
biggestContours = utlis.getCornerPoints(rectCon[0])
gradePoints = utlis.getCornerPoints(rectCon[1])

if biggestContours.size != 0 and gradePoints.size != 0:
    cv2.drawContours(imgBiggestContours,biggestContours,-1,(0,255,0),20)
    cv2.drawContours(imgBiggestContours,gradePoints,-1,(255,0,0),20)


imgBlank = np.zeros_like(img)
imageArray = ([img, imgGray, imgBlur, imgCanny],
            [imgContours,imgBiggestContours,imgBlank,imgBlank])
imgStacked = utlis.stackImages(imageArray,0.5)

cv2.imshow("Stacked Images",imgStacked)
cv2.waitKey(0)