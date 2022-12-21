import cv2 as cv
import numpy as np
import utlis

path = "semoga.jpeg"
lebar = 700
tinggi = 700
questions = 10
choices =5
answers = [0,1,2,3,4,0,1,2,3,3]
maxGrade = 100
webcamFeed=True
count = 0

cap = cv.VideoCapture(1)
cap.set(10,150)


while True:
    if webcamFeed:
        success, img = cap.read()
    else:
        img = cv.imread(path)

    # PREPROCESSING
    img = cv.resize(img, (lebar,tinggi))
    imgGray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    imgBlur = cv.GaussianBlur(imgGray, (5,5),1)
    imgCanny = cv.Canny(imgBlur, 10,50)
    imgContours= img.copy()
    imgBiggestContour = img.copy()
    imgFinal = img.copy()

    try:
        # FIND ALL CONTOURS
        contours, hierarchy, = cv.findContours(imgCanny, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
        cv.drawContours(imgContours, contours, -1, (0, 255, 0), 10) # DRAW ALL DETECTED CONTOURS

        rectCon= utlis.rectContour(contours)
        biggerCon = utlis.getCornerPoints(rectCon[0])
        # print("recon", rectCon[0])
        gradeCon = utlis.getCornerPoints(rectCon[1])
        #print (biggerCon)x


        if biggerCon.size != 0 and gradeCon.size != 0:
            cv.drawContours(imgBiggestContour,biggerCon, -1, (0, 255, 0), 20)
            cv.drawContours(imgBiggestContour,gradeCon, -1, (255, 0, 0), 20)

            biggestContour = utlis.reorder(biggerCon)# REORDER FOR WARPING
            gradePoints = utlis.reorder(gradeCon) # REORDER FOR WARPING

            pt1 = np.float32(biggestContour) # PREPARE POINTS FOR WAR
            pt2 = np.float32([[0, 0], [lebar, 0], [0, tinggi], [lebar, tinggi]])# PREPARE POINTS FOR WARP
                
            matrix = cv.getPerspectiveTransform(pt1, pt2) # GET TRANSFORMATION MATRIX
            imgWarpColored = cv.warpPerspective(img, matrix, (lebar, tinggi)) # APPLY WARP PERSPECTIVE
        
            ptsG1 = np.float32(gradePoints)  # PREPARE POINTS FOR WARP
            # PREPARE POINTS FOR WARP
            ptsG2 = np.float32([[0, 0], [325, 0], [0, 150], [325, 150]])
            matrixG = cv.getPerspectiveTransform(ptsG1, ptsG2)# GET TRANSFORMATION MATRIX
            imgGradeDisplay = cv.warpPerspective(img, matrixG, (325, 150)) # APPLY WARP PERSPECTIVE

            #cv.imshow("Grade",imgGradeDisplay)

            # Apply threshild
            imgWarpGray = cv.cvtColor(imgWarpColored, cv.COLOR_BGR2GRAY) # CONVERT TO GRAYSCALE
            imgThresh = cv.threshold(imgWarpGray, 170, 255, cv.THRESH_BINARY_INV)[1] # APPLY THRESHOLD AND INVERSE

            boxes = utlis.splitBoxes(imgThresh) # GET INDIVIDUAL BOXES
            #cv.imshow("test", boxes[0])
            #pixel value
            myPixelsVal = np.zeros((questions, choices)) # TO STORE THE NON ZERO VALUES OF EACH BOX
            countC = 0
            countR = 0

            for image in boxes:
                totalPixels = cv.countNonZero(image)

                myPixelsVal[countR][countC] = totalPixels
                countC += 1
                if (countC == choices):
                    countR += 1
                    countC = 0
            #print(myPixelsVal)

            #cari index
            myIndex = []
            for x in range(0, questions):
                arr = myPixelsVal[x]
            #print('arr',arr)
                myIndexVal = np.where(arr == np.max(arr))
                #print(myIndexVal[0])
                myIndex.append(myIndexVal[0][0])
            #print(myIndex)

            #compare jawaban 
            grading = []
            for x in range(0, questions):
                if answers[x] == myIndex[x]:
                    grading.append(1)
                else:
                    grading.append(0)
                #print(grading)

            score = (sum(grading)/questions) * maxGrade
            print(score)

        imgResults = imgWarpColored.copy()
        imgResults = utlis.showAnswers(imgResults ,myIndex,grading,answers,questions,choices)

        # Mask and combine the answers over the original image
        imgRawDrawing = np.zeros_like(imgWarpColored)
        imgRawDrawing = utlis.showAnswers(imgRawDrawing, myIndex, grading, answers, questions, choices)# DRAW ON NEW IMAGE
        invmatrix = cv.getPerspectiveTransform(pt2, pt1)# INVERSE TRANSFORMATION MATRIX# INVERSE TRANSFORMATION MATRIX
        imgInvWarp = cv.warpPerspective(imgRawDrawing, invmatrix, (lebar, tinggi)) # INV IMAGE WARP

        # Mask and combine the grade over the original image
        imgRawGrade = np.zeros_like(imgGradeDisplay) # NEW BLANK IMAGE WITH GRADE AREA SIZE
        cv.putText(imgRawGrade, str(round(score, 1))+"%", (30, 100),cv.FONT_HERSHEY_SIMPLEX, 3, (250, 250, 250), 5) # ADD THE GRADE TO NEW IMAGE
        invMatrixG = cv.getPerspectiveTransform(ptsG2, ptsG1)  # INVERSE TRANSFORMATION MATRIX
        imgInvGradeDisplay = cv.warpPerspective(imgRawGrade, invMatrixG, (lebar, tinggi)) # INV IMAGE WARP

        # SHOW ANSWERS AND GRADE ON FINAL IMAGE
        imgFinal = cv.addWeighted(imgFinal, 1, imgInvWarp, 1, 0)
        imgFinal = cv.addWeighted(imgFinal, 1, imgInvGradeDisplay, -1, 0)


        imgBlank = np.zeros_like(img)  
        imageArray = ([img, imgGray, imgBlur, imgCanny],
                    [imgContours, imgBiggestContour, imgWarpColored, imgThresh],
                    [imgResults, imgRawDrawing, imgInvWarp, imgFinal]
                    )
        cv.imshow("Final Result", imgFinal)
        
    except:   
        imgBlank = np.zeros_like(img)  
        imageArray = ([img, imgGray, imgBlur, imgCanny],
                    [imgContours, imgBiggestContour, imgWarpColored, imgThresh],
                    [imgResults, imgRawDrawing, imgInvWarp, imgFinal]
                    )

        imgStacked = utlis.stackImages(imageArray, 0.3)
        
        cv.imshow("Original", imgStacked)
            
    # SAVE IMAGE WHEN 's' key is pressed
    if cv.waitKey(1) & 0xFF == ord('s'):
        cv.imwrite("Scanned/myImage"+ str(count) + ".jpg", imgFinal)
        cv.rectangle(imgStacked, ((int(imgStacked.shape[1] / 2) - 230), int(imgStacked.shape[0] / 2) + 50),
        (1100, 350), (0, 255, 0), cv.FILLED)
        cv.putText(imgStacked, "Scan Saved", (int(imgStacked.shape[1] / 2) - 200, int(imgStacked.shape[0] / 2)),
                    cv.FONT_HERSHEY_DUPLEX, 3, (0, 0, 255), 5, cv.LINE_AA)
        cv.imshow('Original', imgStacked)
        cv.waitKey(300)
        count += 1

    elif cv.waitKey(1) == ord('x'):
        cap.release()
        cv.destroyAllWindows()
