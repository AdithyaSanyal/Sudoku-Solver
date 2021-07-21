print('Setting UP')
import os
from re import sub
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import cv2
import numpy as np
from utils import *
from solver import *

pathImage='Resources/1.jpg'
heightImage=423
widthImage=423
# loading cnn model
model=initializePredictionModel()

img=cv2.imread(pathImage)
# Resize image to make it square
img=cv2.resize(img,(widthImage,heightImage))
# Create blank image for testing
imgBlank=np.zeros((heightImage,widthImage,3),np.uint8)
imgThreshold=preProcess(img)

# finding contours
# copy to display image
imgContours = img.copy()
imgBigContour = img.copy()
# find all contours
# External method is used that is why only outer portion contours are found
contours, hierarchy = cv2.findContours(imgThreshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
# DRAW ALL DETECTED CONTOURS
cv2.drawContours(imgContours, contours, -1, (0, 255, 0), 3)

biggest,maxArea=biggestContour(contours)
print(biggest)
if biggest.size != 0:
    # to rearrange the order in which the points come
    biggest = reorder(biggest)
    # Draw biggest contour
    cv2.drawContours(imgBigContour, biggest, -1, (0, 0, 255), 25)
    # Prepare points for warp
    pts1 = np.float32(biggest)
    pts2 = np.float32([[0, 0],[widthImage, 0], [0, heightImage],[widthImage, heightImage]])
    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    # get the sudoku portion from the image
    imgWarpColored = cv2.warpPerspective(img, matrix, (widthImage, heightImage))
    imgDetectedDigits = imgBlank.copy()
    imgWarpColored = cv2.cvtColor(imgWarpColored,cv2.COLOR_BGR2GRAY)
    
    # splitting image into digits
    # create blank image for display
    imgSolvedDigits = imgBlank.copy()
    boxes = splitBoxes(imgWarpColored)
    print(len(boxes))

    
    numbers = getPrediction(boxes, model)
    print(numbers)
    imgDetectedDigits = displayNumbers(imgDetectedDigits, numbers, color=(255, 0, 255))
    numbers = np.asarray(numbers)
    # for value greater than 0 it will put a 0 else it will put 1
    posArray = np.where(numbers > 0, 0, 1)
    print(posArray)

    board = np.array_split(numbers,9)
    print(board)
    try:
        solve(board)
    except:
        pass
    print(board)
    flatlist=[]
    for sublist in board:
        for item in sublist:
            flatlist.append(item)
    solvedNumbers=flatlist*posArray
    imgSolvedDigits=displayNumbers(imgSolvedDigits,solvedNumbers)

    # to overlay the solution
    pts2 = np.float32(biggest) # PREPARE POINTS FOR WARP
    pts1 =  np.float32([[0, 0],[widthImage, 0], [0, heightImage],[widthImage, heightImage]]) # PREPARE POINTS FOR WARP
    matrix = cv2.getPerspectiveTransform(pts1, pts2)  # GER
    imgInvWarpColored = img.copy()
    imgInvWarpColored = cv2.warpPerspective(imgSolvedDigits, matrix, (widthImage, heightImage))
    inv_perspective = cv2.addWeighted(imgInvWarpColored, 1, img, 0.5, 1)
    imgDetectedDigits = drawGrid(imgDetectedDigits)
    imgSolvedDigits = drawGrid(imgSolvedDigits)

    imageArray=([img,imgThreshold,imgBigContour,imgBlank],
            [imgDetectedDigits,imgSolvedDigits,inv_perspective,imgBlank])

    stacked=stackImages(imageArray,1)
    cv2.imshow('Stacked images',stacked)
else:
    print('No Solution to Sudoku')

cv2.waitKey(0)

