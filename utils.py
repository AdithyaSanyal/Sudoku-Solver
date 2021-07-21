import numpy as np
import cv2
from tensorflow.keras.models import load_model


# read model weights
def initializePredictionModel():
    # myModel.h5 is the model we get after training on mnist dataset
    model = load_model('Resources/myModel.h5')
    return model

def preProcess(img):
    #Convert to grayscale
    imGray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    # adding gaussian blur
    imBlur=cv2.GaussianBlur(imGray,(5,5),1)
    # applying adaptive threshold
    imgThreshold=cv2.adaptiveThreshold(imBlur,255,1,1,11,2)
    return imgThreshold

# Finding biggest contour assuming it is sudoku puzzle
# input all contours
def biggestContour(contours):
    biggest = np.array([])
    max_area = 0
    # Iterate through all contours
    for i in contours:
        area = cv2.contourArea(i)
        # if area of contour less than 50 it is noise
        if area > 50:
            peri = cv2.arcLength(i, True)
            # To find number of corners
            # len==4 means only searching for rectangle or square
            approx = cv2.approxPolyDP(i, 0.02 * peri, True)
            if area > max_area and len(approx) == 4:
                biggest = approx
                max_area = area
    return biggest,max_area

# Reordering points
def reorder(myPoints):
    myPoints = myPoints.reshape((4, 2))
    myPointsNew = np.zeros((4, 1, 2), dtype=np.int32)
    # add all the points
    add = myPoints.sum(1)
    # lowest value of sum will be at index 0
    myPointsNew[0] = myPoints[np.argmin(add)]
    # highest value will be at index 3
    myPointsNew[3] =myPoints[np.argmax(add)]
    # difference taken
    diff = np.diff(myPoints, axis=1)
    # lower value given index 1
    myPointsNew[1] =myPoints[np.argmin(diff)]
    # higher value given index 2
    myPointsNew[2] = myPoints[np.argmax(diff)]
    return myPointsNew

# Splitting sudoku into smaller boxes
def splitBoxes(img):
    # for rows vertically split the image into 9 sections
    rows = np.vsplit(img,9)
    boxes=[]
    for r in rows:
        # for columns horizontally split image into 9 parts
        cols= np.hsplit(r,9)
        for box in cols:
            boxes.append(box)
    return boxes

def displayNumbers(img,numbers,color = (0,255,0)):
    secW = int(img.shape[1]/9)
    secH = int(img.shape[0]/9)
    for x in range (0,9):
        for y in range (0,9):
            if numbers[(y*9)+x] != 0 :
                 cv2.putText(img, str(numbers[(y*9)+x]),
                               (x*secW+int(secW/2)-10, int((y+0.8)*secH)), cv2.FONT_HERSHEY_COMPLEX_SMALL,
                            2, color, 2, cv2.LINE_AA)
    return img

def drawGrid(img):
    secW = int(img.shape[1]/9)
    secH = int(img.shape[0]/9)
    for i in range (0,9):
        pt1 = (0,secH*i)
        pt2 = (img.shape[1],secH*i)
        pt3 = (secW * i, 0)
        pt4 = (secW*i,img.shape[0])
        cv2.line(img, pt1, pt2, (255, 255, 0),2)
        cv2.line(img, pt3, pt4, (255, 255, 0),2)
    return img

def getPrediction(boxes,model):
    result = []
    for image in boxes:
        ## prepare image
        img = np.asarray(image)
        img = img[4:img.shape[0] - 4, 4:img.shape[1] -4]
        img = cv2.resize(img, (28, 28))
        img = img / 255
        img = img.reshape(1, 28, 28, 1)
        ## predict on the model
        predictions = model.predict(img)
        classIndex = model.predict_classes(img)
        probabilityValue = np.amax(predictions)
        ## SAVE TO RESULT
        if probabilityValue > 0.8:
            result.append(classIndex[0])
        else:
            result.append(0)
    return result

# Stack all images in one window
def stackImages(imgArray,scale):
    rows = len(imgArray)
    cols = len(imgArray[0])
    rowsAvailable = isinstance(imgArray[0], list)
    width = imgArray[0][0].shape[1]
    height = imgArray[0][0].shape[0]
    if rowsAvailable:
        for x in range ( 0, rows):
            for y in range(0, cols):
                imgArray[x][y] = cv2.resize(imgArray[x][y], (0, 0), None, scale, scale)
                if len(imgArray[x][y].shape) == 2: imgArray[x][y]= cv2.cvtColor( imgArray[x][y], cv2.COLOR_GRAY2BGR)
        imageBlank = np.zeros((height, width, 3), np.uint8)
        hor = [imageBlank]*rows
        hor_con = [imageBlank]*rows
        for x in range(0, rows):
            hor[x] = np.hstack(imgArray[x])
            hor_con[x] = np.concatenate(imgArray[x])
        ver = np.vstack(hor)
        ver_con = np.concatenate(hor)
    else:
        for x in range(0, rows):
            imgArray[x] = cv2.resize(imgArray[x], (0, 0), None, scale, scale)
            if len(imgArray[x].shape) == 2: imgArray[x] = cv2.cvtColor(imgArray[x], cv2.COLOR_GRAY2BGR)
        hor= np.hstack(imgArray)
        hor_con= np.concatenate(imgArray)
        ver = hor
    return ver
    