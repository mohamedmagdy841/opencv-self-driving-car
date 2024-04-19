import cv2
import numpy as np


def thresholding(img):
    img = cv2.GaussianBlur(img, (5, 5), cv2.BORDER_DEFAULT)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    cv2.namedWindow("Tracking HSV")
    l_h = cv2.getTrackbarPos("LH", "Tracking HSV")
    l_s = cv2.getTrackbarPos("LS", "Tracking HSV")
    l_v = cv2.getTrackbarPos("LV", "Tracking HSV")
    u_h = cv2.getTrackbarPos("UH", "Tracking HSV")
    u_s = cv2.getTrackbarPos("US", "Tracking HSV")
    u_v = cv2.getTrackbarPos("UV", "Tracking HSV")
    l_b = np.array([l_h, l_s, l_v])
    u_b = np.array([u_h, u_s, u_v])
    # lowerWhite = np.array([97, 0, 0])
    # upperWhite = np.array([179, 253, 255])
    maskedWhite = cv2.inRange(hsv, l_b, u_b)
    return maskedWhite


def warpImg(img, points, w, h, inv=False):
    pts1 = np.float32(points)
    pts2 = np.float32([[0, 0], [w, 0], [0, h], [w, h]])
    if inv:
        matrix = cv2.getPerspectiveTransform(pts2, pts1)
    else:
        matrix = cv2.getPerspectiveTransform(pts1, pts2)
    imgWarp = cv2.warpPerspective(img, matrix, (w, h))
    return imgWarp


def nothing(a):
    pass


def initializeTrackbars(intialTracbarVals, wT=480, hT=240):
    cv2.namedWindow("Trackbars")
    cv2.resizeWindow("Trackbars", 360, 240)
    cv2.createTrackbar("Width Top", "Trackbars", intialTracbarVals[0], wT // 2, nothing)
    cv2.createTrackbar("Height Top", "Trackbars", intialTracbarVals[1], hT, nothing)
    cv2.createTrackbar("Width Bottom", "Trackbars", intialTracbarVals[2], wT // 2, nothing)
    cv2.createTrackbar("Height Bottom", "Trackbars", intialTracbarVals[3], hT, nothing)
    cv2.namedWindow("Tracking HSV")
    cv2.createTrackbar("LH", "Tracking HSV", 0, 255, nothing)
    cv2.createTrackbar("LS", "Tracking HSV", 0, 255, nothing)
    cv2.createTrackbar("LV", "Tracking HSV", 0, 255, nothing)
    cv2.createTrackbar("UH", "Tracking HSV", 255, 255, nothing)
    cv2.createTrackbar("US", "Tracking HSV", 255, 255, nothing)
    cv2.createTrackbar("UV", "Tracking HSV", 255, 255, nothing)


def valTrackbars(wT=480, hT=240):
    '''
    cv2.namedWindow("Trackbars")
    cv2.resizeWindow("Trackbars", 360, 240)
    cv2.createTrackbar("Width Top", "Trackbars", 102,wT//2, nothing)
    cv2.createTrackbar("Height Top", "Trackbars", 80, hT, nothing)
    cv2.createTrackbar("Width Bottom", "Trackbars", 20,wT//2, nothing)
    cv2.createTrackbar("Height Bottom", "Trackbars", 214, hT, nothing)
    '''
    cv2.namedWindow("Trackbars")
    widthTop = cv2.getTrackbarPos("Width Top", "Trackbars")
    heightTop = cv2.getTrackbarPos("Height Top", "Trackbars")
    widthBottom = cv2.getTrackbarPos("Width Bottom", "Trackbars")
    heightBottom = cv2.getTrackbarPos("Height Bottom", "Trackbars")
    points = np.float32([(widthTop, heightTop), (wT - widthTop, heightTop),
                         (widthBottom, heightBottom), (wT - widthBottom, heightBottom)])
    return points


def drawPoints(img, points):
    for x in range(0, 4):
        cv2.circle(img, (int(points[x][0]), int(points[x][1])), 15, (0, 0, 255), cv2.FILLED)
    return img


def getHistogram(img, display=False, minPer=0.1, region=4):
    if region == 1:
        histValues = np.sum(img, axis=0)  ##histogram on the whole image
    else:
        histValues = np.sum(img[img.shape[0] // region:, :],
                            axis=0)  # histogram only on the bottom 1/4 part of the image
    # ignore noise values
    maxValue = np.max(histValues)
    minValue = minPer * maxValue

    indexArray = np.where(histValues >= minValue)  # ALL INDICES WITH MIN VALUE OR ABOVE
    basePoint = int(np.average(indexArray))  # AVERAGE ALL MAX INDICES VALUES

    if display:
        imgHist = np.zeros((img.shape[0], img.shape[1], 3), np.uint8)
        for x, intensity in enumerate(histValues):
            if intensity > minValue:
                color = (255, 0, 255)
            else:
                color = (0, 0, 255)
            cv2.line(imgHist, (x, img.shape[0]), (x, int(img.shape[0] - (intensity // 255 // region))), color, 1)
        cv2.circle(imgHist, (basePoint, img.shape[0]), 20, (0, 255, 255), cv2.FILLED)
        return basePoint, imgHist

    return basePoint


def stackImages(scale, imgArray):
    rows = len(imgArray)
    cols = len(imgArray[0])
    rowsAvailable = isinstance(imgArray[0], list)
    width = imgArray[0][0].shape[1]
    height = imgArray[0][0].shape[0]
    if rowsAvailable:
        for x in range(0, rows):
            for y in range(0, cols):
                if imgArray[x][y].shape[:2] == imgArray[0][0].shape[:2]:
                    imgArray[x][y] = cv2.resize(imgArray[x][y], (0, 0), None, scale, scale)
                else:
                    imgArray[x][y] = cv2.resize(imgArray[x][y], (imgArray[0][0].shape[1], imgArray[0][0].shape[0]),
                                                None, scale, scale)
                if len(imgArray[x][y].shape) == 2: imgArray[x][y] = cv2.cvtColor(imgArray[x][y], cv2.COLOR_GRAY2BGR)
        imageBlank = np.zeros((height, width, 3), np.uint8)
        hor = [imageBlank] * rows
        hor_con = [imageBlank] * rows
        for x in range(0, rows):
            hor[x] = np.hstack(imgArray[x])
        ver = np.vstack(hor)
    else:
        for x in range(0, rows):
            if imgArray[x].shape[:2] == imgArray[0].shape[:2]:
                imgArray[x] = cv2.resize(imgArray[x], (0, 0), None, scale, scale)
            else:
                imgArray[x] = cv2.resize(imgArray[x], (imgArray[0].shape[1], imgArray[0].shape[0]), None, scale, scale)
            if len(imgArray[x].shape) == 2: imgArray[x] = cv2.cvtColor(imgArray[x], cv2.COLOR_GRAY2BGR)
        hor = np.hstack(imgArray)
        ver = hor
    return ver


def stopSign(img):
    # Stop Sign Cascade Classifier xml
    stop_sign = cv2.CascadeClassifier('/home/arm/Desktop/project2/Project1/cascade_stop_sign.xml')
    flag = 0
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    stop_sign_scaled = stop_sign.detectMultiScale(gray, 1.3, 10)

    if len(stop_sign_scaled):
        flag = 1

    # Detect the stop sign, x,y = origin points, w = width, h = height
    for (x, y, w, h) in stop_sign_scaled:
        # Draw rectangle around the stop sign
        stop_sign_rectangle = cv2.rectangle(img, (x, y),
                                            (x + w, y + h),
                                            (0, 255, 0), 3)
        # Write "Stop sign" on the bottom of the rectangle
        stop_sign_text = cv2.putText(img=stop_sign_rectangle,
                                     text="Stop Sign",
                                     org=(x, y + h + 30),
                                     fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                                     fontScale=0.5, color=(0, 0, 255),
                                     thickness=2, lineType=cv2.LINE_4)
    # cv2.imshow("stop sign", img)
    return flag, img
