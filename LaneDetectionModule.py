import cv2
import numpy as np

from CV_Workspace.Project1 import utlis1

curveList = []
avgVal = 10


# First we will create a getLaneCurve function and then apply some thresholding to our image.
def getLaneCurve(img, display=2):  # 0 nothing , 1 results , 2 everything -- use 0 for runtime
    imgCopy = img.copy()
    imgResult = img.copy()
    imgLaneColor = np.zeros_like(img)
    # step 1 threshold
    imgThres = utlis1.thresholding(img)

    # step 2 warp
    hT, wT, c = img.shape
    # initialTrackbarVals = [102, 80, 20, 214]
    # utlis.initializeTrackbars(initialTrackbarVals)
    points = utlis1.valTrackbars()
    imgWarp = utlis1.warpImg(imgThres, points, wT, hT)
    imgWarpPoints = utlis1.drawPoints(imgCopy, points)

    # step 3 finding lane curve
    middlePoint = utlis1.getHistogram(imgWarp, minPer=0.5)
    curveAveragePoint, imgHist = utlis1.getHistogram(imgWarp, True, 0.9, 1)
    curveRaw = curveAveragePoint - middlePoint

    # step 4 averaging
    curveList.append(curveRaw)
    if len(curveList) > avgVal:
        curveList.pop(0)
    curve = int(sum(curveList) / len(curveList))

    # step 5 stop sign
    flag, imgSign = utlis1.stopSign(img)

    # normalization
    curve2 = curve / 100  # normalize the output [-1,1]
    if curve2 > 1: curve2 = 1
    if curve2 < -1: curve2 = -1

    # step 6 display
    if display != 0:
        imgInvWarp = utlis1.warpImg(imgWarp, points, wT, hT, inv=True)
        imgInvWarp = cv2.cvtColor(imgInvWarp, cv2.COLOR_GRAY2BGR)
        imgInvWarp[0:hT // 3, 0:wT] = 0, 0, 0
        imgLaneColor[:] = 0, 255, 0
        imgLaneColor = cv2.bitwise_and(imgInvWarp, imgLaneColor)

        imgResult = cv2.addWeighted(imgResult, 1, imgLaneColor, 1, 0)
        imgResult = cv2.bitwise_or(imgSign, imgResult)  # important for stop sign
        midY = 450
        cv2.putText(imgResult, str(curve2), (wT // 2 - 60, 85), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 255), 1)
        cv2.line(imgResult, (wT // 2, midY), (wT // 2 + (curve * 3), midY), (255, 0, 255), 5)
        cv2.line(imgResult, ((wT // 2 + (curve * 3)), midY - 25), (wT // 2 + (curve * 3), midY + 25), (0, 255, 0), 5)

        imgCenter = 240
        cv2.circle(imgResult, (imgCenter, hT - 10), 9, (0, 0, 255), cv2.FILLED)
        cv2.circle(imgResult, (curveAveragePoint, hT - 10), 8, (255, 200, 0), 3)
        cv2.line(imgResult, (imgCenter, hT - 10), (curveAveragePoint, hT - 10), (0, 0, 0), 4)

        for x in range(-30, 30):
            w = wT // 20
            cv2.line(imgResult, (w * x + int(curve // 50), midY - 10),
                     (w * x + int(curve // 50), midY + 10), (0, 0, 255), 2)
        # fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer)
        # cv2.putText(imgResult, 'FPS ' + str(int(fps)), (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (230, 50, 50), 1)
    if display == 2:
        # imgStacked = utlis.stackImages(0.5, ([img, imgWarpPoints, imgWarp],[imgHist, imgLaneColor, imgResult]))
        imgStacked = utlis1.stackImages(0.9, ([img, imgWarpPoints],  # display all windows in one window
                                              [imgWarp, imgResult]))
        cv2.imshow('ImageStack', imgStacked)
    elif display == 1:
        cv2.imshow('Result', imgResult)

    # normalization
    curve = curve / 100  # normalize the output [-1,1]
    if curve > 1: curve = 1
    if curve < -1: curve = -1

    # cv2.imshow('thresh', imgThres)
    # cv2.imshow('warp', imgWarp)
    # cv2.imshow('warp points', imgWarpPoints)
    # cv2.imshow('histogram', imgHist)
    return flag, curve


if __name__ == '__main__':
    cap = cv2.VideoCapture(0)
    initialTrackbarVals = [102, 80, 20, 214]
    utlis1.initializeTrackbars(initialTrackbarVals)
    while True:
        timer = cv2.getTickCount()
        ret, img = cap.read()
        img = cv2.resize(img, (480, 240))
        flag, curve = getLaneCurve(img, display=2)
        print(curve * 100)
        # cv2.imshow('img', img)
        if cv2.waitKey(1) & 0xff == ord('q'):
            break
