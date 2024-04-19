import cv2

# Stop Sign Cascade Classifier xml
stop_sign = cv2.CascadeClassifier('cascade_stop_sign.xml')


def stopSign(img):
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
                                     fontScale=0.7, color=(0, 0, 255),
                                     thickness=2, lineType=cv2.LINE_4)
    cv2.imshow("stop sign", img)
    '''
    key = cv2.waitKey(30)
    if key == ord('q'):
        cap.release()
        cv2.destroyAllWindows()
        break
    '''
    return flag
