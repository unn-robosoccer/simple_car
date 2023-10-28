import cv2
import utils
from cv2 import Canny
import numpy as np


if __name__ == '__main__':
    cap = cv2.VideoCapture('../videos/line_on_floor.mp4')
    lower_red = np.array([0, 100, 100])
    upper_red = np.array([10, 255, 255])


    while(True):
        ret, frame = cap.read()

        cv2.imshow('original', cv2.resize(frame, (0, 0), fx=0.5, fy=0.5))

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, lower_red, upper_red)
        frame = cv2.bitwise_and(frame, frame, mask=mask)

        frame[:200, :] = 0
        frame[-100:, :] = 0

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        edges = Canny(gray, 100, 170)
        #cv2.imshow('Video2', edges)

        res = utils.linedetect(edges)
        res = cv2.addWeighted(frame, 1, res, 0.8, 0)
        res = cv2.resize(res, (0, 0), fx=0.5, fy=0.5)

        cv2.imshow('detected road', res)  # cv2.imshow('frame',gray)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
