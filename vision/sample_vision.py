import cv2
import numpy as np
from vision import vision


if __name__ == '__main__':
    cap = cv2.VideoCapture('../videos/line_on_floor.mp4')

    while(True):
        ret, frame = cap.read()
        state = vision(frame, show=True, center_point=710)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
