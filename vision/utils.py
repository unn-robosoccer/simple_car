import cv2
import numpy as np
import matplotlib.pyplot as plt


par_slope = 0.3

cf_high_image = 0.5

rightSlope, leftSlope, rightIntercept, leftIntercept = [], [], [], []
prev_right_x2, prev_left_x2 = 287, 470

def draw_lines(img, lines, thickness=5):
    global rightSlope, leftSlope, rightIntercept, leftIntercept, prev_right_x1, prev_right_x2, prev_left_x1, prev_left_x2
    rightColor = [0, 255, 0]
    leftColor = [255, 0, 0]



    # this is used to filter out the outlying lines that can affect the average
    # We then use the slope we determined to find the y-intercept of the filtered lines by solving for b in y=mx+b
    if lines is not None:
        for line in lines:
            for x1, y1, x2, y2 in line:
                slope = (y1 - y2) / (x1 - x2)
                if slope > 0.1:
                    if x1 < 706:
                        yintercept = y2 - (slope * x2)
                        leftSlope.append(slope)
                        leftIntercept.append(yintercept)
                    elif x1 > 100:
                        yintercept = y2 - (slope * x2)
                        rightSlope.append(slope)
                        rightIntercept.append(yintercept)
                elif slope < -0.1:
                    if x1 > 706:
                        yintercept = y2 - (slope * x2)
                        rightSlope.append(slope)
                        rightIntercept.append(yintercept)
                    elif x1 < 900:
                        yintercept = y2 - (slope * x2)
                        leftSlope.append(slope)
                        leftIntercept.append(yintercept)
    # We use slicing operators and np.mean() to find the averages of the 30 previous frames
    # This makes the lines more stable, and less likely to shift rapidly
    #print(len(leftSlope))
    leftavgSlope = np.mean(leftSlope[-6:])
    leftavgIntercept = np.mean(leftIntercept[-6:])
    rightavgSlope = np.mean(rightSlope[-6:])
    rightavgIntercept = np.mean(rightIntercept[-6:])

    # print(img.shape[1])
    # prevleftavgSlope = np.mean(leftSlope[-50:-10])
    # prevleftavgIntercept = np.mean(leftIntercept[-50:-10])
    # prevrightavgSlope = np.mean(rightSlope[-50:-10])
    # prevrightavgIntercept = np.mean(rightIntercept[-50:-10])

    # Here we plot the lines and the shape of the lane using the average slope and intercepts
    try:
        left_line_x1 = int((cf_high_image * img.shape[0] - leftavgIntercept) / leftavgSlope)
        left_line_x2 = int((img.shape[0] - leftavgIntercept) / leftavgSlope)
        right_line_x1 = int((cf_high_image * img.shape[0] - rightavgIntercept) / rightavgSlope)
        right_line_x2 = int((img.shape[0] - rightavgIntercept) / rightavgSlope)

        # prev_left_x2 = int((img.shape[0] - prevleftavgIntercept) / prevleftavgSlope)
        # prev_right_x2 = int((img.shape[0] - prevrightavgIntercept) / prevrightavgSlope)
        #
        # if prev_right_x2 - right_line_x2 > 50:
        #     # Текущая правая линия начинается слишком далеко от предыдущей, принудительно относим ее к левой
        #     left_line_x1, left_line_x2 = right_line_x1, right_line_x2
        #
        #     # Обновляем значения для следующего кадра
        #     #prev_left_x1, prev_left_x2 = left_line_x1, left_line_x2
        #
        # # Измените блок сравнения для левой линии на следующий
        # if left_line_x2 - prev_left_x2 > 50:
        #     # Текущая левая линия начинается слишком далеко от предыдущей, принудительно относим ее к правой
        #     right_line_x1, right_line_x2 = left_line_x1, left_line_x2
        #
        #     # Обновляем значения для следующего кадра
        #     #prev_right_x1, prev_right_x2 = right_line_x1, right_line_x2

        pts = np.array([[left_line_x1, int(cf_high_image * img.shape[0])], [left_line_x2, int(img.shape[0])],
                        [right_line_x2, int(img.shape[0])], [right_line_x1, int(cf_high_image * img.shape[0])]], np.int32)
        pts = pts.reshape((-1, 1, 2))
        cv2.fillPoly(img, [pts], (0, 0, 255))
        cv2.line(img, (left_line_x1, int(cf_high_image * img.shape[0])), (left_line_x2, int(img.shape[0])), leftColor, 10)
        cv2.line(img, (right_line_x1, int(cf_high_image * img.shape[0])), (right_line_x2, int(img.shape[0])), rightColor, 10)
    except ValueError:
        pass




def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
    """
    `img` should be the output of a Canny transform.
    """
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len,
                            maxLineGap=max_line_gap)
    line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)



    image_with_lines = np.copy(img)
    image_with_lines = cv2.cvtColor(image_with_lines, cv2.COLOR_GRAY2BGR)

    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(image_with_lines, (x1, y1), (x2, y2), (0, 0, 255), 5)


    image_with_lines = cv2.resize(image_with_lines, (0, 0), fx=0.5, fy=0.5)

    cv2.imshow("detected lines", image_with_lines)








    draw_lines(line_img, lines)

    return line_img




def linedetect(img):
    return hough_lines(img, 1, np.pi / 180, 100, 10, 5)


def display_images(images, cmap=None):
    plt.figure(figsize=(40,40))
    for i, image in enumerate(images):
        plt.subplot(3,2,i+1)
        plt.imshow(image, cmap)
        plt.autoscale(tight=True)
    plt.show()


def region_of_interest(image):
    height = image.shape[0]
    triangle = np.array([[(300, height), (1400, height), (642, 100)]])
    mask = np.zeros_like(image)
    cv2.fillPoly(mask, triangle, [255, 255, 255])
    masked_image = cv2.bitwise_and(image, mask)
    return masked_image




# hough_img = list(map(linedetect, canny_img))
# display_images(hough_img)
