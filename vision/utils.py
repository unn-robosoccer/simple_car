import cv2
import numpy as np
import matplotlib.pyplot as plt


par_slope = 0.3
cf_high_image = 0.5
rightSlope, leftSlope, rightIntercept, leftIntercept = [], [], [], []
prev_right_x2, prev_left_x2 = 287, 470
center_interest = 710


def calculate_distance_to_center(img, lines, center_point=None, thickness=5, draw=True):
    """
    This function finds the distance to the center of the road
    1 Filter lines:
        The line will be left if it is located to the left of the center point
        The line will be right if it is located to the right of the center point
    2 Draw lines if draw is True

    :param img: img to draw on
    :param lines: lines to filter
    :param center_point: the center point
    :param thickness: thickness of drawn lines
    :param draw: draw or not
    :return: float - distance to center
    """

    # Define colors for left and right lines
    right_color = [0, 255, 0]  # Green color for right lines
    left_color = [255, 0, 0]  # Blue color for left lines

    # If center_point is not provided, set it as the horizontal center of the image
    if center_point is None:
        center_point = int(img.shape[1] / 2)

    # Initialize lists to store slope and y-intercept for left and right lines
    left_slopes, left_intercepts, right_slopes, right_intercepts = [], [], [], []

    # Iterate through lines and filter them based on their position relative to the center point
    if lines is not None:
        for line in lines:
            for x1, y1, x2, y2 in line:
                slope = (y1 - y2) / (x1 - x2)
                # Filter out nearly horizontal lines
                if -0.1 < slope < 0.1:
                    continue
                # Categorize lines as left or right based on their x-coordinate
                if x1 < center_point:
                    yintercept = y2 - (slope * x2)
                    leftSlope.append(slope)
                    leftIntercept.append(yintercept)
                if x1 > center_point:
                    yintercept = y2 - (slope * x2)
                    rightSlope.append(slope)
                    rightIntercept.append(yintercept)

    # Calculate average slope and intercept for left and right lines
    leftavgSlope = np.mean(leftSlope[-6:])
    leftavgIntercept = np.mean(leftIntercept[-6:])
    rightavgSlope = np.mean(rightSlope[-6:])
    rightavgIntercept = np.mean(rightIntercept[-6:])

    try:
        # Calculate the x-coordinates for left and right lines based on the averages
        left_line_x1 = int((cf_high_image * img.shape[0] - leftavgIntercept) / leftavgSlope)
        left_line_x2 = int((img.shape[0] - leftavgIntercept) / leftavgSlope)
        right_line_x1 = int((cf_high_image * img.shape[0] - rightavgIntercept) / rightavgSlope)
        right_line_x2 = int((img.shape[0] - rightavgIntercept) / rightavgSlope)

        # Calculate the center of the road as the average of the x-coordinates of left and right lines
        center_of_road = (left_line_x2 + right_line_x2) // 2

        # Calculate the distance from the center of the road to the center point
        distance_to_center = center_of_road - center_point

        # If draw is True, visualize the lines and center point on the image
        if draw:
            # Create a polygon representing the road area between left and right lines
            pts = np.array([[left_line_x1, int(cf_high_image * img.shape[0])],
                            [left_line_x2, int(img.shape[0])],
                            [right_line_x2, int(img.shape[0])],
                            [right_line_x1, int(cf_high_image * img.shape[0])]], np.int32)
            pts = pts.reshape((-1, 1, 2))
            cv2.fillPoly(img, [pts], (0, 0, 255))  # Fill road area with red color

            # Draw left and right lines on the image
            cv2.line(img, (left_line_x1, int(cf_high_image * img.shape[0])), (left_line_x2, int(img.shape[0])),
                     left_color, thickness)
            cv2.line(img, (right_line_x1, int(cf_high_image * img.shape[0])), (right_line_x2, int(img.shape[0])),
                     right_color, thickness)

            # Draw the center point on the road and connect it to the center point of the image
            center_y = img.shape[0] - 30
            cv2.circle(img, (center_of_road, center_y), 10, (0, 200, 250), thickness)
            cv2.line(img, (center_point, int(img.shape[0] - 30)), (center_point, int(img.shape[0])), left_color,
                     thickness)
            cv2.line(img, (center_point, int(img.shape[0] - 30)), (center_of_road, center_y), (0, 0, 0),
                     thickness - thickness // 2)

            # Display the distance to the center on the image
            cv2.putText(img, 'distance to center: ' + str(distance_to_center), (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), )

        return distance_to_center  # Return the distance to the center

    except ValueError:
        pass  # Handle any value errors that might occur




def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
    """
    `img` should be the output of a Canny transform.
    """
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len,
                            maxLineGap=max_line_gap)
    line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)

    # show hough lines
    # image_with_lines = np.copy(img)
    # image_with_lines = cv2.cvtColor(image_with_lines, cv2.COLOR_GRAY2BGR)
    #
    # if lines is not None:
    #     for line in lines:
    #         x1, y1, x2, y2 = line[0]
    #         cv2.line(image_with_lines, (x1, y1), (x2, y2), (0, 0, 255), 5)
    # image_with_lines = cv2.resize(image_with_lines, (0, 0), fx=0.5, fy=0.5)
    #
    # cv2.imshow("detected lines", image_with_lines)

    distance_to_center = calculate_distance_to_center(line_img, lines, 710)

    return {'image': line_img,
            'distance_to_center': distance_to_center}


def linedetect(img):
    return hough_lines(img, 1, np.pi / 180, 70, 10, 5)


def region_of_interest(image):
    image[:550, :] = 0
    height = image.shape[0]
    triangle = np.array([[(300, height), (1400, height), (642, 100)]])
    mask = np.zeros_like(image)
    cv2.fillPoly(mask, triangle, [255, 255, 255])
    masked_image = cv2.bitwise_and(image, mask)
    return masked_image


def prepare_frame(frame, lower_red=np.array([0, 100, 100]), upper_red=np.array([10, 255, 255])):
    """
    :return: gray image to canny
    """
    frame = cv2.GaussianBlur(frame, (45, 45), cv2.BORDER_DEFAULT)

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower_red, upper_red)
    frame = cv2.bitwise_and(frame, frame, mask=mask)

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return frame


def get_edges(frame):
    gray = prepare_frame(frame)
    edges = cv2.Canny(gray, 50, 150)
    edges = region_of_interest(edges)
    return edges

