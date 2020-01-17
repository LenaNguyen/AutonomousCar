import cv2
import numpy as np
import matplotlib.pyplot as plt


def make_coordinates(image, line_parameters):
    slope, intercept = line_parameters
    y1 = image.shape[0]  # Gets the height of the image
    y2 = int(y1*(3/5))  # The line is 3 fifths of the image high
    x1 = int((y1 - intercept) / slope)
    x2 = int((y2 - intercept) / slope)
    return np.array([x1, y1, x2, y2])


def apply_canny_method(image):
    # convert to greyscale
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    # Gaussian Blur
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # Canny Method
    return cv2.Canny(blur, 50, 150)


def apply_region_of_interest(image):
    # The number of rows is the height of the image
    height = image.shape[0]
    polygons = np.array([[(200, height), (1100, height), (550, 250)]])

    # create an array with the same dimensions as image with 0s
    mask = np.zeros_like(image)

    # place the region of interest (triangle) onto the mask. Fill with white.
    cv2.fillPoly(mask, polygons, 255)

    # make the area that is not of intreset black
    masked_image = cv2.bitwise_and(image, mask)
    return masked_image


def display_lines(image, lines):
    line_image = np.zeros_like(image)
    if lines is not None:
        for x1, y1, x2, y2 in lines:
            cv2.line(line_image, (x1, y1), (x2, y2), (255, 0, 0), 10)
        return line_image


def average_slope_intercept(image, lines):
    left_fit = []
    right_fit = []
    for line in lines:
        x1, y1, x2, y2 = line.reshape(4)

        # fits a polynomial of degree n (in this case 1 bc we want a line)
        # to the specified points
        parameters = np.polyfit((x1, x2), (y1, y2), 1)
        slope = parameters[0]
        intercept = parameters[1]

        if slope < 0:
            left_fit.append((slope, intercept))
        else:
            right_fit.append((slope, intercept))

    left_line = [0, 0, 0, 0]
    right_line = [0, 0, 0, 0]

    if len(left_fit) and len(right_fit):
        left_fit_average = np.average(left_fit, axis=0)
        left_line = make_coordinates(image, left_fit_average)

    if len(right_fit):
        right_fit_average = np.average(right_fit, axis=0)
        right_line = make_coordinates(image, right_fit_average)

    return [left_line, right_line]

# image = cv2.imread('Assets/test_image.jpg')
# lane_image = np.copy(image)
# canny_image = apply_canny_method(lane_image)
# cropped_image = apply_region_of_interest(canny_image)
# lines = cv2.HoughLinesP(cropped_image, 2, np.pi/180, 100,
#                         np.array([]), minLineLength=40, maxLineGap=5)
# averaged_lines = average_slope_intercept(lane_image, lines)

# detected_lines_mask_image = display_lines(image, averaged_lines)

# # Combine detected lines with the original image
# # 	- The detected_lines_mask_image is given a higher weighting (1) so that they are more visible
# detected_lines_image = cv2.addWeighted(
#     lane_image, 0.8, detected_lines_mask_image, 1, 1)
# average_slope_intercept(lane_image, lines)

# cv2.imshow("result", detected_lines_image)
# cv2.waitKey(0)


cap = cv2.VideoCapture("Assets/test_video.mp4")

while(cap.isOpened()):
    ret, frame = cap.read()
    if cv2.waitKey(1) == ord('q') or not ret:
        break

    canny_image = apply_canny_method(frame)
    cropped_image = apply_region_of_interest(canny_image)
    lines = cv2.HoughLinesP(cropped_image, 2, np.pi/180, 100,
                            np.array([]), minLineLength=40, maxLineGap=5)
    averaged_lines = average_slope_intercept(frame, lines)

    detected_lines_mask_image = display_lines(frame, averaged_lines)

    # Combine detected lines with the original image
    # 	- The detected_lines_mask_image is given a higher weighting (1) so that they are more visible
    detected_lines_image = cv2.addWeighted(
        frame, 0.8, detected_lines_mask_image, 1, 1)

    cv2.imshow("result", detected_lines_image)

cap.release()
cv2.destroyAllWindows()
