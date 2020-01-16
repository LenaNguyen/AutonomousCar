import cv2
import numpy as np
import matplotlib.pyplot as plt


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


# def display_lines(image, lines):


image = cv2.imread('Image/test_image.jpg')
lane_image = np.copy(image)
canny = apply_canny_method(lane_image)
cropped_image = apply_region_of_interest(canny)
lines = cv2.HoughLinesP(cropped_image, 2, np.pi/180, 100,
                        np.array([]), minLineLength=40, maxLineGap=5)


cv2.imshow("result", cropped_image)
cv2.waitKey(0)
