import cv2


def draw_contours(img, contours):
    output = img.copy()
    cv2.drawContours(output, contours, -1, (0, 255, 0), 2)
    return output
