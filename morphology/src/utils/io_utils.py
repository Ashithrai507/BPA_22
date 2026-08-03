import cv2


def load_image(path):
    return cv2.imread(path)


def save_image(path, img):
    cv2.imwrite(path, img)
