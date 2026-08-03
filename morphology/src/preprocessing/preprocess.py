import cv2


def preprocess_image(img):
    # Resize (optional but stabilizes results)
    img = cv2.resize(img, (512, 512))

    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # CLAHE (contrast enhancement)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)

    # Blur
    blur = cv2.GaussianBlur(enhanced, (5, 5), 0)

    return gray, blur
