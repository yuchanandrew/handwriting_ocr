import easyocr
import cv2

# Initialize OCR reader
reader = easyocr.Reader(['en'])

# Grayscale conversion:
def grayscale(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Denoising:
def blurred(gray):
    return cv2.GaussianBlur(gray, (5, 5), 0)

# Binarization:
def binary(blurred):
    return cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)

# Resizing:
def resize(binary):
    return cv2.resize(binary, (800, 1000))

def main():
    # Load image
    img = cv2.imread("ocr_dir\images\processed_handwriting.jpg")

    # Preprocess image
    gray = grayscale(img)
    blurred_img = blurred(gray)
    binary_img = binary(blurred_img)
    resized_img = resize(binary_img)

    # Save processed
    process_img = cv2.imwrite("ocr_dir\images\processed_handwriting.jpg", resized_img)

    # Create results array
    results = reader.readtext(process_img)

    for result in results:
        bbox = result[0] # Bounding box

        # Bounding box coordinates displayed:
        # bbox = [
        #      [x1, y1], # Top-left corner
        #      [x2, y2], # Top-right corner
        #      [x3, y3], # Bottom-right corner
        #      [x4, y4] # Bottom-left corner
        #]

        text = result[1]

        pt1 = (int(bbox[0][0]), int(bbox[0][1])) # Referencing top-left corner
        pt2 = (int(bbox[2][0]), int(bbox[2][1])) # Referencing bottom-right corner

        print(f"Detected text: {result[1]} at {result[0]}")