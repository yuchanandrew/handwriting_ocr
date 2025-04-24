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

# Find Contours:
def char_seg(binary):
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    character_images = []

    for contour in contours:
        if cv2.contourArea(contour) < 200:
            continue

        # Get bounding box coords
        x, y, w, h = cv2.boundingRect(contour)

        # Extract character image
        char_img = binary[y:y + h, x: x + w]
        character_images.append(char_img)

        cv2.rectangle(binary, (x, y), (x + w, y + h), (255, 255, 0), 2) # Draw rectangle around character
        cv2.putText(binary, "Char", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2) # Label character

    return character_images, binary

def main():
    # Load image
    img = cv2.imread(r"ocr_dir\images\test.jpg")

    if img is None: # Error handling for if image is not found
        print("Error: Image not found or cannot be opened.")
        return

    print("Image loaded successfully. Shape:", img.shape)

    # Preprocess image
    gray = grayscale(img)
    blurred_img = blurred(gray)
    binary_img = binary(blurred_img)
    resized_img = resize(binary_img)

    segmented_chars, processed_img = char_seg(resized_img)

    # Save processed
    cv2.imshow("Processed image: ", processed_img)
    cv2.waitKey(0)

    cv2.imwrite(r"ocr_dir\output\processed_image.jpg", processed_img)

    for i, char_img in enumerate(segmented_chars):
        # Save each segmented character image
        text = reader.readtext(char_img, detail=0, paragraph=False)
        print(f"Character {i + 1}: {text}")

        cv2.imwrite(r"ocr_dir\output\segments\char_{}.jpg".format(i + 1), char_img)

    # Create results array
    # results = reader.readtext(segmented_chars, detail=1, paragraph=False)

    # for result in results:
    #     bbox = result[0] # Bounding box

    #     # Bounding box coordinates displayed:
    #     # bbox = [
    #     #      [x1, y1], # Top-left corner
    #     #      [x2, y2], # Top-right corner
    #     #      [x3, y3], # Bottom-right corner
    #     #      [x4, y4] # Bottom-left corner
    #     #]

    #     text = result[1]

    #     pt1 = (int(bbox[0][0]), int(bbox[0][1])) # Referencing top-left corner
    #     pt2 = (int(bbox[2][0]), int(bbox[2][1])) # Referencing bottom-right corner

    #     print("Bounding Box:", pt1, pt2)
    #     print("Detected Text:", text)

    #     print(f"Drawing rectangle from {pt1} to {pt2}")

    #     cv2.rectangle(resized_img, pt1, pt2, (255, 255, 0), 2) # Draw rectangle around text
    #     cv2.putText(resized_img, text, (pt1[0], pt1[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)

    # # Save image with bounding boxes and text
    # cv2.imwrite(r"ocr_dir\output\output.jpg", resized_img)
    # print("Processed image saved as 'output.jpg'")

if __name__ == "__main__":
    main()