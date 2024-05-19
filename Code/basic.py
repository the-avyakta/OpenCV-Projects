import cv2
import numpy as np

# Read the image in grayscale
img = cv2.imread('src/test2.jpg', 0)
if img is None:
    print("Error: Could not read the image.")
else:
    w, h = 300, 300

    # Resize the image
    resize_img = cv2.resize(img, (w, h), interpolation=cv2.INTER_LINEAR)

    # Apply median blur
    img_blur = cv2.medianBlur(resize_img, ksize=3)

    # Apply Sobel edge detection
    sobelx = cv2.Sobel(src=img_blur, ddepth=cv2.CV_64F, dx=1, dy=0, ksize=5)
    sobely = cv2.Sobel(src=img_blur, ddepth=cv2.CV_64F, dx=0, dy=1, ksize=5)
    sobelxy = cv2.Sobel(src=img_blur, ddepth=cv2.CV_64F, dx=1, dy=1, ksize=5)

    sobelxy2 = cv2.Sobel(src=resize_img, ddepth=cv2.CV_64F, dx=1, dy=1, ksize=5)

    # Convert Sobel images to uint8 for display
    abs_sobelx = cv2.convertScaleAbs(sobelx)
    abs_sobely = cv2.convertScaleAbs(sobely)
    abs_sobelxy = cv2.convertScaleAbs(sobelxy)

    abs_sobelxy2 = cv2.convertScaleAbs(sobelxy2)

    # Concatenate images horizontally
    imgfull = np.concatenate((resize_img, img_blur, abs_sobelxy, abs_sobelxy2), axis=1)
    # Display the concatenated images
    cv2.imshow("all", imgfull)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
