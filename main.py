from Calibration.remove_distort import remove_distortion
from utils import crop_image, find_contours
import numpy as np
import cv2

img_path = 'test_image_01.jpg'

# Step 1: Undistort the image
img = remove_distortion(
    img_path,
    r"Calibration/calibration_data.npz"
)
# Crop the image to remove unnecessary borders
img = crop_image(55, 30, img)  # Adjust the crop size as needed

with open("pixel_mm_ratio.txt", "r") as f:
    pixel_mm_ratio = float(f.read().strip())


output_img = find_contours(img, pixel_mm_ratio)

# Show and save the output image
cv2.imshow("Contour Area and Perimeter", output_img)
cv2.imwrite("output_with_contours.jpg", output_img)
cv2.waitKey(0)

hsv_output_img = cv2.cvtColor(output_img, cv2.COLOR_BGR2HSV)

lower_green = np.array([40, 100, 100])
upper_green = np.array([80, 255, 255])

# Create mask for green color
mask = cv2.inRange(hsv_output_img, lower_green, upper_green)

# Find contours in the mask
contours, _ = cv2.findContours(
                mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )

# If contours found, get the bounding box of the largest contour
if contours:
    c = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(c)

    # Crop the image using the bounding box
    img = img[y:y+h, x:x+w]
    cv2.imwrite("cropped_object.jpg", img)  # Save the cropped object image
    final_img = find_contours(img, pixel_mm_ratio, invert_colors=True)
    cv2.imshow("Contour Area and Perimeter", final_img)
    cv2.imwrite("output_with_contours.jpg", final_img)
    cv2.waitKey(0)

else:
    print("No object found.")
