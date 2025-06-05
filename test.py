import cv2
import numpy as np

# Load images
bg_img = cv2.imread('bg_img.jpg')
obj_img = cv2.imread('bj_img.jpg')

# Resize to match dimensions (if needed)
bg_img = cv2.resize(bg_img, (obj_img.shape[1], obj_img.shape[0]))

# Create background subtractor
fgbg = cv2.createBackgroundSubtractorMOG2(history=2, detectShadows=True)

# Apply background model to the background (training step)
fgbg.apply(bg_img)  # This initializes the background model

# Apply it to the image with object
mask = fgbg.apply(obj_img)

# Remove shadows: MOG2 uses gray (127) for shadows, white (255) for foreground
# Keep only real foreground (object)
foreground_mask = cv2.inRange(mask, 0, 5)

# Morphological operations to clean up
kernel = np.ones((5, 5), np.uint8)
foreground_mask = cv2.morphologyEx(foreground_mask, cv2.MORPH_OPEN, kernel)
foreground_mask = cv2.dilate(foreground_mask, kernel, iterations=1)

# Find contours
contours, _ = cv2.findContours(foreground_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Draw contours
result = obj_img.copy()
cv2.drawContours(result, contours, -1, (0, 255, 0), 2)

# Show and save result
cv2.imshow("Object Detection Without Shadow (MOG2)", result)
cv2.imwrite("detected_object_mog2.jpg", result)
cv2.waitKey(0)
cv2.destroyAllWindows()
