from Calibration.remove_distort import remove_distortion
from utils import crop_image, find_contours, contours_to_dxf
from skimage.metrics import structural_similarity
import numpy as np
import cv2

# Load images
bg_img = cv2.imread('background.jpg')
obj_img = cv2.imread('with_object.jpg')

if bg_img is None or obj_img is None:
    print("Error: Could not load one or both images. Check the file paths.")
    exit()

# Undistort the image
bg_img = remove_distortion(
            "background.jpg", "Calibration/calibration_data.npz"
        )
obj_img = remove_distortion(
            "with_object.jpg", "Calibration/calibration_data.npz"
        )

# Crop the images to remove unnecessary borders
bg_img = crop_image(55, 30, bg_img)
obj_img = crop_image(55, 30, obj_img)

# Resize background to match object image
bg_img = cv2.resize(bg_img, (obj_img.shape[1], obj_img.shape[0]))

# Convert to grayscale
bg_gray = cv2.cvtColor(bg_img, cv2.COLOR_BGR2GRAY)
obj_gray = cv2.cvtColor(obj_img, cv2.COLOR_BGR2GRAY)


(score, diff) = structural_similarity(bg_gray, obj_gray, full=True)

diff = (diff * 255).astype("uint8")

thresh = cv2.threshold(
            diff, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU
        )[1]
contours = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
obj_contours = contours[0] if len(contours) == 2 else contours[1]

mask = np.zeros(bg_img.shape, dtype='uint8')
filled_after = obj_img.copy()

areas = []
for c in obj_contours:
    area = cv2.contourArea(c)
    areas.append(area)
    max_area = max(areas)
    if area == max_area:
        x, y, w, h = cv2.boundingRect(c)
        img = obj_img[y:y+h, x:x+w]

with open("pixel_mm_ratio.txt", "r") as f:
    pixel_mm_ratio = float(f.read().strip())

output_img, obj_contours = find_contours(img, pixel_mm_ratio)
cv2.imwrite("output_image.jpg", output_img)  # Debugging output

hsv_output_img = cv2.cvtColor(output_img, cv2.COLOR_BGR2HSV)

lower_green = np.array([40, 100, 100])
upper_green = np.array([80, 255, 255])

# Create mask for green color
mask = cv2.inRange(hsv_output_img, lower_green, upper_green)
cv2.imwrite('Masked_Debug.jpg', mask)

# Find contours in the mask
contours, _ = cv2.findContours(
                mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )

# If contours found, get the minimal enclosing rectangle
if contours:
    # Get the minimal enclosing rectangle
    rect = cv2.minAreaRect(contours[0])
    box = cv2.boxPoints(rect)
    box = np.int32(box)  # Use int32 for OpenCV compatibility
    box_contour = box.reshape((-1, 1, 2))  # Shape (4,1,2) for OpenCV

    final_img, detected_contours = find_contours(
        img, pixel_mm_ratio, invert_colors=True
    )
    cv2.drawContours(final_img, detected_contours, -1, (255, 0, 0), 2)
    cv2.drawContours(final_img, obj_contours, -1, (0, 255, 0), 2)
    # Filter contours that lie inside the rectangle
    filtered_contours = []
    for contour in detected_contours:
        # Get the center point of the contour
        M = cv2.moments(contour)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            # Check if center point is inside the rectangle
            if cv2.pointPolygonTest(box_contour, (cx, cy), False) >= 0:
                filtered_contours.append(contour)

    cv2.imshow("Contour Area and Perimeter", final_img)
    cv2.imwrite("output_with_contours.jpg", final_img)

    # Generate DXF file from contours
    dxf_file = contours_to_dxf(
        list(detected_contours) + filtered_contours,
        pixel_mm_ratio,
        "output.dxf"
    )
    print(f"DXF file generated: {dxf_file}")

    cv2.waitKey(0)

else:
    print("No object found.")
