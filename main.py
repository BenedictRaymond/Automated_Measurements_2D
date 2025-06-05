from Calibration.remove_distort import remove_distortion
from utils import crop_image
import cv2

img_path = 'bj_img.jpg'

# Step 1: Undistort the image
img = remove_distortion(
    img_path,
    r"Calibration/calibration_data.npz"
)
# Crop the image to remove unnecessary borders
img = crop_image(55, 30, img)  # Adjust the crop size as needed

cv2.imwrite("cropped_image.jpg", img)  # Save the cropped image

output_img = img.copy()  # Copy for drawing

# Step 2: Load pixel-to-mm ratio
with open('pixel_mm_ratio.txt', 'r') as f:
    pixel_mm_ratio = float(f.read().strip())
    print(f"Pixel to mm ratio: {pixel_mm_ratio}")

# Step 3: Preprocessing
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (5, 5), 0)
_, thresh = cv2.threshold(
                blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
            )

# Step 4: Find contours
contours, _ = cv2.findContours(
    thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
)

# Step 5: Process and annotate each contour
for cnt in contours:
    area_pixels = cv2.contourArea(cnt)
    perimeter_pixels = cv2.arcLength(cnt, True)

    # Convert to mm² and mm
    area = area_pixels / (pixel_mm_ratio ** 2)
    perimeter = perimeter_pixels / pixel_mm_ratio

    # Draw the contour
    cv2.drawContours(output_img, [cnt], -1, (0, 255, 0), 2)

    # Get a point to put the text (approx. center)
    M = cv2.moments(cnt)
    if M["m00"] != 0:
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
    else:
        cx, cy = cnt[0][0]  # fallback to the first point

    # Text to display
    text = f"A:{area:.1f} mm², P:{perimeter:.1f} mm"

    # Annotate on image
    cv2.putText(output_img, text, (cx - 50, cy), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (255, 0, 0), 2)

# Show and save the output image
cv2.imshow("Contour Area and Perimeter", output_img)
cv2.imwrite("output_with_contours.jpg", output_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
