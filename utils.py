import cv2


def crop_image(length, width, img):
    img = img[length:1080 - length, width:1920 - width]
    return img


def find_contours(img, pixel_mm_ratio, invert_colors=False):
    output_img = img.copy()
    cv2.imwrite("output_image_in_utils.jpg", output_img)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    if invert_colors:
        _, thresh = cv2.threshold(
            blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
    else:
        _, thresh = cv2.threshold(
                        blurred,
                        0,
                        255,
                        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
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
            cx, cy = cnt[0][0]

        text = f"A:{area:.1f} mm², P:{perimeter:.1f} mm"

        # Annotate on image
        cv2.putText(output_img, text, (cx - 50, cy), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (255, 0, 0), 2)
    return output_img
