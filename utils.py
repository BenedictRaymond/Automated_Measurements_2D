import cv2
import ezdxf


def crop_image(length, width, img):
    img = img[length:1080 - length, width:1920 - width]
    return img


def find_contours(img, pixel_mm_ratio, invert_colors=False):
    output_img = img.copy()
    cv2.imwrite("output_image_in_utils.jpg", output_img)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (31, 31), 0)

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

        if area < perimeter:
            continue

        # Draw the contour
        cv2.drawContours(output_img, [cnt], -1, (0, 255, 0), 1)

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
                    0.5, (255, 0, 0), 1)
    return output_img, contours


def contours_to_dxf(contours, pixel_mm_ratio, output_file="output.dxf"):
    """
    Convert OpenCV contours to DXF format.

    Args:
        contours: List of contours from cv2.findContours
        pixel_mm_ratio: Conversion ratio from pixels to millimeters
        output_file: Name of the output DXF file
    """
    # Create a new DXF document
    doc = ezdxf.new('R2010')  # AutoCAD 2010 format
    msp = doc.modelspace()

    # Process each contour
    for contour in contours:
        # Convert contour points to numpy array and scale to mm
        points = contour.reshape(-1, 2)
        points = points / pixel_mm_ratio

        # Create a polyline for each contour
        if len(points) > 2:  # Ensure we have enough points for a shape
            # Convert points to list of tuples
            points_list = [(float(x), float(y), 0) for x, y in points]

            # Add the polyline to the DXF
            msp.add_lwpolyline(points_list)

    # Save the DXF file
    doc.saveas(output_file)
    return output_file
