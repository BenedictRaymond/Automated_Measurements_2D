import cv2
import numpy as np
import glob
import os
from Calibration.remove_distort import remove_distortion


def calculate_pixel_mm_ratio(
        image_path, checkerboard_size=(9, 6), square_size_mm=2.75
):
    """
    Calculate pixel to millimeter ratio using checkerboard pattern

    Args:
        image_path: Path to checkerboard image or directory containing images
        checkerboard_size: Tuple of (width, height) internal corners
        square_size_mm: Size of each checkerboard square in millimeters

    Returns:
        pixel_mm_ratio: Average pixels per millimeter
    """

    # Prepare object points (3D points in real world space)
    objp = np.zeros(
        (checkerboard_size[0] * checkerboard_size[1], 3), np.float32
    )
    objp[:, :2] = np.mgrid[
        0:checkerboard_size[0], 0:checkerboard_size[1]
    ].T.reshape(-1, 2)
    objp *= square_size_mm  # Scale by actual square size

    # Determine if input is a single image or directory
    if os.path.isfile(image_path):
        image_files = [image_path]
    elif os.path.isdir(image_path):
        # Get all common image formats
        extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff']
        image_files = []
        for ext in extensions:
            image_files.extend(glob.glob(os.path.join(image_path, ext)))
            image_files.extend(
                glob.glob(os.path.join(image_path, ext.upper()))
            )
    else:
        print(f"Error: {image_path} is not a valid file or directory")
        return None

    if not image_files:
        print("No image files found!")
        return None

    print(f"Processing {len(image_files)} image(s)...")

    pixel_distances = []

    for img_file in image_files:
        print(f"Processing: {os.path.basename(img_file)}")

        # Read image
        img = cv2.imread(img_file)
        if img is None:
            print(f"Could not read image: {img_file}")
            continue

        img = remove_distortion(
            img_file,
            r"Calibration/calibration_data.npz"
        )
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Find the chess board corners
        ret, corners = cv2.findChessboardCorners(gray, checkerboard_size, None)

        if ret:
            print("  ✓ Checkerboard found")

            # Refine corner positions for better accuracy
            criteria = (
                cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001
            )
            corners_refined = cv2.cornerSubPix(
                gray, corners, (11, 11), (-1, -1), criteria
            )

            # Calculate distances between adjacent corners
            # Horizontal distances
            for row in range(checkerboard_size[1]):
                for col in range(checkerboard_size[0] - 1):
                    idx1 = row * checkerboard_size[0] + col
                    idx2 = row * checkerboard_size[0] + col + 1

                    p1 = corners_refined[idx1][0]
                    p2 = corners_refined[idx2][0]

                    pixel_dist = np.linalg.norm(p2 - p1)
                    pixel_distances.append(pixel_dist)

            # Vertical distances
            for row in range(checkerboard_size[1] - 1):
                for col in range(checkerboard_size[0]):
                    idx1 = row * checkerboard_size[0] + col
                    idx2 = (row + 1) * checkerboard_size[0] + col

                    p1 = corners_refined[idx1][0]
                    p2 = corners_refined[idx2][0]

                    pixel_dist = np.linalg.norm(p2 - p1)
                    pixel_distances.append(pixel_dist)

            # Optional: Draw and save visualization
            img_with_corners = img.copy()
            cv2.drawChessboardCorners(
                img_with_corners, checkerboard_size, corners_refined, ret
            )

        else:
            print(
                f"  ✗ Checkerboard not found in {os.path.basename(img_file)}"
            )

    if not pixel_distances:
        print("No valid checkerboard patterns found!")
        return None

    # Calculate average pixel distance for one square
    avg_pixel_distance = np.mean(pixel_distances)
    std_pixel_distance = np.std(pixel_distances)

    # Calculate pixel to mm ratio
    pixel_mm_ratio = avg_pixel_distance / square_size_mm

    print("\nResults:")
    print(f"Number of measurements: {len(pixel_distances)}")
    print(
        f"Average pixel distance per square: {avg_pixel_distance:.2f} ± "
        f"{std_pixel_distance:.2f} pixels"
    )
    print(f"Square size: {square_size_mm} mm")
    print(f"Pixel to MM ratio: {pixel_mm_ratio:.4f} pixels/mm")
    print(f"MM to Pixel ratio: {1/pixel_mm_ratio:.4f} mm/pixel")

    return pixel_mm_ratio


def main():
    # Configuration
    CHECKERBOARD_SIZE = (9, 6)  # Internal corners (width, height)
    SQUARE_SIZE_MM = 2.75       # Size of each square in millimeters

    # Get input path from user
    print("Checkerboard Pixel-to-MM Ratio Calculator")
    print("=========================================")

    image_path = r'Calibration\data'

    # Calculate ratio
    ratio = calculate_pixel_mm_ratio(
        image_path=image_path,
        checkerboard_size=CHECKERBOARD_SIZE,
        square_size_mm=SQUARE_SIZE_MM
    )

    if ratio:
        print(f"\n{'='*50}")
        print(f"FINAL RESULT: {ratio:.4f} pixels per millimeter")
        print(f"{'='*50}")

        # Save result to file
        with open("pixel_mm_ratio.txt", "w") as f:
            f.write(f"{ratio:.6f}\n")

        print("Results saved to pixel_mm_ratio.txt")
    else:
        print("Failed to calculate pixel-to-mm ratio")


if __name__ == "__main__":
    main()
