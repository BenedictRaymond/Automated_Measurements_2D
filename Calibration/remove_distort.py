import cv2
import numpy as np


def remove_distortion(img_path, calibration_data_path):
    """
    Remove distortion from an image using pre-calibrated camera parameters.
    :param img_path: Path to the input image.
    :param calibration_data_path: Path to the calibration data file.
    :return: Undistorted image.
    """
    # Load the calibration data
    calibration_data = np.load(calibration_data_path)
    mtx = calibration_data['camera_matrix']
    dist = calibration_data['distortion_coefficients']

    # Load the image
    img = cv2.imread(img_path)
    if img is None:
        raise ValueError(f"Image not found at {img_path}")

    h, w = img.shape[:2]

    # Compute new optimal camera matrix (better cropping)
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(
        mtx, dist, (w, h), 1, (w, h)
    )

    # Apply undistortion
    undistorted = cv2.undistort(img, mtx, dist, None, newcameramtx)

    return undistorted


img = remove_distortion(
    "Calibration/data/image_03.jpg",
    r"Calibration/calibration_data.npz"
)

cv2.imshow("Undistorted Image", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
