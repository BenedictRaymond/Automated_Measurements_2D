import cv2
import numpy as np
import glob

# Define the checkerboard dimensions
CHECKERBOARD = (9, 6)

objpoints = []  # 3D points in real-world space
imgpoints = []  # 2D points in image plane

# Prepare the object points (same for all images)
objp = np.zeros((CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)

# Load all calibration images
images = glob.glob(r'Calibration\data\*.jpg')

for fname in images:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Detect corners
    ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, None)
    if ret:
        imgpoints.append(corners)
        objpoints.append(objp)

# Perform camera calibration
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
    objpoints, imgpoints, gray.shape[::-1], None, None
)

# Print results
print("Camera matrix:\n", mtx)
print("Distortion coefficients:\n", dist)
print("Rotation vectors:\n", rvecs)
print("Translation vectors:\n", tvecs)

# Save the calibration results
np.savez(r'Calibration\calibration_data.npz',
         camera_matrix=mtx,
         distortion_coefficients=dist,
         rotation_vectors=rvecs,
         translation_vectors=tvecs)

mean_error = 0
for i in range(len(objpoints)):
    imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
    error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
    mean_error += error

print(f"Mean Re-projection Error: {mean_error / len(objpoints)}")
