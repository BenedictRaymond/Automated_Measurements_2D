import cv2

# Define the checkerboard dimensions (number of inner corners)
CHECKERBOARD = (9, 6)

img = cv2.imread(r'test_img.jpg')
if img is None:
    print("Error: Could not load image. Check the file path.")
    exit()

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Find the chessboard corners
ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, None)

if ret:
    cv2.drawChessboardCorners(img, CHECKERBOARD, corners, ret)
    cv2.imshow('Corners Detected', img)
    cv2.waitKey(0)
cv2.destroyAllWindows()
