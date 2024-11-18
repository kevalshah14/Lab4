import cv2
import numpy as np
import os

# Set up criteria for termination of corner sub-pixel accuracy search
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# Prepare object points for a checkerboard (0,0,0), (1,0,0), (2,0,0) ... (6,5,0) if the board is 7x6
checkerboard_size = (7, 6)  # Number of internal corners per a chessboard row and column
objp = np.zeros((checkerboard_size[0] * checkerboard_size[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:checkerboard_size[0], 0:checkerboard_size[1]].T.reshape(-1, 2)

# Store object points and image points from all the images
objpoints = []  # 3d points in real-world space
imgpoints = []  # 2d points in image plane

images = [f'checkerboard_{i}.jpg' for i in range(10)]  

for fname in images:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Find the chess board corners
    ret, corners = cv2.findChessboardCorners(gray, checkerboard_size, None)

    # If found, add object points, image points (after refining them)
    if ret:
        objpoints.append(objp)

        # Refine corner detection for sub-pixel accuracy
        corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        imgpoints.append(corners2)

        # Draw and display the corners
        cv2.drawChessboardCorners(img, checkerboard_size, corners2, ret)
        cv2.imshow('img', img)
        cv2.waitKey(500)

cv2.destroyAllWindows()

ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

# Save the calibration results
np.save("camera_matrix.npy", camera_matrix)
np.save("dist_coeffs.npy", dist_coeffs)

print("Camera matrix:\n", camera_matrix)
print("Distortion coefficients:\n", dist_coeffs)
