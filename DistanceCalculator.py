import cv2
import cv2.aruco as aruco
import numpy as np

# Load previously saved camera calibration data
camera_matrix = np.load("camera_matrix.npy")
dist_coeffs = np.load("dist_coeffs.npy")

# Define the ArUco dictionary
aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_6X6_250)
parameters = aruco.DetectorParameters()

# Define the real-world size of the marker (in meters)
marker_size = 0.0254  # 5 cm

def calculate_distance_3d(tvec1, tvec2):
    # Calculate the Euclidean distance between two points in 3D space (tvec1 and tvec2)
    distance = np.linalg.norm(tvec1 - tvec2)
    return distance

# Open the default webcam (index 0)
cap = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect ArUco markers in the grayscale frame
    corners, ids, rejected = aruco.detectMarkers(gray, aruco_dict, parameters=parameters)

    # If markers are detected
    if ids is not None:
        # Draw detected markers
        aruco.drawDetectedMarkers(frame, corners, ids)

        # Estimate pose of each marker (rotation and translation vectors)
        rvecs, tvecs, _ = aruco.estimatePoseSingleMarkers(corners, marker_size, camera_matrix, dist_coeffs)

        for i, id_ in enumerate(ids):
            tvec = tvecs[i][0]  # Translation vector of the marker
            
            # Display the marker ID and its coordinates on the frame
            x, y, z = tvec
            cv2.putText(frame, f"ID {id_[0]}: ({x:.6f}, {y:.6f}, {z:.6f})", 
                        (int(corners[i][0][0][0]), int(corners[i][0][0][1]) - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
            
        # If two or more markers are detected, calculate the 3D distance
        if len(ids) > 1:
            tvec1 = tvecs[0][0]  # 3D position of the first marker
            tvec2 = tvecs[1][0]  # 3D position of the second marker

            # Calculate the distance between the two markers
            distance = calculate_distance_3d(tvec1, tvec2)
            
            # Display the distance on the frame
            cv2.putText(frame, f"Distance: {distance:.6f} meters", (50, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # Print coordinates of both markers and their distance
            print(f"Marker 1 (x, y, z): ({tvec1[0]:.6f}, {tvec1[1]:.6f}, {tvec1[2]:.6f})")
            print(f"Marker 2 (x, y, z): ({tvec2[0]:.6f}, {tvec2[1]:.6f}, {tvec2[2]:.6f})")
            print(f"Distance between markers: {distance:.6f} meters")

    # Display the resulting frame
    cv2.imshow('ArUco Marker Detection', frame)

    # Press 'q' to quit manually
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close OpenCV windows
cap.release()
cv2.destroyAllWindows()


# Marker 1 (x, y, z): (-0.058925, 0.054567, 0.414739)
# Marker 2 (x, y, z): (0.076661, -0.076845, 0.427965)
# Distance between markers: 0.189281 meters

# Marker 1 (x, y, z): (0.073789, 0.056134, 0.411582)
# Marker 2 (x, y, z): (-0.057209, -0.076836, 0.432204)
# Distance between markers: 0.187795 meters

# Marker 1 (x, y, z): (0.046523, 0.080249, 0.401304)
# Marker 2 (x, y, z): (-0.016696, -0.094922, 0.426543)
# Distance between markers: 0.187933 meters
