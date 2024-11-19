import numpy as np

bottom_left = (-172.926, -2.181, 0.246)
top_right = (170.414,1.896,-9.994)
bottom_right = (-174.758, -2.061, -0.062)
def map_aruco_to_robot(known_aruco_coords, known_robot_coords, unknown_aruco_coords):
    # Convert to numpy arrays
    known_aruco_coords = np.array(known_aruco_coords)
    known_robot_coords = np.array(known_robot_coords)
    unknown_aruco_coords = np.array(unknown_aruco_coords)
    
    # Calculate centroids
    centroid_aruco = np.mean(known_aruco_coords, axis=0)
    centroid_robot = np.mean(known_robot_coords, axis=0)
    
    # Center the points around centroids
    aruco_centered = known_aruco_coords - centroid_aruco
    robot_centered = known_robot_coords - centroid_robot
    
    # Compute rotation matrix using Singular Value Decomposition (SVD)
    H = np.dot(aruco_centered.T, robot_centered)
    U, S, Vt = np.linalg.svd(H)
    R_matrix = np.dot(Vt.T, U.T)
    
    # Ensure the determinant is positive (no reflection)
    if np.linalg.det(R_matrix) < 0:
        Vt[-1, :] *= -1
        R_matrix = np.dot(Vt.T, U.T)
    
    # Compute translation vector
    t = centroid_robot - np.dot(R_matrix, centroid_aruco)
    
    # Apply the transformation to the unknown ArUco coordinates
    transformed_coords = np.dot(R_matrix, unknown_aruco_coords.T).T + t
    
    return transformed_coords

# Example usage
known_aruco_coords = [(0.06, -0.06, 0.4), (-0.08, 0.08, 0.4),(-0.08,0.06,0.4)]
known_robot_coords = [(-172.926, -2.181, 0.246), (170.414,1.896,-9.994), (-174.758, -2.061, -0.062)]
unknown_aruco_coords = [(0.04, -0.06, 0.4)]

robot_coords = map_aruco_to_robot(known_aruco_coords, known_robot_coords, unknown_aruco_coords)
print("Transformed Robot Coordinates:", robot_coords)
