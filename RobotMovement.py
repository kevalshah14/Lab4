import numpy as np
from scipy.spatial.transform import Rotation as R

def map_aruco_to_robot(known_aruco_coords, known_robot_coords, unknown_aruco_coords):
    # Convert to numpy arrays
    known_aruco_coords = np.array(known_aruco_coords)
    known_robot_coords = np.array(known_robot_coords)
    unknown_aruco_coords = np.array(unknown_aruco_coords)
    
    # Calculate the transformation matrix
    centroid_aruco = np.mean(known_aruco_coords, axis=0)
    centroid_robot = np.mean(known_robot_coords, axis=0)
    
    H = np.dot((known_aruco_coords - centroid_aruco).T, (known_robot_coords - centroid_robot))
    U, S, Vt = np.linalg.svd(H)
    R_matrix = np.dot(Vt.T, U.T)
    
    if np.linalg.det(R_matrix) < 0:
        Vt[-1, :] *= -1
        R_matrix = np.dot(Vt.T, U.T)
    
    t = centroid_robot.T - np.dot(R_matrix, centroid_aruco.T)
    
    # Apply the transformation to the unknown ArUco coordinates
    transformed_coords = np.dot(R_matrix, unknown_aruco_coords.T).T + t.T
    
    return transformed_coords


known_aruco_coords = [(1, 2), (3, 4), (5, 6)]
known_robot_coords = [(2, 3), (4, 5), (6, 7)]
unknown_aruco_coords = [(7, 8)]

robot_coords = map_aruco_to_robot(known_aruco_coords, known_robot_coords, unknown_aruco_coords)
print(robot_coords)