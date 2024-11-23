import numpy as np

def map_aruco_to_robot_2d(known_aruco_coords, known_robot_coords, unknown_aruco_coords):
    """
    Maps 2D ArUco marker coordinates to robot coordinates using similarity transformation
    (rotation, scaling, and translation).

    Parameters:
    - known_aruco_coords: List of tuples [(x1, y1), (x2, y2), ...] in ArUco frame
    - known_robot_coords: List of tuples [(X1, Y1), (X2, Y2), ...] in Robot frame
    - unknown_aruco_coords: List of tuples [(x, y), ...] to be transformed

    Returns:
    - transformed_coords: Numpy array of transformed robot coordinates
    """
    # Convert to numpy arrays
    known_aruco = np.array(known_aruco_coords)
    known_robot = np.array(known_robot_coords)
    unknown_aruco = np.array(unknown_aruco_coords)

    # Check that there are at least two points
    if known_aruco.shape[0] < 2 or known_robot.shape[0] < 2:
        raise ValueError("At least two known points are required for 2D mapping.")

    # Compute centroids
    centroid_aruco = np.mean(known_aruco, axis=0)
    centroid_robot = np.mean(known_robot, axis=0)

    # Center the points
    aruco_centered = known_aruco - centroid_aruco
    robot_centered = known_robot - centroid_robot

    # Compute scaling factor
    norm_aruco = np.linalg.norm(aruco_centered, axis=1)
    norm_robot = np.linalg.norm(robot_centered, axis=1)
    scale = np.mean(norm_robot / norm_aruco)

    # Scale ArUco points
    aruco_scaled = aruco_centered * scale

    # Compute rotation using Singular Value Decomposition (SVD)
    H = aruco_scaled.T @ robot_centered
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T

    # Ensure a proper rotation (no reflection)
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T

    # Compute translation
    t = centroid_robot - R @ centroid_aruco * scale

    # Apply transformation to unknown ArUco coordinates
    unknown_aruco_centered = unknown_aruco - centroid_aruco
    unknown_aruco_scaled = unknown_aruco_centered * scale
    transformed_coords = (R @ unknown_aruco_scaled.T).T + t

    return transformed_coords

# Example usage
if __name__ == "__main__":
    known_aruco_coords = [(0.074111, 0.057119),(-0.057641, -0.077212)]
    known_robot_coords = [(-455.595, -68.97), (-349.752,49.829)]
    unknown_aruco_coords = [(-0.057212, -0.078666)]

    robot_coords = map_aruco_to_robot_2d(known_aruco_coords, known_robot_coords, unknown_aruco_coords)
    print("Transformed Robot Coordinates:", robot_coords)
