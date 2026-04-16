import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

import image_processing as ip

def get_cube_location(img_hsv, color):
    mask = ip.color_threshold(img_hsv, color)
    blobs = ip.analyze_blobs(mask)
    cube, _ = ip.classify_blobs(blobs)

    return cube[0].centroid

def get_calibration_points(img, cube_calibration = True, scale_factor = 0.1):
    img_scaled_gray = cv.resize(cv.cvtColor(img, cv.COLOR_BGR2GRAY), (0, 0), fx=scale_factor, fy=scale_factor)

    corners = cv.findChessboardCorners(img_scaled_gray, (6,8), None)[1]
    
    corners = corners.reshape(-1, 2)
    corners *= (1/scale_factor)

    points_2d = np.array([
        corners[0],
        corners[5],
        corners[42],
        corners[47]
    ])

    points_3d = np.array([
        [5*40, 0, 0],
        [0, 0, 0],
        [5*40, 7*40, 0],
        [0, 7*40, 0]
    ])

    if cube_calibration:
        img_hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
        blue_cube_2d = get_cube_location(img_hsv, ip.Color.BLUE)
        green_cube_2d = get_cube_location(img_hsv, ip.Color.GREEN)
        points_2d = np.vstack((points_2d, blue_cube_2d, green_cube_2d))
        points_3d = np.vstack((points_3d, [-0.5*40, 7.5*40, 1*40], [5.5*40, -0.5*40, 2*40]))
    else:
        points_2d = np.vstack((points_2d, corners[45], corners[27]))
        points_3d = np.vstack((points_3d, [2*40, 7*40, 0], [2*40, 4*40, 0]))

    return points_2d, points_3d

def calibrate_camera(img, cube_calibration=False):
    M = None

    points_2d, points_3d = get_calibration_points(img, cube_calibration=cube_calibration)
    M = calibrate_norm(points_2d, points_3d)

    return M

def project_point(M, points_2d, z):
    points3d_estimated = []
    for i in range(points_2d.shape[0]):
        x, y = points_2d[i]
        
        A = np.array([
            [M[0, 0] - x * M[2, 0], M[0, 1] - x * M[2, 1]],
            [M[1, 0] - y * M[2, 0], M[1, 1] - y * M[2, 1]]
        ])
        
        b = np.array([
            x * (M[2, 2] * z[i] + M[2, 3]) - (M[0, 2] * z[i] + M[0, 3]), 
            y * (M[2, 2] * z[i] + M[2, 3]) - (M[1, 2] * z[i] + M[1, 3])
        ])

        xy = np.linalg.solve(A, b)
        points3d_estimated.append([xy[0], xy[1], z[i]])

    points3d_estimated = np.array(points3d_estimated)

    return points3d_estimated

def calibrate_from_points(points2d, points3d):

    A = np.zeros((points2d.shape[0] * 2, 12))

    for i in range(points2d.shape[0]):
        x, y = points2d[i]
        X, Y, Z = points3d[i]

        # Composing matrix A
        A[2*i] = [X, Y, Z, 1, 0, 0, 0, 0, -x*X, -x*Y, -x*Z, -x]
        A[2*i + 1] = [0, 0, 0, 0, X, Y, Z, 1, -y*X, -y*Y, -y*Z, -y]

    # Solve m with SVD
    _, _, Vt = np.linalg.svd(A)
    m = Vt[-1]

    return m.reshape(3, 4)

def calibrate_norm(points2d, points3d):

    x_avg = np.mean(points2d[:, 0])
    y_avg = np.mean(points2d[:, 1])
    d_avg = np.mean(np.sqrt((points2d[:, 0] - x_avg)**2 + (points2d[:, 1] - y_avg)**2))

    T = np.array([[np.sqrt(2)/d_avg, 0, -np.sqrt(2)*x_avg/d_avg],
                  [0, np.sqrt(2)/d_avg, -np.sqrt(2)*y_avg/d_avg],
                  [0, 0, 1]])
    
    X_avg = np.mean(points3d[:, 0])
    Y_avg = np.mean(points3d[:, 1])
    Z_avg = np.mean(points3d[:, 2])
    D_avg = np.mean(np.sqrt((points3d[:, 0] - X_avg)**2 + (points3d[:, 1] - Y_avg)**2 + (points3d[:, 2] - Z_avg)**2))

    U = np.array([[np.sqrt(3)/D_avg, 0, 0, -np.sqrt(3)*X_avg/D_avg],
                  [0, np.sqrt(3)/D_avg, 0, -np.sqrt(3)*Y_avg/D_avg],
                  [0, 0, np.sqrt(3)/D_avg, -np.sqrt(3)*Z_avg/D_avg],
                  [0, 0, 0, 1]])

    normalized_2d_pts = (T @ np.hstack((points2d, np.ones((points2d.shape[0], 1)))).T).T[:, :2]
    normalized_3d_pts = (U @ np.hstack((points3d, np.ones((points3d.shape[0], 1)))).T).T[:, :3]

    M = calibrate_from_points(normalized_2d_pts, normalized_3d_pts)

    denormalized_M = np.linalg.inv(T) @ M @ U

    return denormalized_M

if __name__ == "__main__":
 
    img = cv.imread("calibration/img_01.png")
    M = calibrate_camera(img, cube_calibration=True)

    