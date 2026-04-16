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
    M = calibrate_from_points(points_2d, points_3d)

    return M

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

if __name__ == "__main__":
 
    img = cv.imread("calibration/img_01.png")
    M = calibrate_camera(img, cube_calibration=True)

    