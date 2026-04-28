import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import time

import image_processing as ip

def get_cube_location(img_hsv, color):
    mask = ip.color_threshold(img_hsv, color)
    blobs = ip.analyze_blobs(mask)
    cube, _ = ip.classify_blobs(blobs)

    return cube[0].centroid

def get_cube_face_corners(img_hsv, color):
    mask = ip.color_threshold(img_hsv, color)
    blobs = ip.analyze_blobs(mask)
    cube, _ = ip.classify_blobs(blobs)

    blob = cube[0]
    corners_2d = np.array([blob.y_min_point, blob.x_max_point, blob.y_max_point, blob.x_min_point])
    return corners_2d

def get_calibration_points(img, auto=True, scale_factor = 0.1):
    img_scaled_gray = cv.resize(cv.cvtColor(img, cv.COLOR_BGR2GRAY), (0, 0), fx=scale_factor, fy=scale_factor)

    points_2d = None
    points_3d = None
    
    if auto:
        corners = cv.findChessboardCorners(img_scaled_gray, (6,8), None)[1]
        corners = corners.reshape(-1, 2)
        corners *= (1/scale_factor)
        
        x_coords = np.arange(5*40, -40, -40)
        y_coords = np.arange(0, 8*40, 40)

        points_3d = []
        for i, _ in enumerate(corners):
            x_idx = i % 6
            y_idx = i // 6
            points_3d.append([x_coords[x_idx], y_coords[y_idx], 0])
        points_2d = corners 
        points_3d = np.array(points_3d)
        img_hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
        blue_cube_corners_2d = get_cube_face_corners(img_hsv, ip.Color.BLUE)
        green_cube_corners_2d = get_cube_face_corners(img_hsv, ip.Color.GREEN)
        points_2d = np.vstack((points_2d, blue_cube_corners_2d, green_cube_corners_2d))
        points_3d = np.vstack((points_3d, 
            [6*40, 0, 80],
            [6*40, 40, 80],
            [7*40, 40, 80],
            [7*40, 0, 80],
            [0, 8*40, 40],
            [0, 9*40, 40],
            [40, 9*40, 40],
            [40, 8*40, 40]))

    else:
        points_2d = np.array(select_points(cv.cvtColor(img, cv.COLOR_BGR2GRAY)))
        # 3D points for manual selection: 
        # 4x checkboard corners: Z=0, XY locations (40,40), (40,8*40), (6*40,8*40), (6*40,40)
        # Blue cube top face corners: Z=80, XY locations (6*40, 0), (6*40,40), (7*40,40), (7*40,0)
        # Green cube top face corners: Z=40, XY locations (0,8*40), (0,9*40), (40,9*40), (40,8*40)
        points_3d = np.array([
            [40, 40, 0],
            [40, 8*40, 0],
            [6*40, 8*40, 0],
            [6*40, 40, 0],
            [6*40, 0, 80],
            [6*40, 40, 80],
            [7*40, 40, 80],
            [7*40, 0, 80],
            [0, 8*40, 40],
            [0, 9*40, 40],
            [40, 9*40, 40],
            [40, 8*40, 40]
        ])

    return points_2d, points_3d

def image_points_to_world_at_z(M, points_2d, z):
    n = len(points_2d)
    A = np.linalg.inv(M[:, [0, 1, 3]])
    b = M[:, 2:3]
    points_h = np.concatenate((points_2d.T, [[1] * n]), 0)
    out = A @ (points_h - z * b)
    return (out[:-1] / out[-1]).T

def project_point(M, points, z):
    return image_points_to_world_at_z(M, points, z)

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

def img_to_world(points, calib, z):
    points = np.concatenate((points.T, [[1] * n]), 0)
    # Backward-compatible alias. Prefer image_points_to_world_at_z.
    return image_points_to_world_at_z(calib, points, z)

def calibrate_camera(imgs, mode="auto"):
    img = imgs[0] # Only use the first image for calibration
    points_2d, points_3d = get_calibration_points(img, auto=(mode=="auto"))
    M = calibrate_norm(points_2d, points_3d)
    return M

def select_points(img, scale_factor=.3):
    points = []

    def click_event(event, x, y, flags, param):
        if event == cv.EVENT_LBUTTONDOWN:
            original_x = int(x / scale_factor)
            original_y = int(y / scale_factor)
            points.append((original_x, original_y))

            cv.circle(param, (x, y), 3, (0, 255, 0), -1)
            cv.imshow("Image", param)

    # Resize image for display
    height, width = img.shape
    new_height, new_width = int(height * scale_factor), int(width * scale_factor)
    resized_img = cv.cvtColor(cv.resize(img, (new_width, new_height), interpolation=cv.INTER_LINEAR), cv.COLOR_GRAY2BGR)

    cv.imshow("Image", resized_img)
    cv.setMouseCallback("Image", click_event, resized_img)

    # Stop when esc pressed
    while True:
        key = cv.waitKey(20) & 0xFF
        if key == 27:  # ESC key to break
            break
        if cv.getWindowProperty("Image", cv.WND_PROP_VISIBLE) < 1:  # Check if window is closed
            break
    cv.destroyAllWindows()

    plt.figure(figsize=(18, 9))
    plt.imshow(cv.cvtColor(resized_img, cv.COLOR_BGR2RGB))
    plt.tight_layout()
    plt.axis("off")
    plt.show()
    return points

if __name__ == "__main__":
    img = cv.imread("calibration/calib2.png")
    imgs = [img]

    # Visualize points selected in automatic calibration
    points_2d, points_3d = get_calibration_points(img, auto=True)
    plt.imshow(img)
    plt.scatter(points_2d[:, 0], points_2d[:, 1], c='r', marker='x')
    # show index next to each point
    for i, (x, y) in enumerate(points_2d):
        plt.text(x, y, str(i), color='yellow', fontsize=12)
    plt.title("Selected calibration points")


    plt.show()

    M = calibrate_camera(imgs, mode="auto")