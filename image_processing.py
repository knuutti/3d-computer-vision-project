import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

class Color:
    RED = 0
    GREEN = 1
    BLUE = 2
    YELLOW = 3
    PURPLE = 4

class Blob:
    def __init__(self, area, centroid, holeFactor, xmin, ymin, width, height, is_mask_in_centroid, x_max_point, x_min_point, y_max_point, y_min_point):
        self.area = area
        self.centroid = centroid
        self.holeFactor = holeFactor
        self.xmin = xmin
        self.ymin = ymin
        self.width = width
        self.height = height
        self.is_mask_in_centroid = is_mask_in_centroid
        self.x_max_point = x_max_point
        self.x_min_point = x_min_point
        self.y_max_point = y_max_point
        self.y_min_point = y_min_point


def classify_blobs(blobs):
    if len(blobs) == 0:
        return [], []
    elif len(blobs) == 1:
        return blobs, []
    else:
        sorted_indices = np.argsort([blob.holeFactor for blob in blobs])[::-1]
        cube_face_blob = blobs[sorted_indices[0]]
        goal_blob = blobs[sorted_indices[1]]
        return [cube_face_blob], [goal_blob]

# Function for getting a binary mask for a specific color in an image
def color_threshold(img, color):
    if color == Color.RED:
        lower = np.array([0, 200, 50])
        upper = np.array([10, 255, 200])
    elif color == Color.GREEN:
        lower = np.array([50, 20, 30])
        upper = np.array([120, 150, 70])
    elif color == Color.BLUE:
        lower = np.array([100, 100, 40])
        upper = np.array([140, 200, 160])
    elif color == Color.YELLOW:
        lower = np.array([10, 200, 140])
        upper = np.array([80, 255, 255])
    elif color == Color.PURPLE:
        lower = np.array([140, 100, 40])
        upper = np.array([200, 200, 160])
    else:
        raise ValueError("Not a valid color, check if index is correct")

    mask = cv.inRange(img, lower, upper)
    mask = clean_mask(mask)

    if color==Color.GREEN:
        plt.imshow(mask)
        plt.title("Green mask")
        plt.show()
    
    return mask

# Function for cleaning up a binary mask using morphological operations
def clean_mask(mask, kernel_size=5, iter_fill=10, iter_clean=1):
    
    kernel = np.ones((kernel_size, kernel_size), np.uint8)

    mask = cv.erode(mask, kernel, iterations=iter_clean)
    mask = cv.dilate(mask, kernel, iterations=iter_clean)

    mask = cv.dilate(mask, kernel, iterations=iter_fill)
    mask = cv.erode(mask, kernel, iterations=iter_fill)
    
    return mask

# Function for analyzing blobs in a binary mask
def analyze_blobs(mask, min_area=100):
    contours, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    if len(contours) == 0:
        return []

    contour_areas = [cv.contourArea(c) for c in contours]
    sorted_indices = np.argsort(contour_areas)[::-1]
    blobs = []
    for i, idx in enumerate(sorted_indices):
        if i >= 2:
            break
        area = contour_areas[idx]
        is_mask_in_centroid = mask[int(np.mean(contours[idx][:, 0, 1])), int(np.mean(contours[idx][:, 0, 0]))] > 0
        if area >= min_area:
            blobs.append(Blob(
                area=area,
                centroid=np.mean(contours[idx], axis=0)[0],
                holeFactor=cv.contourArea(cv.convexHull(contours[idx])) / area, # feature for separating cubes and targets
                xmin=np.min(contours[idx][:, 0, 0]),
                ymin=np.min(contours[idx][:, 0, 1]),
                width=np.max(contours[idx][:, 0, 0]) - np.min(contours[idx][:, 0, 0]),
                height=np.max(contours[idx][:, 0, 1]) - np.min(contours[idx][:, 0, 1]),
                is_mask_in_centroid=is_mask_in_centroid,
                x_max_point=contours[idx][np.argmax(contours[idx][:, 0, 0])][0],
                x_min_point=contours[idx][np.argmin(contours[idx][:, 0, 0])][0],
                y_max_point=contours[idx][np.argmax(contours[idx][:, 0, 1])][0],
                y_min_point=contours[idx][np.argmin(contours[idx][:, 0, 1])][0]
            ))

    return blobs