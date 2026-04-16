import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

import image_processing as ip
import calibration as cal
from elements import ElementType, ObjectType

class Elements:
    def __init__(self, centroid, color, type):
        self.centroid = centroid
        self.color = color
        self.type = type

def get_mask(img_hsv, color):
    return ip.color_threshold(img_hsv, color)

def analyze_image(img_hsv):

    elements = []

    # Cubes and goals
    for color in [ip.Color.RED, ip.Color.GREEN, ip.Color.BLUE]:
        mask = get_mask(img_hsv, color)
        blobs = ip.analyze_blobs(mask)
        cube_blobs, goal_blobs = ip.classify_blobs(blobs)
        if len(cube_blobs) > 0:
            elements.append(Elements(cube_blobs[0].centroid, color, ElementType.CUBE_FACE))
        if len(goal_blobs) > 0:
            elements.append(Elements(goal_blobs[0].centroid, color, ElementType.GOAL))

    # Robot
    for color in [ip.Color.YELLOW, ip.Color.PURPLE]:
        mask = get_mask(img_hsv, color)
        blobs = ip.analyze_blobs(mask)
        
        if color == ip.Color.YELLOW:
            if len(blobs) > 1:
                # If two yellow blobs, combine them into one
                xmin = min(blobs[0]['xmin'], blobs[1]['xmin'])
                ymin = min(blobs[0]['ymin'], blobs[1]['ymin'])
                width = max(blobs[0]['xmin'] + blobs[0]['width'], blobs[1]['xmin'] + blobs[1]['width']) - xmin
                height = max(blobs[0]['ymin'] + blobs[0]['height'], blobs[1]['ymin'] + blobs[1]['height']) - ymin
                area = blobs[0]['area'] + blobs[1]['area']
                centroid = (xmin + width / 2, ymin + height / 2)
            else:
                xmin = blobs[0].xmin
                ymin = blobs[0].ymin
                width = blobs[0].width
                height = blobs[0].height
                area = blobs[0].area
                centroid = blobs[0].centroid

            yellow_bbox_mask = np.zeros_like(mask)
            if len(blobs) > 0:
                x0 = int(xmin)
                y0 = int(ymin)
                x1 = x0 + int(width)
                y1 = y0 + int(height)
                yellow_bbox_mask[y0:y1, x0:x1] = 255

            yellow_bbox_mask = cv.bitwise_and(get_mask(img_hsv, ip.Color.BLUE), get_mask(img_hsv, ip.Color.BLUE), mask=yellow_bbox_mask)
            yellow_bbox_mask_clean = ip.clean_mask(yellow_bbox_mask, kernel_size=3, iter_fill=5, iter_clean=1)
            blobs_blue_inside_yellow = ip.analyze_blobs(yellow_bbox_mask_clean, min_area=50)
            if len(blobs_blue_inside_yellow) > 0:
                dot_centroids_avg = np.mean([blob.centroid for blob in blobs_blue_inside_yellow[0:min(2, len(blobs_blue_inside_yellow))]], axis=0)
                elements.append(Elements(dot_centroids_avg, ip.Color.BLUE, ElementType.ROBOT_FRONT))
            else:
                elements.append(Elements(centroid, ip.Color.YELLOW, ElementType.ROBOT_FRONT))
        else:
            elements.append(Elements(blobs[0].centroid, color, ElementType.ROBOT_CENTER))

    returned_elements = []
    for element in elements:
        if element.type == ElementType.CUBE_FACE:
            returned_elements.append((element.centroid, element.color, ObjectType.CUBE_RED + element.color))
        elif element.type == ElementType.GOAL:
            returned_elements.append((element.centroid, element.color, ObjectType.GOAL_RED + element.color))
        elif element.type == ElementType.ROBOT_FRONT:
            returned_elements.append((element.centroid, element.color, ObjectType.ROBOT_FRONT))
        elif element.type == ElementType.ROBOT_CENTER:
            returned_elements.append((element.centroid, element.color, ObjectType.ROBOT_CENTER))
        
    return returned_elements

if __name__ == "__main__":
    img = cv.imread("test/img_01.png", cv.IMREAD_COLOR)
    img_hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    elements = analyze_image(img_hsv)
    for element in elements:
        print(f"Centroid: {element[0]}, Type: {element[2]}")