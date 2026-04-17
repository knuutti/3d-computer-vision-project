import calibration as cal
import image_analysis as ia
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

class SceneDetails:
    def __init__(self, block_red, block_green, block_blue, target_red, target_green, target_blue, robot, robot_orientation):
        self.block_red = block_red
        self.block_green = block_green
        self.block_blue = block_blue
        self.target_red = target_red
        self.target_green = target_green
        self.target_blue = target_blue
        self.robot = robot
        self.robot_orientation = robot_orientation

def move_block(blocks, img, calib) -> str:
    """
    blocks: list of strings indicating which block to move and in which order
    img: test image containing the scene
    calib: calibration matrix

    output: instruction for the robot
    """

    points_3d_estimated = get_points_from_image(img, calib)
    scene = get_scene_details(points_3d_estimated)

    return get_instructions(blocks, scene)

def get_instructions(blocks, scene):
    instructions = ""
    
    # TODO
    
    return instructions

def get_points_from_image(img, calib):
    img_hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    elements = ia.analyze_image(img_hsv)
    types = [element["type"] for element in elements]
    idx = np.argsort(types)
    elements = [elements[i] for i in idx]
    heights = [element["height"] for element in elements]
    points_2d = np.array([element["centroid"] for element in elements])
    points_3d_estimated = cal.project_point(calib, points_2d, z=heights)
    return points_3d_estimated[:, :2]

def get_scene_details(points_3d):
    robot_front = points_3d[6]
    robot_center = points_3d[7]

    # Orientation of the robot: angle between robot front and robot center
    robot_orientation = np.arctan2(robot_front[1] - robot_center[1], robot_front[0] - robot_center[0])

    return SceneDetails(
        block_red=points_3d[0],
        block_green=points_3d[1],
        block_blue=points_3d[2],
        target_red=points_3d[3],
        target_green=points_3d[4],
        target_blue=points_3d[5],
        robot=robot_center,
        robot_orientation=robot_orientation
    )

def plot_scene(scene):
    fig, ax = plt.subplots()
    ax.scatter(scene.block_red[0], scene.block_red[1], c='r', s=100, marker='s', label='Block Red')
    ax.scatter(scene.block_green[0], scene.block_green[1], c='g', s=100, marker='s', label='Block Green')
    ax.scatter(scene.block_blue[0], scene.block_blue[1], c='b', s=100, marker='s', label='Block Blue')
    ax.scatter(scene.target_red[0], scene.target_red[1], c='r', s=100, marker='o', label='Target Red')
    ax.scatter(scene.target_green[0], scene.target_green[1], c='g', s=100, marker='o', label='Target Green')
    ax.scatter(scene.target_blue[0], scene.target_blue[1], c='b', s=100, marker='o', label='Target Blue')
    ax.scatter(scene.robot[0], scene.robot[1], c='m', s=100, marker='o', label='Robot')
    # robot front is 120 mm from the center, so we can plot it as a line
    robot_front_x = scene.robot[0] + 120 * np.cos(scene.robot_orientation)
    robot_front_y = scene.robot[1] + 120 * np.sin(scene.robot_orientation)
    ax.plot([scene.robot[0], robot_front_x], [scene.robot[1], robot_front_y], c='m', label='Robot Orientation')
    ax.legend()
    ax.axis('equal')
    ax.grid()
    plt.show()