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
    instructions = get_instructions(blocks, scene)
    return instructions

def get_metrics_for_block_moving(block, scene):
    if block == "red":
        block_pos = scene.block_red
        target_pos = scene.target_red
    elif block == "green":
        block_pos = scene.block_green
        target_pos = scene.target_green
    elif block == "blue":
        block_pos = scene.block_blue
        target_pos = scene.target_blue
    else:
        raise ValueError(f"Unknown block color: {block}")
    
    # Calculate the distance and angle from robot to block
    robot_to_block_vec = block_pos - scene.robot
    distance = np.linalg.norm(robot_to_block_vec)
    angle = np.arctan2(robot_to_block_vec[1], robot_to_block_vec[0])

    # Calculate the distance and angle from block to target
    block_to_target_vec = target_pos - block_pos
    target_distance = np.linalg.norm(block_to_target_vec)
    target_angle = np.arctan2(block_to_target_vec[1], block_to_target_vec[0])

    return (distance, angle, target_distance, target_angle)

# This is for intructions, not robot position updating
def get_turn_angle(current_orientation, target_angle):
    turn_angle = np.degrees(target_angle - current_orientation)
    if abs(turn_angle) > 180:
        turn_angle -= np.sign(turn_angle) * 360
    return -1*turn_angle

def write_instructions_for_moving_block(metrics, scene, cube_color):
    
    # Rotate the robot to face the block
    instructions = f"turn({get_turn_angle(scene.robot_orientation, metrics[1]):.1f});"
    scene.robot_orientation = metrics[1]  # Update robot orientation after rotation
    # Move the robot to the block
    instructions += f"go({metrics[0]/10:.1f});"
    scene.robot += metrics[0] * np.array([np.cos(scene.robot_orientation), np.sin(scene.robot_orientation)])  # Update robot position after moving forward
    # Pick up the block
    instructions += "grab();"
    # Rotate the robot to face the target
    instructions += f"turn({get_turn_angle(metrics[1], metrics[3]):.1f});"
    scene.robot_orientation = metrics[3]  # Update robot orientation after rotation
    # Move the robot to the target (should be -120 mm))
    instructions += f"go({((metrics[2]-120)/10):.1f});"
    scene.robot += (metrics[2] - 120) * np.array([np.cos(scene.robot_orientation), np.sin(scene.robot_orientation)])  # Update robot position after moving forward
    # Drop the block
    instructions += "let_go();"
    # Move back 200 mm to clear the area    
    instructions += "go(-20);"
    scene.robot -= 200 * np.array([np.cos(scene.robot_orientation), np.sin(scene.robot_orientation)])  # Update robot position after moving backward

    if cube_color == "red":
        scene.block_red = scene.target_red
    elif cube_color == "green":
        scene.block_green = scene.target_green
    elif cube_color == "blue":
        scene.block_blue = scene.target_blue

    return instructions, scene

def get_instructions(blocks, scene):
    instructions = ""
    
    for block in blocks:
        metrics = get_metrics_for_block_moving(block, scene)
        block_instructions, scene = write_instructions_for_moving_block(metrics, scene, cube_color=block)
        instructions += block_instructions
    
    return instructions[:-1]

def get_points_from_image(img, calib):
    img_hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    elements = ia.analyze_image(img_hsv)
    types = [element["type"] for element in elements]
    idx = np.argsort(types)
    elements = [elements[i] for i in idx]
    heights = [element["height"] for element in elements]
    points_2d = np.array([element["centroid"] for element in elements])
    points_3d_estimated = cal.image_points_to_world_at_z(calib, points_2d, z=heights)
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