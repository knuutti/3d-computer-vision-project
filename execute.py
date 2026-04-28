import calibration as cal
import image_analysis as ia
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

class SceneDetails:
    def __init__(self, block_red, block_green, block_blue, target_red, target_green, target_blue, robot_center, robot_front, robot_orientation):
        self.block_red = block_red
        self.block_green = block_green
        self.block_blue = block_blue
        self.target_red = target_red
        self.target_green = target_green
        self.target_blue = target_blue
        self.robot_center = robot_center
        self.robot_front = robot_front
        self.robot_orientation = robot_orientation

def calculate_distance(current, target):

    distance = np.linalg.norm(current - target)

    return distance

def calculate_angle(c, f, t):
    # c is a robot center
    # f is a robot front
    # t is a target destination

    i = np.array([f[0] - c[0], f[1] - c[1]])  # A vector from robot center to robot front
    j = np.array([t[0] - c[0], t[1] - c[1]])  # A vector from robot center to target destination

    angle = np.degrees(np.arccos((np.dot(i.T, j) / (np.linalg.norm(i) * np.linalg.norm(j)))))
    
    a = (c[1] - f[1]) / (c[0] - f[0])
    b = c[1] - a * c[0]

    if t[1] < a * t[0] + b:
        angle = -angle        
    
    return angle

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
        cube1_position = scene.block_green
        cube2_position = scene.block_blue
    elif block == "green":
        block_pos = scene.block_green
        target_pos = scene.target_green
        cube1_position = scene.block_red
        cube2_position = scene.block_blue
    elif block == "blue":
        block_pos = scene.block_blue
        target_pos = scene.target_blue
        cube1_position = scene.block_red
        cube2_position = scene.block_green
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

    return (distance, angle, target_distance, target_angle, cube1_position, cube2_position, block_pos, target_pos)

# This is for instructions, not robot position updating
def get_turn_angle(current_orientation, target_angle):
    turn_angle = np.degrees(target_angle - current_orientation)
    if abs(turn_angle) > 180:
        turn_angle -= np.sign(turn_angle) * 360
    return -1*turn_angle

def point_to_segment_distance(point, seg_start, seg_end):
    seg_vec = seg_end - seg_start
    seg_len_sq = np.dot(seg_vec, seg_vec)
    if seg_len_sq == 0:
        return np.linalg.norm(point - seg_start)
    t = np.clip(np.dot(point - seg_start, seg_vec) / seg_len_sq, 0.0, 1.0)
    return np.linalg.norm(point - (seg_start + t * seg_vec))


def is_path_clear(seg_start, seg_end, obstacles, min_clearance=170.0):
    for obs in obstacles:
        if obs is None:
            continue
        if point_to_segment_distance(obs, seg_start, seg_end) < min_clearance:
            return False
    return True


def find_temp_point(start_pos, dest_pos, current_orientation, obstacles, min_clearance=170.0):
    results = []
    for abs_angle_deg in range(180):
        signs = [1] if abs_angle_deg == 0 else [1, -1]
        for sign in signs:
            angle_deg = sign * abs_angle_deg
            direction = current_orientation + np.radians(angle_deg)
            for dist_mm in range(0, 501, 10):
                temp = start_pos + dist_mm * np.array([np.cos(direction), np.sin(direction)])
                if (is_path_clear(start_pos, temp, obstacles, min_clearance) and
                        is_path_clear(temp, dest_pos, obstacles, min_clearance)):
                    results.append((angle_deg, dist_mm, temp))
    
    if results:
        return min(results, key=lambda r: abs(r[0]) * r[1])
    return None


def navigate_to(dest_pos, obstacles, scene, dist_adjustment=0):
    instructions = ""
    temp_result = find_temp_point(scene.robot, dest_pos, scene.robot_orientation, obstacles)
    if temp_result is not None:
        angle_deg, dist_mm, _ = temp_result
        if angle_deg != 0 and dist_mm != 0:
            new_dir = scene.robot_orientation + np.radians(angle_deg)
            instructions += f"turn({get_turn_angle(scene.robot_orientation, new_dir):.1f});"
            scene.robot_orientation = new_dir
        if dist_mm != 0:
            instructions += f"go({dist_mm / 10:.1f});"
            scene.robot = scene.robot + dist_mm * np.array([np.cos(scene.robot_orientation), np.sin(scene.robot_orientation)])

    vec = dest_pos - scene.robot
    dist = np.linalg.norm(vec)
    angle = np.arctan2(vec[1], vec[0])
    instructions += f"turn({get_turn_angle(scene.robot_orientation, angle):.1f});"
    scene.robot_orientation = angle
    go_dist_mm = dist - dist_adjustment
    instructions += f"go({go_dist_mm / 10:.1f});"
    scene.robot = scene.robot + go_dist_mm * np.array([np.cos(scene.robot_orientation), np.sin(scene.robot_orientation)])

    return instructions


def write_instructions_for_moving_block(metrics, scene, cube_color):
    *_, cube1_position, cube2_position, cube_pos, target_pos = metrics
    obstacles = [cube1_position, cube2_position]

    instructions = navigate_to(cube_pos, obstacles, scene, dist_adjustment=0)
    instructions += "grab();"
    instructions += navigate_to(target_pos, obstacles, scene, dist_adjustment=100)
    instructions += "let_go();"
    instructions += "go(-10);"
    scene.robot -= 200 * np.array([np.cos(scene.robot_orientation), np.sin(scene.robot_orientation)])

    if cube_color == "red":
        scene.block_red = scene.target_red
    elif cube_color == "green":
        scene.block_green = scene.target_green
    elif cube_color == "blue":
        scene.block_blue = scene.target_blue

    return instructions, scene

def get_instructions(blocks, scene):
    instructions = ""
    
    # TODO
    
    robot_center_coords = np.array([scene.robot_center]).T
    robot_front_coords = np.array([scene.robot_front]).T

    block_coords = np.array([
        scene.block_red,
        scene.block_green,
        scene.block_blue
    ])

    target_coords = np.array([
        scene.target_red,
        scene.target_green,
        scene.target_blue
    ])

    for i in range(len(block_coords)):
        distance = calculate_distance(robot_center_coords, block_coords[i,:])
        distance /= 10  # Convert to cm
        angle = calculate_angle(robot_center_coords, robot_front_coords, block_coords[i,:])
        instructions += f"turn({angle}); go({distance}); grab(); "
        robot_center_coords = block_coords[i,:]
        robot_front_x = scene.robot_center[0] + 120 * np.cos(robot_front_coords[1] - robot_center_coords[1], robot_front_coords[0] - robot_center_coords[0])
        robot_front_y = scene.robot_center[1] + 120 * np.sin(robot_front_coords[1] - robot_center_coords[1], robot_front_coords[0] - robot_center_coords[0])
        robot_front_coords = [robot_front_x, robot_front_y]
        
        distance = calculate_distance(robot_center_coords, target_coords[i,:])
        distance -= 120 # Length of the grabber
        distance /= 10  # Convert to cm
        angle = calculate_angle(robot_center_coords, robot_front_coords, target_coords[i,:])
        instructions += f"turn({angle}); go({distance}); let_go(); go(-240); "
        robot_center_x = scene.robot_center[0] - 240 * np.cos(robot_front_coords[1] - robot_center_coords[1], robot_front_coords[0] - robot_center_coords[0])
        robot_center_y = scene.robot_center[1] - 240 * np.sin(robot_front_coords[1] - robot_center_coords[1], robot_front_coords[0] - robot_center_coords[0])
        robot_center_coords = [robot_center_x, robot_center_y]
        robot_front_x = scene.robot_center[0] + 120 * np.cos(robot_front_coords[1] - robot_center_coords[1], robot_front_coords[0] - robot_center_coords[0])
        robot_front_y = scene.robot_center[1] + 120 * np.sin(robot_front_coords[1] - robot_center_coords[1], robot_front_coords[0] - robot_center_coords[0])
        robot_front_coords = [robot_front_x, robot_front_y]

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
        robot_center=robot_center,
        robot_front=robot_front,
        robot_orientation=robot_orientation
    )

# Visualization function, shows the projected scene
def plot_scene(scene):
    fig, ax = plt.subplots()
    ax.scatter(scene.block_red[0], scene.block_red[1], c='r', s=100, marker='s', label='Block Red')
    ax.scatter(scene.block_green[0], scene.block_green[1], c='g', s=100, marker='s', label='Block Green')
    ax.scatter(scene.block_blue[0], scene.block_blue[1], c='b', s=100, marker='s', label='Block Blue')
    ax.scatter(scene.target_red[0], scene.target_red[1], c='r', s=100, marker='o', label='Target Red')
    ax.scatter(scene.target_green[0], scene.target_green[1], c='g', s=100, marker='o', label='Target Green')
    ax.scatter(scene.target_blue[0], scene.target_blue[1], c='b', s=100, marker='o', label='Target Blue')
    ax.scatter(scene.robot_center[0], scene.robot_center[1], c='m', s=100, marker='o', label='Robot')
    # robot front is 120 mm from the center, so we can plot it as a line
    robot_front_x = scene.robot_center[0] + 120 * np.cos(scene.robot_orientation)
    robot_front_y = scene.robot_center[1] + 120 * np.sin(scene.robot_orientation)
    ax.plot([scene.robot_center[0], robot_front_x], [scene.robot_center[1], robot_front_y], c='m', label='Robot Orientation')
    ax.legend()
    ax.axis('equal')
    ax.grid()
    plt.show()