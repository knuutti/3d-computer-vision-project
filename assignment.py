from calibration import get_calibration_points, calibrate_norm
from execute import get_points_from_image, get_scene_details, get_instructions
import cv2 as cv

def calibrate_camera(imgs, mode="auto"):
    img = imgs[0]
    points_2d, points_3d = get_calibration_points(img, auto=(mode=="auto"))
    M = calibrate_norm(points_2d, points_3d)
    return M

def move_block(blocks, img, calib):
    points_3d_estimated = get_points_from_image(img, calib)
    scene = get_scene_details(points_3d_estimated)
    instructions = get_instructions(blocks, scene)
    return instructions

def main():
    # Parameters
    calib_img_path = "img/calibration.png"
    test_img_path = "img/test.png"
    blocks_to_move = ["blue", "red", "green"]

    # Calibration
    img_calib = cv.imread(calib_img_path)
    M = calibrate_camera([img_calib], mode="auto")
    
    # Execute
    img_test = cv.imread(test_img_path)
    print(move_block(blocks_to_move, img_test, M))

if __name__ == "__main__":
    main()
    