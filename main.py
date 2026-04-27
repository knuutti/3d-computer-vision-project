import calibration as cal
import execute
import cv2 as cv
import numpy as np

def main():

    # Calibrate
    img = cv.imread("calibration/img_01.png")
    imgs = [img]
    M = cal.calibrate_camera(imgs)

    # Execute
    img_test = cv.imread("test/img_00.png")
    points_3d_estimated = execute.get_points_from_image(img_test, M)
    scene = execute.get_scene_details(points_3d_estimated)
    execute.plot_scene(scene)
    instructions = execute.get_instructions(points_3d_estimated, scene)
    print(instructions)

if __name__ == "__main__":
    main()