import calibration as cal
import execute2 as execute
import cv2 as cv
import numpy as np

def main():

    # Calibrate
    img = cv.imread("calibration/true_test_1.png")
    imgs = [img]
    M = cal.calibrate_camera(imgs, mode="auto")


    # Execute
    img_test = cv.imread("test/test_2.png")

    scene = execute.get_scene_details(execute.get_points_from_image(img_test, M))
    execute.plot_scene(scene)

    instructions = execute.move_block(["red", "blue", "green"], img_test, M)

    print(instructions)


if __name__ == "__main__":
    main()