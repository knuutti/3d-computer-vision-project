import calibration as cal
import execute
import cv2 as cv
import numpy as np

def main():

    # Calibrate
    img = cv.imread("calibration/img_01.png")
    imgs = [img]
    M = cal.calibrate_camera(imgs, mode="auto")

    # Execute
    img_test = cv.imread("test/img_01.png")
    instructions = execute.move_block(["red", "blue", "green"], img_test, M)

    print(instructions)


if __name__ == "__main__":
    main()