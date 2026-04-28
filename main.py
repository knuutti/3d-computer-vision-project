from calibration import calibrate_camera
from execute import move_block
import cv2 as cv
import numpy as np

def main():
    
    # Calibrate
    img = cv.imread("calibration/calib3.png")
    imgs = [img]
    M = calibrate_camera(imgs, mode="auto")
    

    # Execute
    img_test = cv.imread("test/test10.png")
    blocks_to_move = ["blue", "red", "green"]
    
    instructions = move_block(blocks_to_move, img_test, M)

    

    print(instructions)


if __name__ == "__main__":
    main()