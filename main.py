import calibration as cal
import execute2 as execute
import cv2 as cv

def main():

    # Calibrate
    img = cv.imread("calibration/calib3.png")
    imgs = [img]
    M = cal.calibrate_camera(imgs, mode="auto")

    # Execute
    img_test = cv.imread("test/test10.png")
    blocks_to_move = ["blue", "green", "red"]
    
    instructions = execute.move_block(blocks_to_move, img_test, M)

    print(instructions)


if __name__ == "__main__":
    main()