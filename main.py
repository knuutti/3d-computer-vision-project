from calibration import calibrate_camera
import image_processing as ip
import image_analysis as ia
import cv2 as cv
import matplotlib.pyplot as plt

def main():

    img_calibration = cv.imread("calibration/img_01.png")
    M = calibrate_camera(img_calibration, cube_calibration=True)
    img_test = cv.imread("test/img_01.png")
    img_hsv = cv.cvtColor(img_test, cv.COLOR_BGR2HSV)
    elements = ia.analyze_image(img_hsv)

    plt.figure()
    plt.imshow(img_test)
    for element in elements:
        plt.scatter(element[0][0], element[0][1], c='m', marker='x')
    plt.show()
    

if __name__ == "__main__":
    main()