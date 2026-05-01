# 3D Computer Vision: Practical assignment
Project work repository for the practical assignment for the course "3D Computer Vision". Goal of the project is to build a workflow for guiding a robot to move colored cubes to the desired target. The workflow includes camera calibration, object detection and an algorithm for guiding the robot. No deep learning methods were used for the solution.
## Usage
In `assignment.py`, functions `calibrate_camera()` and `move_block()` are defined. Define paths to the calibration image and the test image as well as the array of block you want to move, then use the functions mentioned to calibrate the camera (calculate projection matrix) and get instructions for the robot. Function `main()` displays an example workflow.

You can also import the functions to be used in other code files by writing `from assignment import calibrate_camera, move_block` to the beginning of a `.py` file.