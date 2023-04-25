import cv2
import numpy as np
from detect_board import *

file_path = "./imgs/chess-9.jpg"

image = cv2.cvtColor(cv2.imread(file_path), cv2.COLOR_BGR2GRAY)

image[image <= 60] = 0
image[image > 60] = 255

image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

distorted = distort_chess_board(image)
plt.imshow(distorted)
plt.show()
# detect_checker_board_corners(image)