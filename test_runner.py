import cv2
import numpy as np
from detect_board import *

file_path = "./imgs/chess-1.jpg"

image = cv2.cvtColor(cv2.imread(file_path), cv2.COLOR_BGR2RGB)

detect_checker_board_ext_corners(image)
# detect_checker_board_corners(image)