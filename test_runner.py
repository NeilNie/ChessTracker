import cv2
import numpy as np
from detect_board import *

file_path = "./imgs/chess-3.jpg"

image = cv2.cvtColor(cv2.imread(file_path), cv2.COLOR_BGR2RGB)

detect_checker_board_corners(image)