import cv2
import numpy as np
import matplotlib.pyplot as plt


def detect_checker_board_corners(image):

    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    ret, corners = cv2.findChessboardCorners(gray, (7, 7), None)

    # if chessboard corners are detected
    if ret == True:
    
        # Draw and display the corners
        img = cv2.drawChessboardCorners(image, (7,7), corners, ret)
        # cv2.imshow('Chessboard',img)
        plt.imshow(img)
        plt.show()
        
        return corners
    return None

def segment_image_with_corners(image, corners):
    pass