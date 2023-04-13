import cv2
import numpy as np
import matplotlib.pyplot as plt


def detect_checker_board_corners(image):

    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    gray_corners = cv2.goodFeaturesToTrack(gray, 100, 0.5, 100)
    corners_array = np.int0(gray_corners)

    print(corners_array)
    # Display the corners found int he image

    for i in corners_array:
        x, y = i.ravel()
        cv2.circle(image, (x, y), 10, [0, 255, 255], -1)
    
    plt.imshow(image)
    plt.show()        

    return corners_array


def segment_image_with_corners(image, corners):
    pass