import cv2
import numpy as np
import matplotlib.pyplot as plt


def detect_checker_board_ext_corners(image):

    # Repeated Closing operation to remove chess pieces.
    kernel = np.ones((3, 3), np.uint8)
    img = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel, iterations=13)
    gray = 255 - cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    gray[gray > 0] = 255

    # gray = cv2.GaussianBlur(gray, (11, 11), 0)
    canny = cv2.Canny(gray, 0, 200)
    # canny = cv2.dilate(canny, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))

    # Blank canvas.
    con = np.zeros_like(img)
    # Finding contours for the detected edges.
    contours, hierarchy = cv2.findContours(canny, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    # Keeping only the largest detected contour.
    page = sorted(contours, key=cv2.contourArea, reverse=True)[:5]
    con = cv2.drawContours(con, page, -1, (0, 255, 255), 3)

    # Blank canvas.
    con = np.zeros_like(img)
    # Loop over the contours.
    for c in page:
        # Approximate the contour.
        epsilon = 0.02 * cv2.arcLength(c, True)
        corners = cv2.approxPolyDP(c, epsilon, True)
        # If our approximated contour has four points
        if len(corners) == 4:
            break
    cv2.drawContours(con, c, -1, (0, 255, 255), 3)
    cv2.drawContours(con, corners, -1, (0, 255, 0), 10)
    # Sorting the corners and converting them to desired shape.
    corners = sorted(np.concatenate(corners).tolist())

    # Displaying the corners.
    for index, c in enumerate(corners):
        character = chr(65 + index)
        cv2.putText(con, character, tuple(c), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 1, cv2.LINE_AA)

    for i in corners:
        x, y = i
        cv2.circle(image, (x, y), 15, [0, 255, 255], -1)

    return corners
    # plt.imshow(image)
    # plt.show()


def distort_chess_board(image):
    h = 1000
    w = 1000

    # 4 Points on Original Image
    pt1 = detect_checker_board_ext_corners(image)

    # 4 Corresponding Points of Desired Bird Eye View Image
    pt2 = np.float32([[0,0],[0,h],[w,h],[w,0]])

    matrix = cv2.getPerspectiveTransform(pt1, pt2)
    output = cv2.warpPerspective(image, matrix, (w,h))


def detect_checker_board_corners(image):

    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    gray_corners = cv2.goodFeaturesToTrack(gray, 200, 0.1, 300)
    corners_array = np.int0(gray_corners)

    print(corners_array)
    # Display the corners found int he image

    for i in corners_array:
        x, y = i.ravel()
        cv2.circle(image, (x, y), 15, [0, 255, 255], -1)
    
    plt.imshow(image)
    plt.show()        

    return corners_array


def segment_image_with_corners(image, corners):
    pass