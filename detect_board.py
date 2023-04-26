import cv2
import numpy as np
import matplotlib.pyplot as plt


def detect_checker_board_ext_corners(image):

    # Repeated Closing operation to remove chess pieces.
    kernel = np.ones((3, 3), np.uint8)
    img = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel, iterations=10)
    gray = 255 - cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    gray[gray > 0] = 255

    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    canny = cv2.Canny(gray, 0, 200)
    canny = cv2.dilate(canny, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))

    # Blank canvas.
    con = np.zeros_like(img)
    # Finding contours for the detected edges.
    contours, hierarchy = cv2.findContours(canny, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    # Keeping only the largest detected contour.
    page = sorted(contours, key=cv2.contourArea, reverse=True)[:1]
    con = cv2.drawContours(con, page, -1, (0, 255, 255), 3)

    gray = cv2.cvtColor(con, cv2.COLOR_RGB2GRAY)
    plt.imshow(gray)
    plt.show()
    dest = cv2.cornerHarris(gray, 5, 5, 0.07)

    # Results are marked through the dilated corners
    dest = cv2.dilate(dest, None)
    print(dest)
    # Reverting back to the original image,
    # with optimal threshold value
    image[dest > 0.01 * dest.max()]=[0, 0, 255]

    plt.imshow(image)
    plt.show()

    # Blank canvas.
    con = np.zeros_like(img)
    # Loop over the contours.
    for c in page:
        # Approximate the contour.
        epsilon = 0.02 * cv2.arcLength(c, True)
        corners = cv2.approxPolyDP(c, epsilon, True)
        print(corners, len(corners))
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

    plt.imshow(image)
    plt.show()
    return np.float32(corners)



def distort_chess_board(image, points):
    h = 1000
    w = 1000

    # 4 Points on Original Image

    # 4 Corresponding Points of Desired Bird Eye View Image
    pt2 = np.float32([[0,0],[0,h],[w,h],[w,0]])

    matrix = cv2.getPerspectiveTransform(points, pt2)
    output = cv2.warpPerspective(image, matrix, (w,h))
    return output


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

def divide_checker_board_image(image):

    img2 = img

    height, width, channels = img.shape
    # Number of pieces Horizontally 
    CROP_W_SIZE  = 8
    # Number of pieces Vertically to each Horizontal  
    CROP_H_SIZE = 8
    
    images = []

    for ih in range(CROP_H_SIZE):
        for iw in range(CROP_W_SIZE):

            x = width/CROP_W_SIZE * iw 
            y = height/CROP_H_SIZE * ih
            h = (height / CROP_H_SIZE)
            w = (width / CROP_W_SIZE )
            print(x,y,h,w)
            img = img[y:y+h, x:x+w]

            images.append(img)
            img = img2

    return images
