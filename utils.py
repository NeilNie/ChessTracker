import numpy as np
import cv2


def line_intersection(line1, line2):
    xdiff = (line1[0][0] - line1[1][0], line2[0][0] - line2[1][0])
    ydiff = (line1[0][1] - line1[1][1], line2[0][1] - line2[1][1])

    def det(a, b):
        return a[0] * b[1] - a[1] * b[0]

    div = det(xdiff, ydiff)
    if div == 0:
        return -1, -1
        # raise Exception('lines do not intersect')

    d = (det(*line1), det(*line2))
    x = det(d, xdiff) / div
    y = det(d, ydiff) / div
    return x, y


def find_bounding_box(points):

    points = np.array(points)
    uleft, lright = points[np.argmin(
        points[:, 0])], points[np.argmax(points[:, 0])]

    # points = np.abs(uleft[1] - points[:, 1]
    # print([uleft, lright lleft, uright])
    return [uleft, lright]  # , lleft, uright


def get_smooth_grayscale_image(img):
    gray = img.copy()
    gray[gray <= 60] = 0
    gray[gray > 60] = 255
    kernel = np.ones((2, 2), np.uint8)
    gray = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel, iterations=1)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    gray[gray < 255] = 0
    return gray


def distort_chess_board(image, points, padding=50):
    h = 1000 + padding
    w = 1000 + padding

    # 4 Points on Original Image

    # 4 Corresponding Points of Desired Bird Eye View Image
    pt2 = np.float32([[padding, padding], [padding, h-padding],
                      [w-padding, h-padding], [w-padding, padding]])

    matrix = cv2.getPerspectiveTransform(points, pt2)
    output = cv2.warpPerspective(image, matrix, (w, h))
    return output, matrix