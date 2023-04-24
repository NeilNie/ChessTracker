import numpy as np


def line_intersection(line1, line2):
    xdiff = (line1[0][0] - line1[1][0], line2[0][0] - line2[1][0])
    ydiff = (line1[0][1] - line1[1][1], line2[0][1] - line2[1][1])

    def det(a, b):
        return a[0] * b[1] - a[1] * b[0]

    div = det(xdiff, ydiff)
    if div == 0:
        return 0, 0
        # raise Exception('lines do not intersect')

    d = (det(*line1), det(*line2))
    x = det(d, xdiff) / div
    y = det(d, ydiff) / div
    return x, y


def find_bounding_box(points):

    points = np.array(points)
    uleft, lright = points[np.argmin(points[:, 0])], points[np.argmax(points[:, 0])]

    # points = np.abs(uleft[1] - points[:, 1]
    # print([uleft, lright lleft, uright])
    return [uleft, lright] # , lleft, uright
