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


def get_smooth_grayscale_image(img, threshold=52):
    
    for threshold in range(20, 70):
        gray = img.copy()
        gray[gray <= threshold] = 0
        gray[gray > threshold] = 255
        kernel = np.ones((3, 3), np.uint8)
        gray = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel, iterations=1)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        gray[gray < 255] = 0

        if abs(np.sum(gray == 255) / (img.shape[0] * img.shape[1]) - 0.90) < 0.005:
            print(f"found good threshold: {threshold}")
            return gray
    
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


def segment_chess_pieces(img, top, bottom, img_size, padding, segment_size=75):

    diff = []
    top = sorted(list(np.array(top)[:, 0]))
    for i in range(len(top)-1):
        diff.append(top[i+1]-top[i])
    mean_diff = np.mean(diff)

    diff = []
    bottom = sorted(list(np.array(bottom)[:, 0]))
    for i in range(len(bottom)-1):
        diff.append(bottom[i+1]-bottom[i])
    mean_diff += np.mean(diff)
    mean_diff /= 2

    centers = np.linspace(padding + mean_diff / 2, img_size - mean_diff / 2, 8)

    # fig, axes = plt.subplots(len(centers), len(centers), figsize=(10, 10))

    output = np.zeros(
        (len(centers), len(centers), 2*segment_size, 2*segment_size))

    for i, c_1 in enumerate(centers):
        for j, c_2 in enumerate(centers):
            sub_image = img[int(c_1 - segment_size):int(c_1 + segment_size),
                            int(c_2 - segment_size):int(c_2 + segment_size)]
            output[i, j] = sub_image
    #         axes[i, j].imshow(sub_image, cmap="gray")
    #         axes[i, j].set_axis_off()

    # plt.tight_layout()
    # plt.show()

    return output
