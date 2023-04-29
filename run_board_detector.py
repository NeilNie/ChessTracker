from detect_board_v2 import *
from utils import get_smooth_grayscale_image, distort_chess_board
import numpy as np
import PIL.Image
import matplotlib.pyplot as plt
import cv2  # For Sobel etc
from chess_piece_classifier import *
import os

img_size = 1000
padding = 100

# four boards
# [81, 82, 85, 86, 87, chess-11, chess-10]
file_path = "./imgs/IMG_3336.jpeg" # IMG_3083 up to 101
img_orig = PIL.Image.open(file_path)
img_width, img_height = img_orig.size
print("Image size %dx%d" % (img_width, img_height))

img = np.array(img_orig.convert("L"))  # grayscale uint8 numpy array

intersections, (vertical, horizontal) = find_chess_board_points(img)
gray = get_smooth_grayscale_image(img)

plt.imshow(gray, cmap="gray")
plt.show()

line_array = find_points_on_boarder(intersections, gray)

top_left = line_array[0][np.argmin(line_array[0][:, 0])]
top_right = line_array[0][np.argmax(line_array[0][:, 0])]
bottom_left = line_array[1][np.argmin(line_array[1][:, 0])]
bottom_right = line_array[1][np.argmax(line_array[1][:, 0])]
corners = [top_left, bottom_left, bottom_right, top_right]

# corners = []

plt.figure(figsize=(10, 10))
plt.imshow(img, cmap="gray")


for i, line in enumerate(vertical):
    x1, y1, x2, y2 = line
    plt.plot([x1, x2], [y1, y2], "g", lw=2)
    plt.text(x1 + 2, y1 + 9, "%s" % i, color="white", size=8)

for i, line in enumerate(horizontal):
    x1, y1, x2, y2 = line
    plt.plot([x1, x2], [y1, y2], "r", lw=2)
    plt.text(x1 + 2, y1 + 9, "%s" % i, color="white", size=8)


# intersections = find_bounding_box(intersections)
colors = 'krgbykrcmykrgbykcmyk'
for i, sec in enumerate(line_array):
    for s in sec:
        plt.text(s[0] + 2, s[1] + 9, "%s" % i, color="white", size=8)
        plt.scatter(s[0], s[1], s=50, color=colors[i])
for s in corners:
    plt.scatter(s[0], s[1], s=100, marker="x")

plt.show()

distorted, transform = distort_chess_board(np.float32(img), np.float32(corners), padding=100)

# distort the points
transformed = []
for i, sec in enumerate(line_array):    
    transformed.append(cv2.perspectiveTransform(np.float32([sec]), transform)[0])

fig = plt.figure(figsize=(10, 10))
plt.imshow(distorted, cmap="gray")
plt.show()

for i, sec in enumerate(transformed):
    for s in sec:
        plt.text(s[0] + 2, s[1] + 9, "%s" % i, color="white", size=8)
        plt.scatter(s[0], s[1], s=50, color=colors[i])


output = segment_chess_pieces(distorted, transformed[0], transformed[1], img_size=img_size, padding=padding)
print(output.shape)
count = 1200
for i in range(output.shape[0]):
    for j in range(output.shape[1]):
        img = output[i, j]
        count += 1
        cv2.imwrite(f"./empty/{count}.png", img)


# knight = [output[1, 0, :, :], 
#           output[-2, 0, :, :], 
#           output[1, -1, :, :], 
#           output[-2, -1, :, :]]
# for img in knight:
#     count = len(os.listdir("./dataset/knight/"))
#     cv2.imwrite(f"./dataset/knight/{count}.png", img)

# bishop = [output[2, 0, :, :], 
#           output[-3, 0, :, :], 
#           output[2, -1, :, :], 
#           output[-3, -1, :, :]]
# for img in bishop:
#     count = len(os.listdir("./dataset/bishop/"))
#     cv2.imwrite(f"./dataset/bishop/{count}.png", img)

# castle = [output[0, 0, :, :], 
#           output[0, -1, :, :], 
#           output[-1, -1, :, :], 
#           output[-1, 0, :, :]]
# for img in castle:
#     count = len(os.listdir("./dataset/castle/"))
#     cv2.imwrite(f"./dataset/castle/{count}.png", img)


# for img in output[:, 1, :, :]:
#     count = len(os.listdir("./dataset/pawn/"))
#     cv2.imwrite(f"./dataset/pawn/{count}.png", img)

# for img in output[:, -2, :, :]:
#     count = len(os.listdir("./dataset/pawn/"))
#     cv2.imwrite(f"./dataset/pawn/{count}.png", img)
