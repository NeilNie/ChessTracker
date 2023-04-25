from detect_board_v2 import *
from utils import find_bounding_box, get_smooth_grayscale_image
import numpy as np
import PIL.Image
import matplotlib.pyplot as plt
import cv2  # For Sobel etc


file_path = "./imgs/chess-9.jpg"
img_orig = PIL.Image.open(file_path)
img_width, img_height = img_orig.size
print("Image size %dx%d" % (img_width, img_height))

img = np.array(img_orig.convert("L"))  # grayscale uint8 numpy array

grad_phase_masked, grad_mag, sobelx, sobely = run_sobel_filters(img)
# fig = plt.figure(figsize=(10, 10))
edges = cv2.Canny(np.uint8(img), 50, 150, apertureSize=3)

h2, thetas, rhos = informedHough(
    edges,
    grad_phase_masked,
    grad_mag,
    theta_bin_size=2 * 360,
    rho_bin_size=2 * 360,
    inform_range=8,
)

input_h = h2.copy()
# Generate peak image
is_peak = getLocalMaxArray(input_h, winsize=11)
local_max_img = input_h.copy()
local_max_img[~is_peak] = 0  # Set peaks with intensity of peak

# fig = plt.figure(figsi ze=(5,5))
# plt.imshow(local_max_img.T > 0, interpolation='none', cmap=cm.inferno);

# local_max_img[local_max_img<100000] = 0
peaks = np.argwhere(local_max_img)

peak_mags = local_max_img[peaks[:, 0], peaks[:, 1]]
peak_order = np.argsort(peak_mags)[::-1]  # Strongest to weakest

# Sort peaks by strength
peaks = peaks[peak_order, :]
peak_mags = peak_mags[peak_order]

# fig = plt.figure(figsize=(10, 10))

# Only want peaks that are within half a standard deviation of the mean
threshold_good_peak = peak_mags.mean() + peak_mags.std() / 2
n_good_peaks = peaks.shape[0] - np.searchsorted(peak_mags[::-1], threshold_good_peak)

n_peaks = min(n_good_peaks, 100)
print(
    "Found",
    peaks.shape[0],
    "peaks,",
    n_good_peaks,
    "strong peaks, keeping only the first",
    n_peaks,
)

# plt.imshow(input_h.T, interpolation="none")
# plt.plot(peaks[:n_peaks, 0], peaks[:n_peaks, 1], "xr")
# for idx, [px, py] in enumerate(peaks[:n_peaks, :]):
#     plt.text(px, py, "%s" % idx, color="white", size=8)
# plt.axis("tight")
# plt.title("Hough Peaks")
# plt.show()

# ===========

lines = getHoughLines(peaks[:n_peaks], thetas, rhos, img.shape)

fig = plt.figure(figsize=(10, 10))
plt.imshow(img_orig)
plt.axis([0, img.shape[1], img.shape[0], 0])
for i, [x1, y1, x2, y2] in enumerate(lines):
    # Make first 20 lines strongest
    alpha_ = 1.0 if i < min(peaks.shape[0], 25) else 0.3
    plt.plot([x1, x2], [y1, y2], "r-", alpha=alpha_, lw=1)
plt.show()

# =========== a_segment_first_7, b_segment_first_7

vertical, horizontal = find_interior_lines(
    lines, n_peaks, peak_mags, (img_height, img_width), sobelx, sobely
)
intersections = find_intersections(vertical, horizontal, (img_height, img_width))

gray = get_smooth_grayscale_image(img)
line_array = find_points_on_boarder_line(intersections, gray)

import pdb; pdb.set_trace()
top_left = line_array[0][np.argmin(line_array[0][:, 1])]
bottom_left = line_array[0][np.argmax(line_array[0][:, 1])]
top_right = line_array[1][np.argmin(line_array[1][:, 1])]
bottom_right = line_array[1][np.argmax(line_array[1][:, 1])]
corners = [top_left, bottom_left, top_right, bottom_right]


fig = plt.figure(figsize=(10, 10))
plt.imshow(img, cmap="gray")
plt.axis([0, gray.shape[1], gray.shape[0], 0])

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
    plt.scatter(s[0], s[1], s=50, marker="x")

plt.show()
