from detect_board_v2 import *
from utils import find_bounding_box
import numpy as np
import PIL.Image
import matplotlib.pyplot as plt
import cv2  # For Sobel etc


file_path = "./imgs/chess-4.jpg"
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
    theta_bin_size=4 * 360,
    rho_bin_size=4 * 360,
    inform_range=10,
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
    alpha_ = 1.0 if i < min(peaks.shape[0], 22) else 0.3
    plt.plot([x1, x2], [y1, y2], "r-", alpha=alpha_, lw=1)
plt.show()

# ===========

a_segment_first_7, b_segment_first_7 = find_interior_lines(
    lines, n_peaks, peak_mags, sobelx, sobely
)
intersections = find_corners(lines, a_segment_first_7, b_segment_first_7)

fig = plt.figure(figsize=(10, 10))
plt.imshow(img_orig)
plt.axis([0, img.shape[1], img.shape[0], 0])


for k in a_segment_first_7:
    line = lines[k, :]
    x1, y1, x2, y2 = line
    plt.plot([x1, x2], [y1, y2], "g", lw=2)
    plt.text(x1 + 2, y1 + 9, "%s" % k, color="white", size=8)

for k in b_segment_first_7:
    line = lines[k, :]
    x1, y1, x2, y2 = line
    plt.plot([x1, x2], [y1, y2], "r", lw=2)
    plt.text(x1 + 2, y1 + 9, "%s" % k, color="white", size=8)

for sec in intersections:
    plt.scatter(sec[0], sec[1], s=50, color="yellow")

intersections = find_bounding_box(intersections)
for sec in intersections:
    plt.scatter(sec[0], sec[1], s=50, color="red")

plt.show()
