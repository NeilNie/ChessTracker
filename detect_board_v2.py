import numpy as np
import PIL.Image
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import scipy.ndimage
from utils import line_intersection
import cv2  # For Sobel etc

np.set_printoptions(suppress=True)  # Better printing of arrays
plt.rcParams["image.cmap"] = "jet"  # Default colormap is jet


def find_chess_board_points(img):

    img_height, img_width = img.shape
    grad_phase_masked, grad_mag, sobelx, sobely = run_sobel_filters(img)
    edges = cv2.Canny(np.uint8(img), 50, 150, apertureSize=3)

    h2, thetas, rhos = informed_hough(
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


    # local_max_img[local_max_img<100000] = 0
    peaks = np.argwhere(local_max_img)

    peak_mags = local_max_img[peaks[:, 0], peaks[:, 1]]
    peak_order = np.argsort(peak_mags)[::-1]  # Strongest to weakest

    # Sort peaks by strength
    peaks = peaks[peak_order, :]
    peak_mags = peak_mags[peak_order]

    # Only want peaks that are within half a standard deviation of the mean
    threshold_good_peak = peak_mags.mean() + peak_mags.std() / 2
    n_good_peaks = peaks.shape[0] - np.searchsorted(peak_mags[::-1], threshold_good_peak)

    n_peaks = min(n_good_peaks, 100)

    # =========== find all lines ======== #
    lines = getHoughLines(peaks[:n_peaks], thetas, rhos, img.shape)

    # =========== good lines lines ======== #
    vertical, horizontal = find_good_lines(
        lines, n_peaks, peak_mags, (img_height, img_width), sobelx, sobely
    )
    intersections = find_line_intersections(horizontal, vertical)

    return intersections, (vertical, horizontal)


def find_line_intersections(vertical, horizontal):

    intersections = []
    for k in vertical:
        x1, y1, x2, y2 = k
        ls = []
        for j in horizontal:
            x_1, y_1, x_2, y_2 = j
            intersect = line_intersection(
                [[x1, y1], [x2, y2]], [[x_1, y_1], [x_2, y_2]])
            # if size[0] > intersect[0] >= 0 and size[1] > intersect[1] >= 0:
            ls.append(intersect)

        if len(ls) > 0:
            intersections.append(np.array(ls))
    return np.array(intersections)


def find_points_on_boarder(intersections, boarders):

    points = np.argwhere(boarders == 0)
    tmp = points[:, 0].copy()
    points[:, 0] = points[:, 1].copy()
    points[:, 1] = tmp

    line_array = []

    for intersection in intersections:
        valid = []
        for point in intersection:
            # distance to black line must be less than 10 pixels
            if np.amin(np.linalg.norm(np.array(point) - points, axis=1)) < 5:
                if len(valid) > 1:
                    if np.amin(np.linalg.norm(np.array(valid) - np.array(point), axis=1)) > 1:
                        valid.append(point)
                else:
                    valid.append(point)
        # print(len(valid))
        if len(valid) >= 9:
            line_array.append(valid)
    
    # must only be two groups
    # print(len(line_array))
    assert(len(line_array) == 2)
    if np.mean(np.array(line_array[0])[:, 1]) < np.mean(np.array(line_array[1])[:, 1]):
        return np.array(line_array)
    else:
        return np.array([np.array(line_array[1]), np.array(line_array[0])])


def run_sobel_filters(img):
    sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=5)
    sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=5)
    grad_mag = np.sqrt(sobelx**2 + sobely**2)
    grad_phase = np.arctan2(sobely, sobelx)  # from -pi to pi

    # Remove phase from where gradient magnitude is less than the mean * constant
    grad_phase_masked = grad_phase.copy()
    gradient_mask_threshold = 2 * np.mean(grad_mag.flatten())
    grad_phase_masked[grad_mag < gradient_mask_threshold] = np.nan

    return grad_phase_masked, grad_mag, sobelx, sobely


def informed_hough(
    bin_img,
    gradient_phase_img,
    gradient_magnitude_img,
    theta_bin_size=100,
    rho_bin_size=100,
    inform_range=5,
):
    """Return informed hough space of input binary image"""
    thetas = np.linspace(0, np.pi, theta_bin_size)
    rho_diagonal = np.sqrt(np.sum(np.array(bin_img.shape) ** 2))
    rhos = np.linspace(-rho_diagonal, rho_diagonal, rho_bin_size)  # length of diagonal
    hough_space = np.zeros([theta_bin_size, rho_bin_size])

    for i in range(bin_img.shape[0]):
        for j in range(bin_img.shape[1]):
            g = gradient_phase_img[i, j]
            if bin_img[i, j] and not np.isnan(g):
                if g < 0:
                    g += np.pi  # + 180 degrees
                # Get informed theta range
                # Get bin index in thetas
                theta_idx = np.searchsorted(thetas, g)
                theta_left = max(0, theta_idx - inform_range)
                theta_right = min(len(thetas) - 1, theta_idx + inform_range)

                # Over the informed theta sweep range
                for t, theta in enumerate(thetas[theta_left:theta_right]):
                    # TODO continue to use %prun and see if building a cos/sin table is valuable
                    rho = j * np.cos(theta) + i * np.sin(theta)
                    # Get bin index for rhos
                    rho_idx = np.searchsorted(rhos, rho)
                    # Add gradient magnitude
                    hough_space[
                        theta_idx - inform_range + t, rho_idx
                    ] += gradient_magnitude_img[i, j]
    return (hough_space, thetas, rhos)


def generic_filter_function(neighborhood, center_idx):
    # Definitely not if no hits to hough at this point
    center_val = neighborhood[center_idx]
    if center_val == 0:
        return False
    neighborhood[center_idx] = 0  # Don't compare to self
    return not np.any(neighborhood >= center_val)


def getLocalMaxArray(h, winsize=7):
    """Returns matrix with only the peaks of a given input matrix"""
    # winsize needs to be odd to choose center_idx correctly
    center_idx = (winsize + 1) * (winsize >> 1)
    return scipy.ndimage.generic_filter(
        h,
        generic_filter_function,
        size=winsize,
        mode="wrap",
        extra_arguments=(center_idx,),
    ).astype(bool)


def getLineGradients(line, gradient_x, gradient_y, sampling_rate=1):
    """Calculate normal gradient values along line given x/y gradients and a line segment."""

    # 1 - Get gradient values
    line = np.array(line)
    ptA = line[:2]
    ptB = line[2:]

    # unit vector in direction of line
    line_length = np.linalg.norm(ptB - ptA)
    line_direction = (ptB - ptA) / line_length

    # Convert to normal
    # -y, x for normal in one direction
    line_normal = np.array([-line_direction[1], line_direction[0]])
    line_angle = np.math.atan2(line_normal[1], line_normal[0])

    # Get points along line, choosing number of points giving a sampling rate in pixels per points (1-1 is good)
    num_pts_on_line = int(np.ceil(np.sqrt(np.sum((ptB - ptA) ** 2)) / sampling_rate))
    guessx = np.linspace(ptA[1], ptB[1], num_pts_on_line)
    guessy = np.linspace(ptA[0], ptB[0], num_pts_on_line)

    line_indices = np.floor(np.vstack((guessx, guessy)).T).astype(int)
    gradients = np.vstack(
        [
            gradient_x[line_indices[:, 0], line_indices[:, 1]],
            gradient_y[line_indices[:, 0], line_indices[:, 1]],
        ]
    )

    # Magnitude of gradient along normal
    normal_gradients = line_normal.dot(gradients)

    # Calculate fft, since sampling rate is static, we can just use indices as a comparison method
    # fft_result = np.abs(np.fft.rfft(normal_gradients).real)

    # strongest_freq = np.argmax(fft_result)

    return 0, normal_gradients, 0, line_angle


def angleClose(a, b, angle_threshold=10 * np.pi / 180):
    d = np.abs(a - b)
    # Handle around the edge
    return d < angle_threshold or np.abs(2 * np.pi - d) < angle_threshold


def angle_gap(x, y):
    return np.arctan2(np.sin(x-y), np.cos(x-y))

def intersect_in_bounds(line1, line2, img_size):
    x, y = line_intersection(line1, line2)
    if 0 <= x < img_size[1] and 0 <= y < img_size[0]:
        return True
    return False


def segmentAngles(lines, angles, good_mask, img_size, 
                  intersection=False, angle_threshold=10 * np.pi / 180):

    # Partition lines based on similar angles int segments/groups
    segment_mask = np.zeros(angles.shape, dtype=int)

    segment_idx = 1
    for i in range(angles.size):
        # Skip if not a good line or line already grouped
        if not good_mask[i] or segment_mask[i] != 0:
            continue

        # Create new group
        segment_mask[i] = segment_idx
        for j in range(i + 1, angles.size):
            # If good line, not yet grouped, and is close in angle, add to segment group
            if (
                good_mask[j]
                and segment_mask[j] == 0
                and angleClose(angles[i], angles[j], angle_threshold)
                and (intersection and intersect_in_bounds(
                    lines[i].reshape(2, 2), lines[j].reshape(2, 2), img_size))
            ):
                segment_mask[j] = segment_idx
        # Iterate to next group
        segment_idx += 1
    return segment_mask, segment_idx  # segments and segment count


def group_lines(lines, angles, img_size, line_mags, angle_threshold=10 * np.pi / 180):

    # Partition lines based on similar angles int segments/groups
    groups = [[i] for i in range(len(lines))]

    for i in range(len(angles)):

        # Create new group
        for j in range(len(angles)):
            # If good line, not yet grouped, and is close in angle, add to segment group
            if (
                angleClose(angles[i], angles[j], angle_threshold)
                and intersect_in_bounds(
                    lines[i].reshape(2, 2), lines[j].reshape(2, 2), img_size)
            ):
                groups[i].append(j)
        groups[i] = tuple(groups[i])
        # Iterate to next group

    groups = set(groups)
    good_lines = []
    good_angles = []
    for i, g in enumerate(groups):
        sum = np.zeros((2, 2))
        for a in g:
            sum += np.array(lines[a]).reshape(2, 2)
        sum /= len(g)
        good_lines.append(sum.flatten())
        good_angles.append(np.average(np.array(angles)[list(g)]))

    # separate vertical and horizontal
    horizontal = []
    vertical = []
    for i, angle in enumerate(good_angles):
        if (good_lines[i] < 0).any():
            continue
        if abs(angle_gap(angle, np.pi/2)) < 20 * np.pi / 180:
            horizontal.append(good_lines[i])
        elif abs(angle_gap(angle, np.pi)) < 20 * np.pi / 180 or abs(angle_gap(angle, 0)) < 20 * np.pi / 180:
            vertical.append(good_lines[i])

    return vertical, horizontal  # segments and segment count


def chooseBestSegments(segments, num_segments, line_mags, top=2):
    print(segments)
    segment_mags = np.zeros(num_segments)
    for i in range(1, num_segments):
        
        segment_mags[i] = np.sum(
                line_mags[: segments.size][segments == i]
            ) / np.sum(segments == i)
        
        # if np.sum(segments == i) < 2:
        #     # Need at least 4 lines in a segment
        #     segment_mags[i] = 0
        # else:
        #     # Get average line gradient magnitude for that segment

    # import pdb; pdb.set_trace()
    # print(segment_mags)
    order = np.argsort(segment_mags)[::-1]
    return order[:top]


def getHoughLines(peaks, thetas, rhos, img_shape):
    # lines segments within image bounds x1 y1 x2 y2
    lines = np.zeros([peaks.shape[0], 4])

    for i, [theta_, rho_] in enumerate(peaks):
        theta = thetas[theta_]
        rho = rhos[rho_]
        c = np.cos(theta)
        s = np.sin(theta)

        img_x_max = img_shape[1] - 1
        img_y_max = img_shape[0] - 1

        if np.abs(c) < np.abs(s):
            # angle is closer to 0 or 180 degrees, horizontal line so use x limits
            #             print("H")
            x1, x2 = 0, img_x_max
            y1, y2 = (rho - x1 * c) / s, (rho - x2 * c) / s
        else:
            # angle closer to 90 degrees, vertical line so use y limits
            #             print("V")
            y1, y2 = 0, img_y_max
            x1, x2 = (rho - y1 * s) / c, (rho - y2 * s) / c

        # Get line ends within image bounds
        # TODO : Fails on very close to vertical/horizontal lines due to divide by ~zero
        if np.abs(s) > 0.01 and np.abs(c) > 0.01:
            if y1 < 0:
                x1 = (rho - 0 * s) / c
                y1 = (rho - x1 * c) / s
            elif y1 > img_y_max:
                x1 = (rho - img_y_max * s) / c
                y1 = (rho - x1 * c) / s
            if y2 < 0:
                x2 = (rho - 0 * s) / c
                y2 = (rho - x2 * c) / s
            elif y2 > img_y_max:
                x2 = (rho - img_y_max * s) / c
                y2 = (rho - x2 * c) / s

            if x1 < 0:
                y1 = (rho - 0 * c) / s
                x1 = (rho - y1 * s) / c
            elif x1 > img_x_max:
                y1 = (rho - img_x_max * c) / s
                x1 = (rho - y1 * s) / c
            if x2 < 0:
                y2 = (rho - 0 * c) / s
                x2 = (rho - y2 * s) / c
            elif x2 > img_x_max:
                y2 = (rho - img_x_max * c) / s
                x2 = (rho - y2 * s) / c

        lines[i, :] = [x1, y1, x2, y2]

    return lines


def find_good_lines(lines, n_peaks, peak_mags, img_size, sobelx, sobely):
    """Find good lines given a list of possible lines
    from hough

    Args:
        lines (_type_): _description_
        n_peaks (_type_): _description_
        peak_mags (_type_): _description_
        img_size (_type_): _description_
        sobelx (_type_): _description_
        sobely (_type_): _description_

    Returns:
        tuple: vertical, horizontal lines as lists
    """

    angles = np.zeros(n_peaks)

    freq_threshold = 0

    good_mask = np.zeros(angles.shape, dtype=bool)
    for k in range(n_peaks):
        line = lines[k, :]
        freq, line_grad, fft_result, line_angle = getLineGradients(line, sobelx, sobely)
        angles[k] = line_angle
        if freq >= freq_threshold:
            good_mask[k] = True
    
    vertical, horizontal = group_lines(
        lines, angles=angles, img_size=img_size, line_mags=peak_mags)

    return vertical, horizontal
