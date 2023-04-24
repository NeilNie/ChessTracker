import numpy as np
import PIL.Image
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import scipy.ndimage
from utils import line_intersection
import cv2  # For Sobel etc

np.set_printoptions(suppress=True)  # Better printing of arrays
plt.rcParams["image.cmap"] = "jet"  # Default colormap is jet


def find_corners(lines, lines_a, lines_b):
    
    intersections = []
    for k in lines_a:
        line = lines[k, :]
        x1, y1, x2, y2 = line
        
        for j in lines_b:
            line2 = lines[j, :]
            x_1, y_1, x_2, y_2 = line2
            
            intersect = line_intersection(
                [[x1, y1], [x2, y2]], [[x_1, y_1], [x_2, y_2]])
            intersections.append(intersect)
    return intersections


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


def informedHough(
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
    fft_result = np.abs(np.fft.rfft(normal_gradients).real)

    strongest_freq = np.argmax(fft_result)

    return strongest_freq, normal_gradients, fft_result, line_angle


def angleClose(a, b, angle_threshold=10 * np.pi / 180):
    d = np.abs(a - b)
    # Handle around the edge
    return d < angle_threshold or np.abs(2 * np.pi - d) < angle_threshold


def segmentAngles(freqs, angles, good_mask, angle_threshold=10 * np.pi / 180):
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
            ):
                segment_mask[j] = segment_idx
        # Iterate to next group
        segment_idx += 1
    return segment_mask, segment_idx  # segments and segment count


def chooseBestSegments(segments, num_segments, line_mags):
    segment_mags = np.zeros(num_segments)
    for i in range(1, num_segments):
        if np.sum(segments == i) < 4:
            # Need at least 4 lines in a segment
            segment_mags[i] = 0
        else:
            # Get average line gradient magnitude for that segment
            segment_mags[i] = np.sum(
                line_mags[: segments.size][segments == i]
            ) / np.sum(segments == i)

    order = np.argsort(segment_mags)[::-1]
    return order[:2]


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


def find_interior_lines(lines, n_peaks, peak_mags, sobelx, sobely):
    # k = 8
    freqs = np.zeros(n_peaks, dtype=int)
    angles = np.zeros(n_peaks)

    freq_threshold = 2

    # for k in [7, 8, 32]:
    # for k in [21]:
    good_mask = np.zeros(freqs.shape, dtype=bool)
    for k in range(n_peaks):
        line = lines[k, :]
        freq, line_grad, fft_result, line_angle = getLineGradients(line, sobelx, sobely)
        freqs[k] = freq
        angles[k] = line_angle
        # print(freq)
        if freq > freq_threshold:
            good_mask[k] = True

    segments, num_segments = segmentAngles(
        freqs, angles, good_mask, angle_threshold=20 * np.pi / 180
    )
    top_two_segments = chooseBestSegments(segments, num_segments, peak_mags)

    # Update good_mask to only include top two groups
    a_segment = segments == top_two_segments[0]
    b_segment = segments == top_two_segments[1]
    good_mask = a_segment | b_segment

    a_segment_first_7 = np.argwhere(a_segment)[:20].flatten()
    b_segment_first_7 = np.argwhere(b_segment)[:20].flatten()
    return a_segment_first_7, b_segment_first_7
