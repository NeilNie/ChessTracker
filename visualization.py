import matplotlib.pyplot as plt


def visualize_lines(img_orig, lines):
    fig = plt.figure(figsize=(10, 10))
    plt.imshow(img_orig, cmap="gray")
    plt.axis([0, img_orig.shape[1], img_orig.shape[0], 0])
    for i, [x1, y1, x2, y2] in enumerate(lines):
        # Make first 20 lines strongest
        # alpha_ = 1.0 if i < min(peaks.shape[0], 25) else 0.3
        plt.plot([x1, x2], [y1, y2], "r-", alpha=1.0, lw=1)
    plt.show()


def visualize_hough():
    plt.imshow(input_h, interpolation="none")
    plt.plot(peaks[:n_peaks, 0], peaks[:n_peaks, 1], "xr")
    for idx, [px, py] in enumerate(peaks[:n_peaks, :]):
        plt.text(px, py, "%s" % idx, color="white", size=8)
    plt.axis("tight")
    plt.title("Hough Peaks")
    plt.show()
