import numpy as np
import matplotlib.pyplot as plt


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


    fig, axes = plt.subplots(len(centers), len(centers), figsize=(10, 10))

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
