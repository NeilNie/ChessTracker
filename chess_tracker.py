import cv2
from detect_board_v2 import *
from utils import get_smooth_grayscale_image, distort_chess_board
from chess_piece_classifier import segment_chess_pieces
from model.convnet import build_model


class ChessTracker():
    
    def __init__(self) -> None:
        self.model = build_model()
        self.corners = None
        self.img_size = 1000
        self.padding = 100

    def initialize_chess_board(self, img_orig):

        img_height, img_width = img_orig.shape
        print("Image size %dx%d" % (img_width, img_height))

        img = np.array(cv2.cvtColor(img_orig, cv2.COLOR_RGB2GRAY))  # grayscale uint8 numpy array

        intersections, (vertical, horizontal) = find_chess_board_points(img)
        gray = get_smooth_grayscale_image(img)

        plt.imshow(gray, cmap="gray")
        plt.show()

        line_array = find_points_on_boarder(intersections, gray)

        top_left = line_array[0][np.argmin(line_array[0][:, 0])]
        top_right = line_array[0][np.argmax(line_array[0][:, 0])]
        bottom_left = line_array[1][np.argmin(line_array[1][:, 0])]
        bottom_right = line_array[1][np.argmax(line_array[1][:, 0])]
        self.corners = [top_left, bottom_left, bottom_right, top_right]

    def capture_loop(self):
        cam = cv2.VideoCapture(0)

        cv2.namedWindow("chess tracking (made by Neil & Jordan)")

        while True:
            ret, img = cam.read()
            if not ret:
                print("failed to grab frame")
                break
            
            distorted, transform = distort_chess_board(
                np.float32(img), np.float32(self.corners), padding=100)

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
                        
            cv2.imshow("test", frame)
            k = cv2.waitKey(5)
            if k%256 == 27:
                # ESC pressed
                print("Escape hit, closing...")
                break
            # elif k%256 == 32:
            #     # SPACE pressed
            #     img_name = "opencv_frame_{}.png".format(img_counter)
            #     cv2.imwrite(img_name, frame)
            #     print("{} written!".format(img_name))
            #     img_counter += 1

        cam.release()

        cv2.destroyAllWindows()