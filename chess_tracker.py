import cv2
from enum import IntEnum
from tensorflow import keras
from detect_board_v2 import *
from chessboard import ChessBoard, ChessPiece
from utils import get_smooth_grayscale_image, distort_chess_board, segment_chess_pieces
from model.convnet import build_piece_classification_model
from hand_detector.hand_detector import HandDetector


class PieceType(IntEnum):
    BLACK = 1
    WHITE = 3
    EMPTY = 2


class ChessTracker():
    
    def __init__(self) -> None:
        # self.piece_model = build_piece_classification_model(2)
        self.piece_model = keras.models.load_model('empty_detector')
        self.type_model = keras.models.load_model('type_detector')
        self.hand_detector = HandDetector("./hand-detector-model")
        self.board = ChessBoard()

        self.corners = None
        self.img_size = 1000
        self.padding = 100

    def initialize_chess_board(self, img_orig):

        img_height, img_width, _ = img_orig.shape
        print("Image size %dx%d" % (img_width, img_height))

        img = np.array(cv2.cvtColor(img_orig, cv2.COLOR_RGB2GRAY))  # grayscale uint8 numpy array

        intersections, (vertical, horizontal) = find_chess_board_points(img)
        
        # for sec in intersections:
        #     max_y, min_y = np.amax(np.array(sec)[:, 1]), np.amin(np.array(sec)[:, 1])
        #     max_x, min_x = np.amax(np.array(sec)[:, 0]), np.amin(np.array(sec)[:, 0])

        #     line = np.sum(img[int(min_y-10): int(max_y+10), int(min_x): int(max_x)], axis=0)
            
        #     fft_result = np.abs(np.fft.rfft(line).real)
        #     strongest_freq = np.argmax(fft_result)
        #     # print(strongest_freq)
        #     plt.plot(line)

        # plt.show()

        gray = get_smooth_grayscale_image(img)

        plt.imshow(gray, cmap="gray")
        plt.show()

        self.line_array = find_points_on_boarder(intersections, gray)

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
        # colors = 'krgbykrcmykrgbykcmyk'
        # for i, sec in enumerate(self.line_array):
        #     for s in sec:
        #         plt.text(s[0] + 2, s[1] + 9, "%s" % i, color="white", size=8)
        #         plt.scatter(s[0], s[1], s=50, color=colors[i])
        # for s in corners:
        #     plt.scatter(s[0], s[1], s=100, marker="x")

        plt.show()

        if self.line_array is not None:
            top_left = self.line_array[0][np.argmin(self.line_array[0][:, 0])]
            top_right = self.line_array[0][np.argmax(self.line_array[0][:, 0])]
            bottom_left = self.line_array[1][np.argmin(self.line_array[1][:, 0])]
            bottom_right = self.line_array[1][np.argmax(self.line_array[1][:, 0])]
            self.corners = [top_left, bottom_left, bottom_right, top_right]
            print("found board corners")

    def infer_board_state(self, output):
        """infer the board state given 64 images of each 
        checker square. The tree states are: empty, white, 
        and black. 

        Args:
            output (ndarray): numpy array of the checkers

        Returns:
            ndarray: 8x8 numpy array representing the board
        """
        state = np.zeros((len(output), len(output)))
        for i in range(len(output)):
            for j in range(len(output)):
                input = cv2.resize(np.array(output[i, j]), (100, 100))
                model_output = self.piece_model(np.expand_dims(input, (0, -1)))
                type_output = self.type_model(np.expand_dims(input, (0, -1)))
                # print(type_output)
                # print(model_output, model_output > 0.5)
                state[i, j] = np.argmax(type_output) + 1
        return state

    def capture_loop(self):
        cam = cv2.VideoCapture(0)

        cv2.namedWindow("chess tracking (made by Neil & Jordan)")

        while True:
            ret, img = cam.read()
            img = img[200:-50, 400:-300]

            # ret = True
            # img = cv2.imread("./imgs/chess-11.jpg") # IMG_3330.jpeg
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            if not ret:
                print("failed to grab frame")
                break
            
            # check for corners
            if self.corners == None:
                self.initialize_chess_board(img)
                continue

            # detect hands
            has_hand = self.hand_detector.predict(img)
            if has_hand:
                print("warning, has hand")
                continue

            img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

            # distort the chessboard
            distorted, transform = distort_chess_board(
                np.float32(img), np.float32(self.corners), padding=100)

            # distort the points
            transformed = []
            for i, sec in enumerate(self.line_array):    
                transformed.append(cv2.perspectiveTransform(np.float32([sec]), transform)[0])

            # segment the chess pieces into 64 images
            output = segment_chess_pieces(
                distorted, transformed[0], transformed[1], 
                img_size=self.img_size, padding=self.padding)

            # run inferences
            state = self.infer_board_state(output)

            # update the board
            self.board.update_board(state)

            cv2.imshow("image", np.hstack((img, np.uint8(cv2.resize(distorted, (img.shape[0], img.shape[0]))))))

            k = cv2.waitKey(10)
            if k%256 == 27:
                # ESC pressed
                print("Escape hit, closing...")
                break

        cam.release()

        cv2.destroyAllWindows()
        
        
if __name__ == "__main__":
    tracker = ChessTracker()
    tracker.capture_loop()