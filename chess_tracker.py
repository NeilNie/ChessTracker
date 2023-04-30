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
        gray = get_smooth_grayscale_image(img)

        # plt.imshow(gray, cmap="gray")
        # plt.show()

        self.line_array = find_points_on_boarder(intersections, gray)

        if self.line_array is not None:
            top_left = self.line_array[0][np.argmin(self.line_array[0][:, 0])]
            top_right = self.line_array[0][np.argmax(self.line_array[0][:, 0])]
            bottom_left = self.line_array[1][np.argmin(self.line_array[1][:, 0])]
            bottom_right = self.line_array[1][np.argmax(self.line_array[1][:, 0])]
            self.corners = [top_left, bottom_left, bottom_right, top_right]
            print("found board corners")

    def capture_loop(self):
        cam = cv2.VideoCapture(0)

        cv2.namedWindow("chess tracking (made by Neil & Jordan)")

        while True:
            # ret, img = cam.read()            
            ret = True
            img = cv2.imread("./imgs/chess-11.jpg") # IMG_3330.jpeg
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
            state = np.zeros((len(output), len(output)))
            for i in range(len(output)):
                for j in range(len(output)):
                    input = cv2.resize(np.array(output[i, j]), (100, 100))
                    model_output = self.piece_model(np.expand_dims(input, (0, -1)))
                    type_output = self.type_model(np.expand_dims(input, (0, -1)))
                    # print(type_output)
                    # print(model_output, model_output > 0.5)
                    state[i, j] = np.argmax(type_output) + 1

            # update the board
            self.board.update_board(state)

            print(state)
            import pdb; pdb.set_trace()
            # plt.show()
            # plt.imshow(distorted, cmap="gray")
            # plt.show()
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