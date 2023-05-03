from enum import IntEnum
import numpy as np
import time
from stockfish import Stockfish
from controller.controller import board_pos_to_chess
import matplotlib.pyplot as plt


class ChessPiece(IntEnum):
    EMPTY = 0
    W_KING = 1
    W_CASTLE = 2
    W_BISHOP = 3
    W_QUEEN = 4
    W_PAWN = 5
    W_KNIGHT = 6
    
    B_KING = 7
    B_CASTLE = 8
    B_BISHOP = 9
    B_QUEEN = 10
    B_PAWN = 11
    B_KNIGHT = 12

rows = "abcdefgh"
columns = "12345678"

piece_to_unicode = {
    Stockfish.Piece.WHITE_KING:     '\u2654',
    Stockfish.Piece.WHITE_QUEEN:    '\u2655',
    Stockfish.Piece.WHITE_ROOK:     '\u2656',
    Stockfish.Piece.WHITE_BISHOP:   '\u2657',
    Stockfish.Piece.WHITE_KNIGHT:   '\u2658',
    Stockfish.Piece.WHITE_PAWN:     '\u2659',
    Stockfish.Piece.BLACK_KING:     '\u265a',
    Stockfish.Piece.BLACK_QUEEN:    '\u265b',
    Stockfish.Piece.BLACK_ROOK:     '\u265c',
    Stockfish.Piece.BLACK_BISHOP:   '\u265d',
    Stockfish.Piece.BLACK_KNIGHT:   '\u265e',
    Stockfish.Piece.BLACK_PAWN:     '\u265f',
    None:                           ''
}

custom_piece_to_unicode = {
    ChessPiece.W_KING:     '\u2654',
    ChessPiece.W_QUEEN:    '\u2655',
    ChessPiece.W_CASTLE:   '\u2656',
    ChessPiece.W_BISHOP:   '\u2657',
    ChessPiece.W_KNIGHT:   '\u2658',
    ChessPiece.W_PAWN:     '\u2659',
    ChessPiece.B_KING:     '\u265a',
    ChessPiece.B_QUEEN:    '\u265b',
    ChessPiece.B_CASTLE:   '\u265c',
    ChessPiece.B_BISHOP:   '\u265d',
    ChessPiece.B_KNIGHT:   '\u265e',
    ChessPiece.B_PAWN:     '\u265f',
    ChessPiece.EMPTY:      ''
}


def chess_coordinate_to_rc(string):
    return ord(string[0]) - ord("a"), ord(string[-1]) - ord("1")


class ChessBoard():

    def __init__(self, path="/opt/homebrew/bin/stockfish") -> None:
        self.board = np.zeros((8, 8))
        self.board[:, 1] = ChessPiece.W_PAWN
        self.board[:, -2] = ChessPiece.B_PAWN

        self.board[0, 0] = ChessPiece.W_CASTLE
        self.board[-1, -1] = ChessPiece.B_CASTLE
        self.board[-1, 0] = ChessPiece.W_CASTLE
        self.board[0, -1] = ChessPiece.B_CASTLE

        self.board[1, 0] = ChessPiece.W_KNIGHT
        self.board[-2, 0] = ChessPiece.W_KNIGHT
        self.board[1, -1] = ChessPiece.B_KNIGHT
        self.board[-2, -1] = ChessPiece.B_KNIGHT
        
        self.board[2, 0] = ChessPiece.W_BISHOP
        self.board[-3, 0] = ChessPiece.W_BISHOP
        self.board[2, -1] = ChessPiece.B_BISHOP
        self.board[-3, -1] = ChessPiece.B_BISHOP
        
        self.board[3, 0] = ChessPiece.W_QUEEN
        self.board[4, 0] = ChessPiece.W_KING

        self.board[3, -1] = ChessPiece.B_KING
        self.board[4, -1] = ChessPiece.B_QUEEN

        print("KING = 1\nCASTLE = 2\nBISHOP = 3\nQUEEN = 4\nPAWN = 5\nKNIGHT = 6")
        print("Initial Board State")
        print(self.board)

        self.best_moves = []

        self.stockfish = Stockfish(path=path)

        self.curr_player = True # white
        self.white_clock = 300
        self.black_clock = 300
        self.prev_time = None

    def initialize_board_gui(self):
        print("initializing gui")
        self.fig, self.axs = plt.subplots(8, 8, figsize=(8, 8))
        plt.subplots_adjust(wspace=0, hspace=0)
        self.text_box = plt.figtext(0.05, 0.045, "[status]: nominal", 
                               fontsize=18, va="bottom", ha="left", color="b")
        self.block_text = plt.figtext(0.05, 0.035, "[clock]: white: -- | black: -- ", 
                               fontsize=18, va="top", ha="left", color="k")
        self.redraw_board(True)
        plt.ion()
        self.fig.suptitle("Welcome to the ChessTracker")
        plt.show()

    def redraw_board(self, valid):

        if not valid:
            self.text_box.set_text("[status]: invalid move")
            self.text_box.set_c("r")
        else:
            self.text_box.set_text("[status]: nominal")
            self.text_box.set_c("b")

        self.block_text.set_text(f"[clock]: white: {int(self.white_clock)} | black: {int(self.black_clock)}")

        for i in range(64):
            x = i % 8
            y = i // 8

            self.axs[y, x].clear()

            if (y, x) in self.best_moves:
                print("set face color green")
                # self.axs[y, x].set_facecolor("green")
            elif (i + y) % 2 == 0:
                self.axs[y, x].set_facecolor("white")
            else:
                self.axs[y, x].set_facecolor("grey")

            self.axs[y, x].xaxis.set_ticks([])
            self.axs[y, x].yaxis.set_ticks([])

            pos = chr(ord('a') + y) + str(x + 1)

            # piece = ChessPiece(int(self.board[y, x]))
            # self.axs[y, x].text(.1, .15, custom_piece_to_unicode[piece], fontsize=50)

            self.axs[y, x].text(.1, .15, piece_to_unicode[self.stockfish.get_what_is_on_square(
                pos)], fontsize=50)

    def update_board(self, array):

        if self.prev_time is None:
            self.prev_time = time.time()
            return None
        else:
            if self.curr_player:
                self.white_clock -= time.time() - self.prev_time
            else:
                self.black_clock -= time.time() - self.prev_time
            if time.time() - self.prev_time < 1:
                self.redraw_board(True)
                return None

        self.prev_time = time.time()
        
        binary = (self.board > 0).astype(np.int8)
        binary_current = (np.abs(array - 2) > 0).astype(np.int8)
        diff = np.sum(np.abs(binary - binary_current))

        print(array, diff)

        if diff == 2:

            where = np.argwhere(np.abs(binary - binary_current) == 1)

            print(where)

            if array[where[1][0], where[1][1]] == 2:
                move = board_pos_to_chess(*where[1]) + board_pos_to_chess(*where[0])
            elif array[where[0][0], where[0][1]] == 2:
                move = board_pos_to_chess(*where[0]) + board_pos_to_chess(*where[1])

            valid = self.stockfish.is_move_correct(move)
            if valid:
                
                # update the board
                if array[where[1][0], where[1][1]] == 2:
                    temp = self.board[where[1][0], where[1][1]]
                    self.board[where[0][0], where[0][1]] = temp
                    self.board[where[1][0], where[1][1]] = 0                
                elif array[where[0][0], where[0][1]] == 2:
                    temp = self.board[where[0][0], where[0][1]]
                    self.board[where[1][0], where[1][1]] = temp
                    self.board[where[0][0], where[0][1]] = 0
                self.curr_player = not self.curr_player
                
                self.stockfish.make_moves_from_current_position([move])
                move = self.stockfish.get_best_move()
                print(f"current move: {move}")
                self.best_moves = [chess_coordinate_to_rc(move[:2]), chess_coordinate_to_rc(move[2:])]
                # import pdb; pdb.set_trace()
            else:
                print("invalid move!")
            
            self.redraw_board(valid)
            return move

