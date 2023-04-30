from enum import IntEnum
import numpy as np


class ChessPiece(IntEnum):
    KING = 1
    CASTLE = 2
    BISHOP = 3
    QUEEN = 4
    PAWN = 5
    KNIGHT = 6


class ChessBoard():

    def __init__(self) -> None:
        self.board = np.zeros((8, 8))
        self.board[:, 1] = ChessPiece.PAWN
        self.board[:, -2] = ChessPiece.PAWN

        self.board[0, 0] = ChessPiece.CASTLE
        self.board[-1, -1] = ChessPiece.CASTLE
        self.board[-1, 0] = ChessPiece.CASTLE
        self.board[0, -1] = ChessPiece.CASTLE

        self.board[1, 0] = ChessPiece.KNIGHT
        self.board[-2, 0] = ChessPiece.KNIGHT
        self.board[1, -1] = ChessPiece.KNIGHT
        self.board[-2, -1] = ChessPiece.KNIGHT
        
        self.board[2, 0] = ChessPiece.BISHOP
        self.board[-3, 0] = ChessPiece.BISHOP
        self.board[2, -1] = ChessPiece.BISHOP
        self.board[-3, -1] = ChessPiece.BISHOP
        
        self.board[3, 0] = ChessPiece.QUEEN
        self.board[4, 0] = ChessPiece.KING

        self.board[3, -1] = ChessPiece.KING
        self.board[4, -1] = ChessPiece.QUEEN

        print("KING = 1\nCASTLE = 2\nBISHOP = 3\nQUEEN = 4\nPAWN = 5\nKNIGHT = 6")
        print("Initial Board State")
        print(self.board)

    def update_board(self, array):
        binary = self.board > 0
        diff = np.sum(binary - array)
        if diff > 0:
            # print(array)
            where = np.argwhere(diff > 1)
            print(where)
