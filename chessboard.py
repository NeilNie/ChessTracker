from enum import IntEnum
import numpy as np


class ChessPiece(IntEnum):
    KING = 0
    CASTLE = 1
    BISHOP = 2
    QUEEN = 3
    PAWN = 4
    KNIGHT = 5


class ChessBoard():

    def __init__(self) -> None:
        self.board = np.zeros((8, 9))
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

        print(self.board)
