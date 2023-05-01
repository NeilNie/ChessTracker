from enum import IntEnum
import numpy as np
import time


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
        
        self.prev_time = None

    def update_board(self, array):
        
        if self.prev_time is None:
            self.prev_time = time.time()
            return
        else:
            if time.time() - self.prev_time < 3:
                return

        self.prev_time = time.time()
        
        binary = (self.board > 0).astype(np.int8)
        binary_current = (np.abs(array - 2) > 0).astype(np.int8)
        diff = np.sum(np.abs(binary - binary_current))

        print(array)

        if diff == 2:
            
            where = np.argwhere(np.abs(binary - binary_current) == 1)

            if array[where[0][0], where[0][1]] == 2:
                temp = self.board[where[1][0], where[1][1]]
                self.board[where[0][0], where[0][1]] = temp
                self.board[where[1][0], where[1][1]] = 0
            elif array[where[1][0], where[1][1]] == 2:
                temp = self.board[where[0][0], where[0][1]]
                self.board[where[1][0], where[1][1]] = temp
                self.board[where[0][0], where[0][1]] = 0
                
            import pdb; pdb.set_trace()

            print(len(where))
            print(where)
