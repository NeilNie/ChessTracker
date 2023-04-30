import numpy as np
from stockfish import Stockfish

import numpy as np
from stockfish import Stockfish

def board_pos_to_chess(row, col):
    return chr(ord('a') + row) + str(col + 1)

class controller():
    def __init__(self, path):
        self.stockfish = Stockfish(path=path)

        self.current_board = np.zeros(shape=(8, 8))
        self.current_board[:, 0:2] = 1
        self.current_board[:, -2:] = 2

        print(self.current_board)


    def update_board(self, board):
        # returns -1 if error
        # returns 0 if no move was made
        # returns 1 if move was made
        differences = []
        
        for i in range(board.shape[0]):
            for j in range(board.shape[1]):
                if board[i, j] != self.current_board[i, j]:
                    differences.append((i, j))
        
        if len(differences) > 3:
            return -1, "Too many moved pieces"
        
        elif len(differences) == 0:
            return 0, "No Change"
        
        elif len(differences) < 2:
            return -1, "piece missing"
        
        elif len(differences) == 2: # one piece is moved
            # must be 1 new empty square
            
            if board[differences[0]] != 0 and board[differences[1]] != 0:
                return -1, "invalid or too many moves"
            
            # swap differences if first one is not empty (source square)
            if board[differences[0]] != 0:
                differences[0], differences[1] = differences[1], differences[0]
                
            move = board_pos_to_chess(*differences[0]) + board_pos_to_chess(*differences[1])
            
            print(move)
            
            if self.stockfish.is_move_correct(move):
                self.stockfish.make_moves_from_current_position([move])
                self.current_board = np.copy(board)
                
                return 1, move
            
            else:
                return -1, "invalid move from current position"
            
        
        elif len(differences) == 3: # check if en pessant
            # should now be 2 new empty squares
            # the source will be the color with a piece remaining
            
            empty_count = 0
            remaining = 0
            
            for i in range(len(differences)):
                if board[differences[i]] == 0:
                    empty_count += 1
                else:
                    # will only run once in correct cases
                    remaining = i
                    
            
            if empty_count != 2:
                return -1, "invalid or too many moves"
            
            source = None
            target = differences[remaining]
            
            
            for i in range(len(differences)):
                # found correct source
                if self.current_board[differences[i]] == board[differences[remaining]]:
                        source = differences[i]
                        
            if target is None:
                return -1, "invalid or too many moves"
                        
            move = board_pos_to_chess(*source) + board_pos_to_chess(*target)
            
            print(move)
            
            if self.stockfish.is_move_correct(move):
                self.stockfish.make_moves_from_current_position([move])
                self.current_board = np.copy(board)
                
                return 1, move
            
            else:
                return -1, "invalid move from current position"
                
            
            
            

            
            
            
        
        
        
        
        
        
        



        


    def update_board(board):
        pass

controller()