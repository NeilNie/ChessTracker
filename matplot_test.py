import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import time
from matplotlib.animation import FuncAnimation

from stockfish import Stockfish

fig, axs = plt.subplots(8, 8, figsize=(8, 8))

plt.subplots_adjust(wspace=0, hspace=0)

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

stockfish = Stockfish(
    path="C:\\Users\\jjpla\\Downloads\\stockfish_15.1_win_x64_avx2\\stockfish_15.1_win_x64_avx2\\stockfish-windows-2022-x86-64-avx2.exe")


def redraw_board(a):
    print("run!", a)

    for i in range(64):
        x = i % 8
        y = i // 8

        axs[y, x].clear()

        if (i + y) % 2 == 0:
            axs[y, x].set_facecolor("white")
        else:
            axs[y, x].set_facecolor("grey")

        axs[y, x].xaxis.set_ticks([])
        axs[y, x].yaxis.set_ticks([])

        pos = chr(ord('a') + y) + str(x + 1)

        axs[y, x].text(.1, .15, piece_to_unicode[stockfish.get_what_is_on_square(
            pos)], fontsize=50)


# Animated_Figure = FuncAnimation(fig, redraw_board, interval=1000)
redraw_board(1)
plt.ion()
plt.pause(2)

print("here!")
stockfish.make_moves_from_current_position(["a2a4"])
redraw_board(0)
plt.show()
plt.pause(2)
