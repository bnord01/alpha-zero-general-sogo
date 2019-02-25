import numpy as np

"""
Random and Human-ineracting players for the game of Sogo.

Author: Benedikt Nordhoff

Based on the TicTacToe players by Evgeny Tyurin.

"""
class RandomPlayer():
    def __init__(self, game):
        self.game = game

    def play(self, board):
        a = np.random.randint(self.game.getActionSize())
        valids = self.game.getValidMoves(board, 1)
        while valids[a]!=1:
            a = np.random.randint(self.game.getActionSize())
        return a


class HumanSogoPlayer():
    def __init__(self, game):
        self.game = game

    def format(self, i):
        return f"{int(i%self.game.n)} {int(i/self.game.n)}"

    def play(self, board):
        valid = self.game.getValidMoves(board, 1)
        for i in range(len(valid)):
            if valid[i]:
                print(self.format(i))
        while True: 
            try:
                a = input().strip()
                x, y = [int(x) for x in a.split(' ')]
                a = self.game.n * y + x
                if valid[a]:
                    break
                else:
                    print(f"Invalid input '{a}'")         
            except (ValueError, IndexError): 
                print(f"Couldn't parse input '{a}'")

        return a
