
import sys
sys.path.append('..')
from Game import Game

import numpy as np
from tf_sogo import evaluate


"""
Game class implementation for the game of Sogo.

"""
class SogoGame(Game):
    def __init__(self, n=3):
        self.n = n

    def getInitBoard(self):        
        'Returns the initial state of the game.'
        return np.zeros((4, 4, 4), dtype=np.int32)

    def getBoardSize(self):
        # (a,b) tuple
        return (self.n, self.n, self.n)

    def getActionSize(self):
        # return number of actions
        return self.n*self.n + 1

    def getNextState(self, board, player, action):
        'Performs a valid action and returns the resulting state.'
        new_state = np.copy(state)
        player = state[0, 0, 0, 2]
        if action < self.getActionSize():
            x = self.action_x(action)
            y = self.action_y(action)
            z = self.action_z(state, x, y)
            new_state[x, y, z] = player
        return (new_state, -player)

    def getValidMoves(self, board, player):
        # return a fixed size binary vector
        valids = [0]*self.getActionSize()
        b = Board(self.n)
        b.pieces = np.copy(board)
        legalMoves =  b.get_legal_moves(player)
        if len(legalMoves)==0:
            valids[-1]=1
            return np.array(valids)
        for x, y in legalMoves:
            valids[self.n*x+y]=1
        return np.array(valids)

    def getGameEnded(self, board, player):
        # return 0 if not ended, 1 if player 1 won, -1 if player 1 lost
        # player = 1
        b = Board(self.n)
        b.pieces = np.copy(board)

        if b.is_win(player):
            return 1
        if b.is_win(-player):
            return -1
        if b.has_legal_moves():
            return 0
        # draw has a very little value 
        return 1e-4

    def getCanonicalForm(self, board, player):
        # return state if player==1, else return -state if player==-1
        return player*board

    def getSymmetries(self, board, pi):
        # mirror, rotational
        assert(len(pi) == self.n**2+1)  # 1 for pass
        pi_board = np.reshape(pi[:-1], (self.n, self.n))
        l = []

        for i in range(1, 5):
            for j in [True, False]:
                newB = np.rot90(board, i)
                newPi = np.rot90(pi_board, i)
                if j:
                    newB = np.fliplr(newB)
                    newPi = np.fliplr(newPi)
                l += [(newB, list(newPi.ravel()) + [pi[-1]])]
        return l

    def stringRepresentation(self, board):
        # 8x8 numpy array (canonical board)
        return board.tostring()

    @staticmethod
    def action_x(num: int):
        return num % 4

    @staticmethod
    def action_y(num: int):
        return num // 4

    @staticmethod
    def action_z(state: np.ndarray, x: int, y: int):
        z = 0
        while state[x, y, z, 0] or state[x, y, z, 1]:
            z += 1
        return z

    def initial_state(self) -> np.ndarray:

    def num_actions(self) -> int:
        'Returns the number of valid actions.'
        return 17

    def valid_action(self, state: np.ndarray, action: int) -> bool:
        'Returns whether an action is valid in the given state or leads to the opponent winning.'
        return action == 16 or \
            not state[self.action_x(action), self.action_y(action), 3, 0] and \
            not state[self.action_x(action), self.action_y(action), 3, 1]

    def step(self, state: np.ndarray, action: int) -> np.ndarray:
        'Performs a valid action and returns the resulting state.'
        new_state = np.copy(state)
        player = state[0, 0, 0, 2]
        new_state[:, :, :, 2] = 1 - player
        if action < 16:
            x = self.action_x(action)
            y = self.action_y(action)
            z = self.action_z(state, x, y)
            new_state[x, y, z, player] = 1
        return new_state

    def winning_action(self, state: np.ndarray, action: int) -> bool:
        'Returns whether a valid action leads to a winning state for the performing player.'
        pass

    def terminal_state(self, state: np.ndarray) -> bool:
        'Returns whether a state is terminal.'
        return np.sum(state[..., 0:2]) == 64 or not self.evaluate_state(state) == 0

    def evaluate_state(self, state: np.ndarray) -> int:
        'Returns -1 iff the current player won the game, 1 if the other player won the game or 0 otherwise.'        
        player = state[0, 0, 0, 2]
        if evaluate(state[np.newaxis, :, :, :, 0]):
            return 1 if player else -1
        if evaluate(state[np.newaxis, :, :, :, 1]):
            return -1 if player else 1
        return 0
        
        

def display(board):
    n = board.shape[0]

    print("   ", end="")
    for y in range(n):
        print (y,"", end="")
    print("")
    print("  ", end="")
    for _ in range(n):
        print ("-", end="-")
    print("--")
    for y in range(n):
        print(y, "|",end="")    # print the row #
        for x in range(n):
            piece = board[y][x]    # get the piece to print
            if piece == -1: print("X ",end="")
            elif piece == 1: print("O ",end="")
            else:
                if x==n:
                    print("-",end="")
                else:
                    print("- ",end="")
        print("|")

    print("  ", end="")
    for _ in range(n):
        print ("-", end="-")
    print("--")


    