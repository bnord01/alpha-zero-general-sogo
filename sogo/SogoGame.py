
import sys
sys.path.append('..')
from Game import Game

import numpy as np
from sogo.tf_sogo import evaluate


"""
Game class implementation for the game of Sogo.

"""
class SogoGame(Game):
    def __init__(self, n=4):
        self.n = n

    def getInitBoard(self):        
        'Returns the initial state of the game.'
        return np.zeros((self.n, self.n, self.n, 2), dtype=np.int32)

    def getBoardSize(self):
        # (a,b) tuple
        return (self.n, self.n, self.n)

    def getActionSize(self):
        # return number of actions
        return self.n*self.n

    def getNextState(self, board, player, action):
        'Performs a valid action and returns the resulting state.'
        new_board = np.copy(board)
        x = self.action_x(action)
        y = self.action_y(action)
        z = self.action_z(board, x, y)            
        pl = int((player+1)/2)
        new_board[x, y, z, pl] = 1
        return (new_board, -player)

    def getValidMoves(self, board, player):
        return np.array([self.valid_action(board,i) for i in range(0,self.getActionSize())])        

    def getGameEnded(self, board, player):
        if evaluate(board[np.newaxis, :, :, :, 0]):
            return player
        if evaluate(board[np.newaxis, :, :, :, 1]):
            return -player
        if np.sum(board) == self.n ** 3:
            return 1e-4
        return 0

    def getCanonicalForm(self, board, player):
        if player == 1:
            return board
        else:
            new_board = board.copy()
            new_board[:,:,:,0],new_board[:,:,:,1] = new_board[:,:,:,1],new_board[:,:,:,0].copy()            
            return new_board

    def getSymmetries(self, board, pi):
        return [(board,pi)]

    def stringRepresentation(self, board):        
        # 8x8 numpy array (canonical board)        
        return board.tostring()

    def action_x(self, num: int):
        return num % self.n

    def action_y(self, num: int):
        return num // self.n 

    def action_z(self, state: np.ndarray, x: int, y: int):
        z = 0
        while state[x, y, z, 0] or state[x, y, z, 1]:
            z += 1
        return z

    def valid_action(self, state: np.ndarray, action: int) -> int:
        'Returns whether an action is valid in the given state or leads to the opponent winning.'
        return 1 if \
                not state[self.action_x(action), self.action_y(action), self.n-1, 0] and \
                not state[self.action_x(action), self.action_y(action), self.n-1, 1] \
            else 0        

def display(board):
    print(board[:,:,:,0]+board[:,:,:,1]*8)