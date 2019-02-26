
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

    def init_board(self):        
        'Returns the initial state of the game.'
        return np.zeros((self.n, self.n, self.n, 2), dtype=np.int32)

    def board_size(self):
        # (a,b) tuple
        return (self.n, self.n, self.n)

    def action_size(self):
        # return number of actions
        return self.n*self.n

    def next_state(self, board, player, action):
        'Performs a valid action and returns the resulting state.'
        new_board = np.copy(board)
        x = self.action_x(action)
        y = self.action_y(action)
        z = self.action_z(board, x, y)            
        pl = int((player+1)/2)
        new_board[x, y, z, pl] = 1
        return (new_board, -player)

    def valid_actions(self, board, player):
        return np.array([self.valid_action(board,i) for i in range(0,self.action_size())])        

    def terminal_value(self, board, player):
        if evaluate(board[np.newaxis, :, :, :, 1]):
            return player
        if evaluate(board[np.newaxis, :, :, :, 0]):
            return -player
        if np.sum(board) == self.n ** 3:
            return 1e-4
        return 0

    def canonical_board(self, board, player):
        if player == 1:
            return board
        else:
            new_board = board.copy()
            new_board[:,:,:,0],new_board[:,:,:,1] = new_board[:,:,:,1],new_board[:,:,:,0].copy()            
            return new_board

    def symmetries(self, board, pi):
        pi_board = np.reshape(pi, (self.n, self.n), order='F')
        l = []
        for i in range(1, 5):
            for j in [True, False]:
                newB = np.rot90(board, i)
                newPi = np.rot90(pi_board, i)
                if j:
                    newB = np.fliplr(newB)
                    newPi = np.fliplr(newPi)
                l += [(newB, np.ravel(newPi, order='F'))]
        return l

    def string_representation(self, board):        
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
    (nx,ny,nz,_) = board.shape
    for z in range(nz-1,-1,-1):
        print("z"+str(z)+"+", end="")
        for _ in range(nx):
            print ("--", end="")
        print("+")
        for y in range(ny-1,-1,-1):
            print(y, "|",end="")    # print the row #
            for x in range(nx):
                if board[x,y,z,0]:
                    print("O ", end="")
                elif board[x,y,z,1]:
                    print("X ", end="")
                else:
                    if x==nx:
                        print("-",end="")
                    else:
                        print("- ",end="")
            print("|")
        print("z"+str(z)+"+", end="")
        for _ in range(nx):
            print ("--", end="")
        print("+")
        print("   ", end="")
        for x in range(nx):
            print (x,"", end="")
        print("")        
    print("--")
