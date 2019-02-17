from MCTS import MCTS
from MCTS2 import MCTS2
from NeuralNet import NeuralNet
from Game import Game

import numpy as np


class NN(NeuralNet):
  def __init__(self,game:Game):
    self.game = game
  def predict(self, board):
    return np.ones(self.game.getActionSize())/self.game.getActionSize(), 0


class TestGame(Game):
  def getInitBoard(self):
    return np.zeros(3)

  def getBoardSize(self):
    return (3,)

  def getActionSize(self):
    return 3

  def getNextState(self, board, player, action):
    b = board.copy()
    b[action] = b[action] + 1
    return b, -player

  def getValidMoves(self, board, player):
    return range(3)

  def getGameEnded(self, board, player):
    if board[1] > 0:
      return player
    if board[2] > 0:
      return -player
    return 0

  def getTerminal(self, board): 
    return sum(board) > 3 or board[1] > 0 or board[2] > 0

  def getCanonicalForm(self, board, player):
    if player == 1:
      return board
    board = board.copy()
    temp = board[1]
    board[1] = board[2]
    board[2] = temp
    return board
  
  def stringRepresentation(self, board):        
    return board.tostring()


class Config(object):
    def __init__(self):    
      self.num_sampling_moves = 30
      self.max_moves = 512  # for chess and shogi, 722 for Go.
      self.numMCTSSims = 100

      # Root prior exploration noise.
      self.root_dirichlet_alpha = 0.3  # for chess, 0.03 for Go and 0.15 for shogi.
      self.root_exploration_fraction = 0.0

      # UCB formula
      self.pb_c_base = 19652
      self.pb_c_init = 1.25

from tictactoe.TicTacToeGame import TicTacToeGame

game = TicTacToeGame(3)
nn = NN(game)
config = Config()

mcts = MCTS(game,nn,config)
board,_ = game.getNextState(game.getInitBoard(),1,0)
board,_ = game.getNextState(game.getInitBoard(),1,1)
print(mcts.get_action_prob(board))

#mcts2 = MCTS2(game,nn,config)
#print(mcts2.getActionProb(game.getInitBoard()))