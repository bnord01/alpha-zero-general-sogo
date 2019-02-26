from MCTS import MCTS
from MCTS2 import MCTS2
from NeuralNet import NeuralNet
from Game import Game

import numpy as np


class NN(NeuralNet):
  def __init__(self,game:Game):
    self.game = game
  def predict(self, board):
    return np.ones(self.game.action_size())/self.game.action_size(), 0


class TestGame(Game):
  def init_board(self):
    return np.zeros(3)

  def board_size(self):
    return (3,)

  def action_size(self):
    return 3

  def next_state(self, board, player, action):
    b = board.copy()
    b[action] = b[action] + 1
    return b, -player

  def valid_actions(self, board, player):
    return range(3)

  def terminal_value(self, board, player):
    if board[1] > 0:
      return player
    if board[2] > 0:
      return -player
    return 0

  def getTerminal(self, board): 
    return sum(board) > 3 or board[1] > 0 or board[2] > 0

  def canonical_board(self, board, player):
    if player == 1:
      return board
    board = board.copy()
    temp = board[1]
    board[1] = board[2]
    board[2] = temp
    return board
  
  def string_representation(self, board):        
    return board.tostring()


class Config(object):
    def __init__(self):    
      self.num_sampling_moves = 30
      self.max_moves = 512  # for chess and shogi, 722 for Go.
      self.num_mcts_sims = 50000

      # Root prior exploration noise.
      self.root_dirichlet_alpha = 0.3  # for chess, 0.03 for Go and 0.15 for shogi.
      self.root_exploration_fraction = 0.0

      # UCB formula
      self.pb_c_base = 19652
      self.pb_c_init = 1.25

      # Load model

      self.load_model = True
      self.load_folder_file = ('./models/','checkpoint_25.pth.tar')

from tictactoe.TicTacToeGame import TicTacToeGame

game = TicTacToeGame(3)
config = Config()
board, _ = game.next_state(game.init_board(),-1,0)
board, _ = game.next_state(board,-1,4)
board, _ = game.next_state(board,1,3)
board, _ = game.next_state(board,1,6)


#from tictactoe.keras.NNet import NNetWrapper as NNet
#nn1 = NNet(game)
#nn1.load_checkpoint('pretrained_models/tictactoe/keras/','best-25eps-25sim-10epch.pth.tar')
#mcts1 = MCTS(game,nn1,config)

from timeit import default_timer as timer

#start = timer()
#print(mcts1.get_action_prob(board))
#end = timer()
#print(f"With neural net: {end-start}")

board = game.init_board()

board, _ = game.next_state(game.init_board(),-1,1)
board, _ = game.next_state(board,1,4)

board, _ = game.next_state(game.init_board(),-1,4)


nn2 = NN(game)
mcts2 = MCTS(game,nn2,config)
start = timer()
print(mcts2.get_action_prob(board))
end = timer()
print(f"With dummy net: {end-start}")
