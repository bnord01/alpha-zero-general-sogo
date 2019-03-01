import Arena
from MCTS import MCTS
from tictactoe.TicTacToeGame import TicTacToeGame, display
from tictactoe.TicTacToePlayers import HumanTicTacToePlayer
from NeuralNet import NeuralNet
from Game import Game

import numpy as np

"""
use this script to play any two agents against each other, or play manually with
any agent.
"""

g = TicTacToeGame(3)

# all players
#rp = RandomPlayer(g).play
#gp = GreedyOthelloPlayer(g).play
hp = HumanTicTacToePlayer(g).play

# nnet players
class Config(object):
    def __init__(self):    
      self.num_sampling_moves = 30
      self.max_moves = 512  # for chess and shogi, 722 for Go.
      self.num_mcts_sims = 5000

      # Root prior exploration noise.
      self.root_dirichlet_alpha = 0.3  # for chess, 0.03 for Go and 0.15 for shogi.
      self.root_exploration_fraction = 0.0

      # UCB formula
      self.pb_c_base = 19652
      self.pb_c_init = 1.25

      # Load model

      self.load_model = True
      self.load_folder_file = ('./models/','checkpoint_25.pth.tar')


class NN(NeuralNet):
  def __init__(self,game:Game):
    self.game = game
  def predict(self, board):
    return np.ones(self.game.action_size())/self.game.action_size(), 0

nn = NN(g)
mcts1 = MCTS(g, nn, Config())
n1p = lambda x: np.argmax(mcts1.get_action_prob(x)[0])

arena = Arena.Arena(n1p, hp, g, display=display)
print(arena.play_games(20, verbose=True))
