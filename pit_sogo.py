import Arena
from MCTS import MCTS
from sogo.SogoGame import SogoGame, display
from sogo.SogoPlayers import HumanSogoPlayer

import numpy as np
from NeuralNet import NeuralNet
from Game import Game

import numpy as np

"""
use this script to play any two agents against each other, or play manually with
any agent.
"""

g = SogoGame(4)


# nnet players
class Config(object):
    def __init__(self):    
      self.num_sampling_moves = 30
      self.max_moves = 512  # for chess and shogi, 722 for Go.
      self.numMCTSSims = 5000

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
    return np.ones(self.game.getActionSize())/self.game.getActionSize(), 0

nn = NN(g)
mcts1 = MCTS(g, nn, Config())
n1p = lambda x: np.argmax(mcts1.getActionProb(x, temp=0))

hp = HumanSogoPlayer(g).play
arena = Arena.Arena(n1p, hp, g, display=display)
print(arena.playGames(20, verbose=True))

