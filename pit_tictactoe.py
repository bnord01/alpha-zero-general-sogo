import Arena
from MCTS import MCTS
from tictactoe.TicTacToeGame import TicTacToeGame, display
from tictactoe.TicTacToePlayers import HumanTicTacToePlayer

import numpy as np
from NeuralNet import NeuralNet
from tictactoe.keras.NNet import NNetWrapper as NNet

from Game import Game

from Timer import Timer

"""
use this script to play any two agents against each other, or play manually with
any agent.
"""

g = TicTacToeGame(3)


# nnet players
class Config(object):
    def __init__(self):
        self.num_sampling_moves = 30
        self.max_moves = 512  # for chess and shogi, 722 for Go.
        self.num_mcts_sims = 20

        # Root prior exploration noise.
        # for chess, 0.03 for Go and 0.15 for shogi.
        self.root_dirichlet_alpha = 0.3
        self.root_exploration_fraction = 0.0

        # UCB formula
        self.pb_c_base = 19652
        self.pb_c_init = 1.25

        # Load model

        self.load_model = True
        self.load_folder_file = ('./temp/', 'checkpoint_2.pth.tar')


class NN(NeuralNet):
    def __init__(self, game: Game):
        self.game = game

    def predict(self, board):
        return np.ones(self.game.action_size())/self.game.action_size(), 0


config = Config()
# nn = NN(g)

nn = NNet(g)
nn.load_checkpoint(*(config.load_folder_file))
mcts1 = MCTS(g, nn, config)
hp = HumanTicTacToePlayer(g)

root = None


def advance_root(a):
    global root
    if root:
        root = root.children[a] if a in root.children else None


def human_player(board):
    a = hp.play(board)
    advance_root(a)
    return a


def ai_player(board):
    global root
    with Timer("AI"):
        pi, root = mcts1.get_action_prob(board, root=root)
    a = np.argmax(pi)
    advance_root(a)
    return a


p1, p2 = ai_player, human_player
while True:
    root = None
    arena = Arena.Arena(p1, p2, g, display=display)
    arena.play_games(2, verbose=True)
    p1, p2 = p2, p1
