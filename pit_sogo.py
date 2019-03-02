import Arena
from MCTS import MCTS
from sogo.SogoGame import SogoGame, display
from sogo.SogoPlayers import HumanSogoPlayer

import numpy as np
from NeuralNet import NeuralNet
from sogo.keras.NNet import NNetWrapper as NNet

from Game import Game

from Timer import Timer

"""
use this script to play any two agents against each other, or play manually with
any agent.
"""

g = SogoGame(4)


# nnet players
class Config(object):
    def __init__(self):
        self.num_mcts_sims = 16*3

        # Root prior exploration noise.
        # for chess, 0.03 for Go and 0.15 for shogi.
        self.root_dirichlet_alpha = 0.3
        self.root_exploration_fraction = 0.0

        # UCB formula
        self.pb_c_base = 19652
        self.pb_c_init = 1.25

        # Load model

        self.load_model = True
        self.load_folder_file = ('./save/', 'mixed3.h5')


class NN(NeuralNet):
    def __init__(self, game: Game):
        self.game = game

    def predict(self, board):
        return np.ones(self.game.action_size())/self.game.action_size(), 0


config = Config()
# nn = NN(g)

nn = NNet(g, config)
nn.load_checkpoint(*(config.load_folder_file))
mcts1 = MCTS(g, nn, config)
hp = HumanSogoPlayer(g)

def human_player(board):
    pi, v = nn.predict(board)
    print(f"NNet: {np.array2string(np.array(pi), precision=2, separator=',', suppress_small=True, max_line_width=200)} value: {v}, prefered move: {hp.format(np.argmax(pi))}")

    a = hp.play(board)
    return a


def ai_player(board):
    pi, v = nn.predict(board)
    print(f"NNet: {np.array2string(np.array(pi), precision=2, separator=',', suppress_small=True, max_line_width=200)} value: {v}, prefered move: {hp.format(np.argmax(pi))}")

    with Timer("AI"):
        pi, _ = mcts1.get_action_prob(board)
    a = np.argmax(pi)
    print(f"MCTS: {np.array2string(np.array(pi), precision=2, separator=',', suppress_small=True, max_line_width=200)} selected move: {hp.format(a)}")

    return a


p1, p2 = ai_player, human_player
while True:
    arena = Arena.Arena(p1, p2, g, display=display)
    arena.play_games(2, verbose=True)
    p1, p2 = p2, p1
