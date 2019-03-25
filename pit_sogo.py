import Arena
from MCTS import MCTS
from sogo.SogoGame import SogoGame, display
from sogo.SogoPlayers import HumanSogoPlayer

import numpy as np
from NeuralNet import NeuralNet
from sogo.keras.agz.NNet import NNetWrapper as NNet

from Game import Game

from Timer import Timer

"""
use this script to play any two agents against each other, or play manually with
any agent.
"""

g = SogoGame(4)

from Config import Config
from sogo.keras.agz.NNet import NNArgs
# nnet players
config = Config(
    load_folder_file=('./agz/', 'latest.h5'),
    num_mcts_sims=150,
    root_dirichlet_alpha=0.3,
    root_exploration_fraction=0.0,
    mcts_discount=0.925,
    pb_c_base=19652,
    pb_c_init=1.25)
config.nnet_args = NNArgs(lr=0.001, 
                              batch_size=1024, 
                              epochs=20)
class NN(NeuralNet):
    def __init__(self, game: Game):
        self.game = game

    def predict(self, board):
        return np.ones(self.game.action_size())/self.game.action_size(), 0


# nn = NN(g)

nn = NNet(g,config)
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


p1, p2 = human_player, ai_player
while True:
    arena = Arena.Arena(p1, p2, g, display=display)
    arena.play_games(2, verbose=True)
    p1, p2 = p2, p1
