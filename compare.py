SAVE1 = ('./save/', 'mixed2.h5')
#SAVE2 = ('./temp/', 'checkpoint_5.h5')
SAVE2 = ('./save/', 'mixed.h5')

MCTS_SIMS1 = 16*3
MCTS_SIMS2 = MCTS_SIMS1

GAMES = 10


SAMPLING1 = 8
SAMPLING2 = SAMPLING1

import numpy as np
from sogo.keras.NNet import NNetWrapper as NNet
from sogo.SogoPlayers import HumanSogoPlayer
from sogo.SogoGame import SogoGame, display
from MCTS import MCTS
import Arena


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
        self.num_mcts_sims = None

        # Root prior exploration noise.
        # for chess, 0.03 for Go and 0.15 for shogi.
        self.root_dirichlet_alpha = 0.3
        self.root_exploration_fraction = 0.0

        # UCB formula
        self.pb_c_base = 19652
        self.pb_c_init = 1.25


config1 = Config()
config1.num_mcts_sims = MCTS_SIMS1

nn1 = NNet(g)
nn1.load_checkpoint(*SAVE1)
mcts1 = MCTS(g, nn1, config1)


def player1(board):
    pi, root = mcts1.get_action_prob(board)
    a = np.argmax(pi) if np.sum(board) >= SAMPLING1 else np.random.choice(len(pi), p=pi)
    return a


config2 = Config()
config2.num_mcts_sims = MCTS_SIMS2

nn2 = NNet(g)
nn2.load_checkpoint(*SAVE2)
mcts2 = MCTS(g, nn2, config2)


def player2(board):
    pi, root = mcts2.get_action_prob(board)
    a = np.argmax(pi) if np.sum(board) >= SAMPLING2 else np.random.choice(len(pi), p=pi)
    return a


arena = Arena.Arena(player1, player2, g, display=display)
one, two, draw = arena.play_games(GAMES, verbose=False)

print(f"Result {one}:{two}, {draw} draws!")
if one > two:
    print("Player one (", *SAVE1, ") won!")
elif two > one:
    print("Player two (", *SAVE2, ") won!")
else:
    print("Draw!")
