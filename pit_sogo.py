import Arena
from MCTS import MCTS
from sogo.SogoGame import SogoGame, display
from sogo.SogoPlayers import *

import numpy as np
from utils import *

"""
use this script to play any two agents against each other, or play manually with
any agent.
"""

g = SogoGame(4)

# all players
rp = RandomPlayer(g).play
hp = HumanSogoPlayer(g).play

arena = Arena.Arena(rp, hp, g, display=display)
print(arena.playGames(2, verbose=True))
