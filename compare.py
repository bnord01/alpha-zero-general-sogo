from sogo.keras.LargeNetBuilder import LargeNetBuilder
from sogo.keras.SmallNetBuilder import SmallNetBuilder
from sogo.keras.AGZLargeNetBuilder import AGZLargeNetBuilder
from sogo.keras.AGZSmallNetBuilder import AGZSmallNetBuilder
from sogo.keras.SimpleNetBuilder import SimpleNetBuilder

# Define the models

BUILDER1 = AGZLargeNetBuilder
SAVE1 = ('./pretrained_models/sogo/agz_large/', 'best.h5')

BUILDER2 = AGZSmallNetBuilder
SAVE2 = ('./pretrained_models/sogo/agz_small/', 'best.h5')

# Number of Games to play
GAMES = 30

# Number of playouts in the Monte Carlo Tree Search
MCTS_SIMS1 = 0
MCTS_SIMS2 = MCTS_SIMS1

# Number of plays to sample according to the provided distribution before always choosing the best.
SAMPLING1 = 10
SAMPLING2 = SAMPLING1

# Discount factor for the MCTS
DISCOUNT1 = 0.925
DISCOUNT2 = DISCOUNT1

###########################
#### Setup the players ####
###########################

from Config import Config
from sogo.keras.NNet import NNetWrapper
from sogo.keras.NNet import NNArgs    
import numpy as np
from sogo.SogoGame import SogoGame, display
from MCTS import MCTS
import Arena

g = SogoGame(4)

####################
#### Player one ####
####################

config1 = Config(
    num_sampling_moves=SAMPLING1,
    num_mcts_sims=MCTS_SIMS1,
    
    # Root prior exploration noise.
    root_dirichlet_alpha=0.3,
    root_exploration_fraction=0.0,
    mcts_discount=DISCOUNT1,

    # UCB formula
    pb_c_base=19652,
    pb_c_init=1.25)

config1.nnet_args = NNArgs(builder = BUILDER1,
                              lr=0.02,
                              batch_size=2048,
                              epochs=20)


nn1 = NNetWrapper(g, config1)

nn1.load_checkpoint(*SAVE1)
mcts1 = MCTS(g, nn1, config1)

def nnpred1(board):
    pi, _ = nn1.predict(board)
    valids = g.valid_actions(board,1)
    pi = pi * valids
    s = np.sum(pi)
    if s > 0:
        return pi / s
    else:
        print('NN1 no mass on valid actions!')
        return np.ones((g.action_size(),))/g.action_size()

def player1(board):    
    if MCTS_SIMS1 > 0:
        pi, _ = mcts1.get_action_prob(board) 
    else:
        pi = nnpred1(board)    
    a = np.argmax(pi) if np.sum(board) >= SAMPLING1 else np.random.choice(len(pi), p=pi)
    return a

####################
#### Player two ####
####################

config2 = Config(
    num_sampling_moves=SAMPLING2,
    num_mcts_sims=MCTS_SIMS2,
    
    # Root prior exploration noise.
    root_dirichlet_alpha=0.3,
    root_exploration_fraction=0.0,
    mcts_discount=DISCOUNT2,

    # UCB formula
    pb_c_base=19652,
    pb_c_init=1.25)

config2.nnet_args = NNArgs(builder = BUILDER2,
                              lr=0.02,
                              batch_size=2048,
                              epochs=20)


nn2 = NNetWrapper(g, config2)

nn2.load_checkpoint(*SAVE2)
mcts2 = MCTS(g, nn2, config2)

def nnpred2(board):
    pi, _ = nn2.predict(board)
    valids = g.valid_actions(board,1)
    pi = pi * valids
    s = np.sum(pi)
    if s > 0:
        return pi / s
    else:
        print('NN2 no mass on valid actions!')
        return np.ones((g.action_size(),))/g.action_size()

def player2(board):
    if MCTS_SIMS2 > 0:
        pi, _ = mcts2.get_action_prob(board) 
    else:
        pi = nnpred2(board)
    a = np.argmax(pi) if np.sum(board) >= SAMPLING2 else np.random.choice(len(pi), p=pi)
    return a

#############################
#### Compare the players ####
#############################

arena = Arena.Arena(player1, player2, g, display=display)
(one_first, one_second), (two_first, two_second), (draw_one_first, draw_two_first) = arena.play_games(GAMES, verbose=False)

one = one_first + one_second
two = two_first + two_second
draw = draw_one_first + draw_two_first

print()
print(f"Result overall {one}:{two}, {draw} draws")
if one > two:
    print("Player one (", *SAVE1, ") won!")
elif two > one:
    print("Player two (", *SAVE2, ") won!")
else:
    print("Draw!")

print()
print("Player one first (", *SAVE1,"):")
print(f"Result {one_first}:{two_second}, {draw_one_first} draws")

print()
print("Player two first (", *SAVE2,"):")
print(f"Result {two_first}:{one_second}, {draw_two_first} draws")
