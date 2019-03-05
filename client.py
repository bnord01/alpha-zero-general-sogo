
from MCTS import MCTS, Play
from sogo.SogoGame import SogoGame

from sogo.keras.NNet import NNetWrapper as NNet
import numpy as np

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


config = Config()

nn = NNet(g)
nn.load_checkpoint(*(config.load_folder_file))
mcts1 = MCTS(g, nn, config)

def ai_player(board):
    pi, _ = mcts1.get_action_prob(board)
    a = np.argmax(pi)
    return a

import socketio

sio = socketio.Client()

play = Play(g, g.init_board())


@sio.on('move')
def on_move(s):
    print(s)
    i, j = s['i'], s['j']
    a = 4*i+j
    play.apply(a)    
    a = ai_player(play.canonical_board())
    play.apply(a)
    sio.emit({'i':a//4,'j':a%4,'n':np.sum(play.canonical_board[a//4,a%4])})
   

@sio.on('connect')
def on_connect():
    print('Connected')
    
sio.connect('http://playsogo.herokuapp.com')
