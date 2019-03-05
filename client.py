import tensorflow as tf
graph = tf.get_default_graph()

from MCTS import MCTS, Play
from sogo.SogoGame import SogoGame

from sogo.keras.NNet import NNetWrapper as NNet
import numpy as np

g = SogoGame(4)

# nnet players
from Config import Config
from sogo.keras.NNet import NNArgs
# nnet players
config = Config(
    load_folder_file=('./save/', 'mixed3.h5'),
    num_mcts_sims=20,
    root_dirichlet_alpha=0.3,
    root_exploration_fraction=0.0,
    pb_c_base=19652,
    pb_c_init=1.25)
config.nnet_args = NNArgs(lr=0.001, 
                              batch_size=1024, 
                              epochs=20)


nn = NNet(g,config)

# nn.load_checkpoint(*(config.load_folder_file))
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
    with graph.as_default():
        print(s)
        i, j = s['i'], s['j']
        a = 4*i+j
        play.apply(a)
        a = ai_player(play.canonical_board())
        play.apply(a)
        a = int(a)
        response = {'i':a//4,'j':a%4,'n':int(np.sum(play.canonical_board()[a//4,a%4]))}
        print(response)
        sio.emit('move', response)
   

@sio.on('connect')
def on_connect():
    print('Connected')
    
sio.connect('http://playsogo.herokuapp.com')
