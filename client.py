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
    load_folder_file=('./save/', 'mcts1024_eps40_iter17.h5'),
    num_mcts_sims=150,
    mcts_discount=0.95,
    root_dirichlet_alpha=0.3,
    root_exploration_fraction=0.0,
    pb_c_base=19652,
    pb_c_init=1.25)
config.nnet_args = NNArgs(lr=0.001, 
                              batch_size=1024, 
                              epochs=20)


nn = NNet(g,config)

nn.load_checkpoint(*(config.load_folder_file))
mcts1 = MCTS(g, nn, config)

def ai_player(board):
    pi, _ = mcts1.get_action_prob(board)
    a = np.argmax(pi)
    return a

import socketio

sio = socketio.Client()

play = Play(g, g.init_board())

def check_win():
    global play
    

@sio.on('move')
def on_move(s):
    global play
    with graph.as_default():
        print(s)
        i, j = s['i'], s['j']
        a = i+4*j
        play.apply(a)
        if play.terminal():
            print('Game over, result:', play.terminal_value(), "for player", play.player)
            value = int(play.terminal_value() * play.player)
            sio.emit('reset', value)        
            play = Play(g, g.init_board())

        else:            
            a = ai_player(play.canonical_board())
            play.apply(a)
            i,j = int(a%4), int(a//4)
            response = {'i':i,'j':j}
            print(response)
            sio.emit('move', response)
            if play.terminal():
                print('Game over, result:', play.terminal_value(), "for player",play.player)
                value = int(play.terminal_value() * play.player)
                sio.emit('reset', value)  
                play = Play(g, g.init_board())
                

@sio.on('connect')
def on_connect():
    print('Connected')

@sio.on('reset')
def on_reset(v=None):
    global play
    print('Reset')
    play = Play(g, g.init_board())


    
sio.connect('http://playsogo.herokuapp.com')
#sio.connect('http://localhost:3003')
