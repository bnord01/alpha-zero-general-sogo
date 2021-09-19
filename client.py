import numpy as np
from MCTS import MCTS, Play
from sogo.SogoGame import SogoGame
from Config import Config
from sogo.keras.NNet import NNArgs
from sogo.keras.NNet import NNetWrapper
from sogo.keras.LargeNetBuilder import LargeNetBuilder
from sogo.keras.SmallNetBuilder import SmallNetBuilder
from sogo.keras.AGZLargeNetBuilder import AGZLargeNetBuilder
from sogo.keras.AGZSmallNetBuilder import AGZSmallNetBuilder
from sogo.keras.SimpleNetBuilder import SimpleNetBuilder


MCTS_SIMS = 10

FOLDER_FILE = ('./pretrained_models/sogo/agz_large/', 'best.h5')
BUILDER = AGZLargeNetBuilder


#############################
#### Setup the AI Player ####
#############################

g = SogoGame(4)

config = Config(
    load_folder_file=FOLDER_FILE,
    num_mcts_sims=MCTS_SIMS,
    mcts_discount=0.925,
    root_dirichlet_alpha=0.3,
    root_exploration_fraction=0.0,
    pb_c_base=19652,
    pb_c_init=1.25,
    nnet_args = NNArgs(builder=BUILDER,
                       lr=0.001, 
                       batch_size=1024, 
                       epochs=20))
                            


nn = NNetWrapper(g,config)

nn.load_checkpoint(*(config.load_folder_file))
mcts1 = MCTS(g, nn, config)

def nn_pred(board):
    pi, _ = nn.predict(board)
    valids = g.valid_actions(board,1)
    pi = pi * valids
    s = np.sum(pi)
    if s > 0:
        return pi / s
    else:
        print('NN1 no mass on valid actions!')
        return np.ones((g.action_size(),))/g.action_size()

def ai_player(board):
    if config.num_mcts_sims > 0:
        pi, _ = mcts1.get_action_prob(board) 
    else:
        pi = nn_pred(board)    
    a = np.argmax(pi)
    return a

###########################
#### Define the client ####
###########################

import socketio

sio = socketio.Client()

play = Play(g, g.init_board())
    

@sio.on('move')
def on_move(s):
    global play
    print(s)
    i, j = s['i'], s['j']
    a = i+4*j
    play.apply(a)
    if play.terminal():
        print('Game over, result:', play.terminal_value(), "for player", play.player)
        value = int(play.terminal_value() * play.player)
        sio.emit('reset', value)        
        play = Play(g, g.init_board())
        a = ai_player(play.canonical_board())
        play.apply(a)
        i,j = int(a%4), int(a//4)
        response = {'i':i,'j':j}
        print(response)
        sio.emit('move', response)

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
