from Coach import Coach
#from othello.OthelloGame import OthelloGame as Game
#from othello.tensorflow.NNet import NNetWrapper as nn
from sogo.SogoGame import SogoGame as Game
from sogo.keras.NNet import NNetWrapper as nn
from utils import *

args = dotdict({
    'numIters': 100,
    'numEps': 50,
    'tempThreshold': 15,
    'updateThreshold': 0.5,
    'maxlenOfQueue': 200000,
    'numMCTSSims': 150,
    'arenaCompare': 50,
    'pb_c_base' : 19652,
    'pb_c_init' : 1.25,

    'save_all_examples': False,
    'checkpoint': './temp/',
    'load_model': False,
    'load_folder_file': ('./saves/','latest'),
    'numItersForTrainExamplesHistory': 20,

})

if __name__=="__main__":
    g = Game(4)
    nnet = nn(g)

    #if args.load_model:
    #    nnet.load_checkpoint(args.load_folder_file[0], args.load_folder_file[1])

    c = Coach(g, nnet, args)
    if args.load_model:
        print("Load trainExamples from file")
        c.loadTrainExamples()
    c.learn()
