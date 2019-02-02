from Coach import Coach
#from othello.OthelloGame import OthelloGame as Game
#from othello.tensorflow.NNet import NNetWrapper as nn
from sogo.SogoGame import SogoGame as Game
from sogo.keras.NNet import NNetWrapper as nn
from utils import *

args = dotdict({
    'numIters': 100,
    'numEps': 1000,
    'tempThreshold': 15,
    'updateThreshold': 0.5,
    'maxlenOfQueue': 200000,
    'numMCTSSims': 30,
    'arenaCompare': 50,
    'cpuct': 1,

    'save_all_examples': False,
    'checkpoint': './temp/',
    'load_model': True,
    'load_folder_file': ('./saves/','checkpoint_12.pth.tar'),
    'numItersForTrainExamplesHistory': 20,

})

if __name__=="__main__":
    g = Game(4)
    nnet = nn(g)

    if args.load_model:
        nnet.load_checkpoint(args.load_folder_file[0], args.load_folder_file[1])

    c = Coach(g, nnet, args)
    if args.load_model:
        print("Load trainExamples from file")
        c.loadTrainExamples()
    c.learn()
