from Coach import Coach
#from othello.OthelloGame import OthelloGame as Game
#from othello.tensorflow.NNet import NNetWrapper as nn
from sogo.SogoGame import SogoGame as Game
from sogo.keras.NNet import NNetWrapper as nn
from utils import *

class Config(object):
    def __init__(self):    
     
      self.numIters = 100
      self.numEps = 50
      self.tempThreshold = 15
      self.updateThreshold = 0.5
      self.maxlenOfQueue = 200000
      self.save_all_examples = False
      self.checkpoint = './temp/'
      self.load_model = False
      self.load_folder_file = ('./saves/','latest')
      self.numItersForTrainExamplesHistory = 20

      self.num_sampling_moves = 10
      self.max_moves = 512  # for chess and shogi, 722 for Go.
      self.numMCTSSims = 400

      # Root prior exploration noise.
      self.root_dirichlet_alpha = 0.3  # for chess, 0.03 for Go and 0.15 for shogi.
      self.root_exploration_fraction = 0.2

      # UCB formula
      self.pb_c_base = 19652
      self.pb_c_init = 1.25


args = Config()

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
