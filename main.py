from Coach import Coach
#from othello.OthelloGame import OthelloGame as Game
#from othello.tensorflow.NNet import NNetWrapper as nn
from sogo.SogoGame import SogoGame as Game
from sogo.keras.NNet import NNetWrapper as nn

class Config(object):
    def __init__(self):    
     
      self.numIters = 100
      self.numEps = 50
      self.tempThreshold = 15
      self.updateThreshold = 0.05
      self.maxlenOfQueue = 200000
      self.save_all_examples = True
      self.checkpoint = './temp/'
      self.load_model = True
      self.load_examles = True
      self.load_folder_file = ('./save/','new_mcst_10.pth.tar')
      self.numItersForTrainExamplesHistory = 20

      self.num_sampling_moves = 10
      self.max_moves = 512  # for chess and shogi, 722 for Go.
      self.numMCTSSims = 100

      self.arenaCompare = 5

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

    if args.load_model:
        nnet.load_checkpoint(*args.load_folder_file)

    c = Coach(g, nnet, args)

    if args.load_examles:
        print("Load trainExamples from file")
        c.loadTrainExamples()
        
    c.learn()
