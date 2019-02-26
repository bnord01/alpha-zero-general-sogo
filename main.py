from Coach import Coach
#from othello.OthelloGame import OthelloGame as Game
#from othello.tensorflow.NNet import NNetWrapper as nn
from sogo.SogoGame import SogoGame as Game
from sogo.keras.NNet import NNetWrapper as nn

class Config(object):
    def __init__(self):    
     
      self.num_iterations = 100
      self.num_episodes = 10
      self.episode_queue_length = 200000
      self.save_all_examples = True
      self.checkpoint = './temp/'
      self.load_model = True
      self.load_examles = False
      self.load_folder_file = ('./save/','new_mcts_15.pth.tar')
      self.iteration_history_length = 20

      self.num_sampling_moves = 10
      self.num_mcts_sims = 514
      self.reuse_mcts_root = True

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
        print("Loading model from ", *args.load_folder_file)
        nnet.load_checkpoint(*args.load_folder_file)

    c = Coach(g, nnet, args)

    if args.load_examles:
        print("Load trainExamples from ", args.load_folder_file[0], args.load_folder_file[1]+".examples")
        c.loadTrainExamples()
        
    c.learn()
