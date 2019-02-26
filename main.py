from Coach import Coach

from sogo.SogoGame import SogoGame as Game
from sogo.keras.NNet import NNetWrapper as nn
GAME_SIZE = 4

# from tictactoe.TicTacToeGame import TicTacToeGame as Game
# from tictactoe.keras.NNet import NNetWrapper as nn
# GAME_SIZE = 3

class Config(object):
    def __init__(self):    
     
      self.num_iterations = 1000
      self.num_episodes = 10
      self.episode_queue_length = 200000
      self.save_all_examples = True
      self.checkpoint = './temp/'
      self.load_model = True
      self.load_examles = True
      self.load_folder_file = ('./save/','sogo1.pth.tar')
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
    g = Game(GAME_SIZE)
    nnet = nn(g)

    if args.load_model:
        print("Loading model from ", *args.load_folder_file)
        nnet.load_checkpoint(*args.load_folder_file)

    c = Coach(g, nnet, args)

    if args.load_examles:
        print("Load train_examples from ", args.load_folder_file[0], args.load_folder_file[1]+".examples")
        c.loadtrain_examples()
        
    c.learn()
