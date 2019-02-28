from Timer import Timer

with Timer("Overall"):
  from MCTS import MCTS

  from sogo.SogoGame import SogoGame, display as display_board
  import numpy as np
  from sogo.keras.NNet import NNetWrapper as NNet

  class Config(object):
      def __init__(self):    
        self.num_sampling_moves = 30
        self.max_moves = 512  # for chess and shogi, 722 for Go.
        self.num_mcts_sims = 50

        # Root prior exploration noise.
        self.root_dirichlet_alpha = 0.3  # for chess, 0.03 for Go and 0.15 for shogi.
        self.root_exploration_fraction = 0.0

        # UCB formula
        self.pb_c_base = 19652
        self.pb_c_init = 1.25

        # Load model

        self.load_model = True
        self.load_folder_file = ('./save/','mixed3.h5')
          
      def initialize(self):
          global config 
          config = self
          self.game = SogoGame(4)

          self.nn = NNet(self.game)
          self.nn.load_checkpoint(*(config.load_folder_file))

          self.mcts = MCTS(self.game, self.nn, self)

      def setup_board(self, plays,verbose=True): 
          board = self.game.init_board()
          player = 1    
          for play in plays:
              board, player = self.game.next_state(board, player,play)
          return board, player
      
      def mcts_pred(self, plays, root=None, verbose=False):
          b,p = self.setup_board(plays, verbose = verbose)
          with Timer("MCTS prediction"):
              pi, root = self.mcts.get_action_prob(b, p, root)          
          return pi
            
      def test(self,blah):
            return np.sum(blah)

            
  config = Config()
            
  def mcts_task(value):
      global config
      return config.mcts_pred(value)

  args = [[],[0],[0,0],[0,1,0,1],[1,1,1,0,0,2,3,4],[0,1,2,3,4,5,6],[15,14,13],[1]]

  with Timer("Warmup"):
    config.initialize()
    [mcts_task(x) for x in args]

  with Timer("First Run"):
    [mcts_task(x) for x in args]

  with Timer("Second Run"):
    [mcts_task(x) for x in args]

  with Timer("Third Run"):
    [mcts_task(x) for x in args]

  with Timer("Fourth Run"):
    [mcts_task(x) for x in args+args]
    
  with Timer("Fifth Run"):
    [mcts_task(x) for x in args+args]