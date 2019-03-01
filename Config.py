class Config(object):
    def __init__(self,
                 num_iterations=None,
                 num_episodes=None,
                 episode_queue_length=None,
                 save_all_examples=None,
                 checkpoint=None,
                 load_model=None,
                 load_examles=None,
                 load_folder_file=None,
                 iteration_history_length=None,

                 num_sampling_moves=None,
                 num_mcts_sims=None,
                 reuse_mcts_root=None,

                 # Root prior exploration noise.
                 root_dirichlet_alpha=None,
                 root_exploration_fraction=None,

                 # UCB formula
                 pb_c_base=None,
                 pb_c_init=None
                 ):
        self.num_iterations = num_iterations
        self.num_episodes = num_episodes
        self.episode_queue_length = episode_queue_length
        self.save_all_examples = save_all_examples
        self.checkpoint = checkpoint
        self.load_model = load_model
        self.load_examles = load_examles
        self.load_folder_file = load_folder_file
        self.iteration_history_length = iteration_history_length

        self.num_sampling_moves = num_sampling_moves
        self.num_mcts_sims = num_mcts_sims
        self.reuse_mcts_root = reuse_mcts_root

        # Root prior exploration noise.
        self.root_dirichlet_alpha = root_dirichlet_alpha
        self.root_exploration_fraction = root_exploration_fraction

        # UCB formula
        self.pb_c_base = pb_c_base
        self.pb_c_init = pb_c_init
