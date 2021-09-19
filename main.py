from Config import Config
from sogo.keras.NNet import NNArgs    
from Coach import Coach
from sogo.SogoGame import SogoGame as Game
from sogo.keras.NNet import NNetWrapper
from sogo.keras.LargeNetBuilder import LargeNetBuilder
from sogo.keras.SmallNetBuilder import SmallNetBuilder
from sogo.keras.AGZLargeNetBuilder import AGZLargeNetBuilder
from sogo.keras.AGZSmallNetBuilder import AGZSmallNetBuilder
from sogo.keras.SimpleNetBuilder import SimpleNetBuilder

config = Config(
    start_iteration=1,
    num_iterations=100,
    num_episodes=10,
    episode_queue_length=200000,
    save_all_examples=False,
    checkpoint='./checkpoints/agz_large/',
    load_model=False,
    load_examles=False,
    load_folder_file=('./agz/', 'latest.h5'),
    tensorboard_dir='./logs/agz_large/',
    iteration_history_length=20,
    num_sampling_moves=5,
    num_mcts_sims=15,
    reuse_mcts_root=True,
    mcts_discount=0.925,
    train_discount=0.925,

    # Neural net args
    nnet_args = NNArgs(builder = AGZLargeNetBuilder,
                       lr=0.02,
                       batch_size=2048,
                       epochs=20),

    # Root prior exploration noise.
    root_dirichlet_alpha=0.3,
    root_exploration_fraction=0.2,

    # UCB formula
    pb_c_base=19652,
    pb_c_init=1.25)


g = Game(4)
nnet = NNetWrapper(g, config)

if config.load_model:
    print("Loading model from ", *config.load_folder_file)
    nnet.load_checkpoint(*config.load_folder_file)

c = Coach(g, nnet, config)

if config.load_examles:
    print("Load train_examples from ",
            config.load_folder_file[0], config.load_folder_file[1]+".examples")
    c.loadtrain_examples()

c.learn()
