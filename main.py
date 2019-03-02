from Config import Config

config = Config(
    num_iterations=1000,
    num_episodes=200,
    episode_queue_length=200000,
    save_all_examples=False,
    checkpoint='./temp/',
    load_model=True,
    load_examles=True,
    load_folder_file=('./temp/', 'latest.h5'),
    iteration_history_length=20,
    num_sampling_moves=10,
    num_mcts_sims=512,
    reuse_mcts_root=True,

    # Root prior exploration noise.
    root_dirichlet_alpha=0.3,
    root_exploration_fraction=0.2,

    # UCB formula
    pb_c_base=19652,
    pb_c_init=1.25)

if __name__ == "__main__":
    from Coach import Coach

    from sogo.SogoGame import SogoGame as Game
    from sogo.keras.NNet import NNetWrapper as nn

    from sogo.keras.NNet import NNArgs
    config.nnet_args = NNArgs(lr=0.0001, batch_size=1024, epochs=10)

    g = Game(4)
    nnet = nn(g, config)

    if config.load_model:
        print("Loading model from ", *config.load_folder_file)
        nnet.load_checkpoint(*config.load_folder_file)

    c = Coach(g, nnet, config)

    if config.load_examles:
        print("Load train_examples from ",
              config.load_folder_file[0], config.load_folder_file[1]+".examples")
        c.loadtrain_examples()

    c.learn()
