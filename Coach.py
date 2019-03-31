from collections import deque
from Arena import Arena
from MCTS import MCTS
import numpy as np
from pytorch_classification.utils import Bar, AverageMeter
import time
import os
import sys
from pickle import Pickler, Unpickler
from random import shuffle


class Coach():
    """
    This class executes the self-play + learning. It uses the functions defined
    in Game and NeuralNet. config are specified in main.py.
    """

    def __init__(self, game, nnet, config):
        self.game = game
        self.nnet = nnet
        self.config = config
        self.mcts = MCTS(self.game, self.nnet, self.config)
        # history of examples from config.iteration_history_length latest iterations
        self.train_example_history = []
        self.skip_first_self_play = False  # can be overriden in loadtrain_examples()
        self.game_lengths = []

    def execute_episode(self):
        """
        This function executes one episode of self-play, starting with player 1.
        As the game is played, each turn is added as a training example to
        train_examples. The game is played till the game ends. After the game
        ends, the outcome of the game is used to assign values to each example
        in train_examples.

        It uses a temp=1 if episode_step < tempThreshold, and thereafter
        uses temp=0.

        Returns:
            train_examples: a list of examples of the form (canonical_board,pi,v)
                           pi is the MCTS informed policy vector, v is +1 if
                           the player eventually won the game, else -1.
        """
        train_examples = []
        board = self.game.init_board()
        player = 1
        episode_step = 0
        root = None
        while True:
            episode_step += 1

            pi, root = self.mcts.get_action_prob(
                board, root=root, player=player)

            canonical_board = self.game.canonical_board(board, player)
            sym = self.game.symmetries(canonical_board, pi)
            for brd, prb in sym:
                train_examples.append([brd, player, prb, episode_step])

            action = np.argmax(pi) if episode_step > self.config.num_sampling_moves \
                else np.random.choice(len(pi), p=pi)

            board, player = self.game.next_state(board, player, action)
            root = root.children[action] if self.config.reuse_mcts_root else None

            r = self.game.terminal_value(board, player)
            if r != 0:
                return [
                    (brd,
                     prb,
                     (r if plyr == player else -r)*(self.config.train_discount ** (episode_step-stp)))
                    for brd, plyr, prb, stp in train_examples
                ], episode_step

    def learn(self):
        """
        Performs numIters iterations with numEps episodes of self-play in each
        iteration. After every iteration, it retrains neural network with
        examples in train_examples (which has a maximium length of maxlenofQueue).
        It then pits the new neural network against the old one and accepts it
        only if it wins >= updateThreshold fraction of games.
        """

        for i in range(self.config.start_iteration, self.config.num_iterations+1):
            # bookkeeping
            print('------ITER ' + str(i) + '------')
            # examples of the iteration
            if not self.skip_first_self_play or i > 1:
                iteration_train_examples = deque(
                    [], maxlen=self.config.episode_queue_length)

                eps_lengths = []
                eps_time = AverageMeter()
                bar = Bar('Self Play', max=self.config.num_episodes)
                end = time.time()

                for eps in range(self.config.num_episodes):
                    # reset search tree
                    self.mcts = MCTS(self.game, self.nnet, self.config)
                    examples, steps = self.execute_episode()
                    iteration_train_examples += examples
                    eps_lengths.append(steps)

                    # bookkeeping + plot progress
                    eps_time.update(time.time() - end)
                    end = time.time()
                    bar.suffix = '({eps}/{maxeps}) Eps Time: {et:.3f}s | Total: {total:} | ETA: {eta:}'.format(eps=eps+1, maxeps=self.config.num_episodes, et=eps_time.avg,
                                                                                                               total=bar.elapsed_td, eta=bar.eta_td)
                    bar.next()
                bar.finish()

                print(
                    f"This episode game lengths, min:{np.min(eps_lengths):0.0f}, avg:{np.average(eps_lengths):0.2f}, max:{np.max(eps_lengths):0.0f}, std:{np.std(eps_lengths):0.2f}")

                self.game_lengths += eps_lengths

                print(
                    f"All episodes game lengths, min:{np.min(self.game_lengths):0.0f}, avg:{np.average(self.game_lengths):0.2f}, max:{np.max(self.game_lengths):0.0f}, std:{np.std(self.game_lengths):0.2f}")
                # save the iteration examples to the history
                self.train_example_history.append(iteration_train_examples)

            while len(self.train_example_history) > self.config.iteration_history_length:
                print("len(train_example_history) =", len(
                    self.train_example_history), " => remove the oldest train_examples")
                self.train_example_history.pop(0)
            # backup history to a file
            # NB! the examples were collected using the model from the previous iteration, so (i-1)
            self.savetrain_examples(i-1)

            if self.config.nnet_args.epochs > 0:
                # shuffle examlpes before training
                train_examples = []
                for e in self.train_example_history:
                    train_examples.extend(e)
                shuffle(train_examples)

                # training new network
                self.nnet.train(train_examples)

                self.nnet.save_checkpoint(
                    folder=self.config.checkpoint, filename="latest.h5")
                if i % 5 == 0:
                    self.nnet.save_checkpoint(
                        folder=self.config.checkpoint, filename=self.checkpoint_file_name(i))

    def checkpoint_file_name(self, iteration):
        return 'checkpoint_' + str(iteration) + '.h5'

    def savetrain_examples(self, iteration):
        folder = self.config.checkpoint
        if not os.path.exists(folder):
            os.makedirs(folder)
        filename = self.checkpoint_file_name(
            iteration) if (self.config.save_all_examples or iteration % self.config.iteration_history_length == 0) else "latest.h5"
        filepath = os.path.join(folder, filename+".examples")
        with open(filepath, "wb+") as f:
            Pickler(f).dump(self.train_example_history)
        f.closed

    def loadtrain_examples(self):
        modelFile = os.path.join(
            self.config.load_folder_file[0], self.config.load_folder_file[1])
        examples_file = modelFile+".examples"
        if not os.path.isfile(examples_file):
            print(examples_file)
            r = input("File with train_examples not found. Continue? [y|n]")
            if r != "y":
                sys.exit()
        else:
            print("File with train_examples found. Read it.")
            with open(examples_file, "rb") as f:
                self.train_example_history = Unpickler(f).load()
            f.closed
            # examples based on the model were already collected (loaded)
            self.skip_first_self_play = True
