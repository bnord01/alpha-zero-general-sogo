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
    in Game and NeuralNet. args are specified in main.py.
    """

    def __init__(self, game, nnet, args):
        self.game = game
        self.nnet = nnet
        self.pnet = self.nnet.__class__(self.game)  # the competitor network
        self.args = args
        self.mcts = MCTS(self.game, self.nnet, self.args)
        # history of examples from args.iteration_history_length latest iterations
        self.train_example_history = []
        self.skip_first_self_play = False  # can be overriden in loadTrainExamples()
        self.game_lengths = []

    def execute_episode(self):
        """
        This function executes one episode of self-play, starting with player 1.
        As the game is played, each turn is added as a training example to
        trainExamples. The game is played till the game ends. After the game
        ends, the outcome of the game is used to assign values to each example
        in trainExamples.

        It uses a temp=1 if episodeStep < tempThreshold, and thereafter
        uses temp=0.

        Returns:
            trainExamples: a list of examples of the form (canonicalBoard,pi,v)
                           pi is the MCTS informed policy vector, v is +1 if
                           the player eventually won the game, else -1.
        """
        trainExamples = []
        board = self.game.getInitBoard()
        player = 1
        episodeStep = 0
        root = None
        while True:
            episodeStep += 1
            
            pi, root = self.mcts.get_action_prob(board, root=root, player=player)
                        
            canonicalBoard = self.game.getCanonicalForm(board, player)
            sym = self.game.getSymmetries(canonicalBoard, pi)
            for brd, prb in sym:
                trainExamples.append([brd, player, prb])

            action = np.argmax(pi) if episodeStep > self.args.num_sampling_moves \
                else np.random.choice(len(pi), p=pi)

            board, player = self.game.getNextState(board, player, action)            
            root = root.children[action] if self.args.reuse_mcts_root else None

            r = self.game.getGameEnded(board, player)
            if r != 0:
                return [(brd, prb, r if plyr == player else -r) for brd, plyr, prb in trainExamples], episodeStep

    def learn(self):
        """
        Performs numIters iterations with numEps episodes of self-play in each
        iteration. After every iteration, it retrains neural network with
        examples in trainExamples (which has a maximium length of maxlenofQueue).
        It then pits the new neural network against the old one and accepts it
        only if it wins >= updateThreshold fraction of games.
        """

        for i in range(1, self.args.num_iterations+1):
            # bookkeeping
            print('------ITER ' + str(i) + '------')
            # examples of the iteration
            if not self.skip_first_self_play or i > 1:
                iterationTrainExamples = deque(
                    [], maxlen=self.args.episode_queue_length)

                eps_lengths = []
                eps_time = AverageMeter()
                bar = Bar('Self Play', max=self.args.num_episodes)
                end = time.time()

                for eps in range(self.args.num_episodes):
                    # reset search tree
                    self.mcts = MCTS(self.game, self.nnet, self.args)
                    examples, steps = self.execute_episode()
                    iterationTrainExamples += examples
                    eps_lengths.append(steps)

                    # bookkeeping + plot progress
                    eps_time.update(time.time() - end)
                    end = time.time()
                    bar.suffix = '({eps}/{maxeps}) Eps Time: {et:.3f}s | Total: {total:} | ETA: {eta:}'.format(eps=eps+1, maxeps=self.args.num_episodes, et=eps_time.avg,
                                                                                                               total=bar.elapsed_td, eta=bar.eta_td)
                    bar.next()
                bar.finish()

                print(f"This episode game lengths, min:{np.min(eps_lengths):0.0f}, avg:{np.average(eps_lengths):0.2f}, max:{np.max(eps_lengths):0.0f}, std:{np.std(eps_lengths):0.2f}")

                self.game_lengths += eps_lengths

                print(f"All episodes game lengths, min:{np.min(self.game_lengths):0.0f}, avg:{np.average(self.game_lengths):0.2f}, max:{np.max(self.game_lengths):0.0f}, std:{np.std(self.game_lengths):0.2f}")
                # save the iteration examples to the history
                self.train_example_history.append(iterationTrainExamples)

            if len(self.train_example_history) > self.args.iteration_history_length:
                print("len(trainExamplesHistory) =", len(
                    self.train_example_history), " => remove the oldest trainExamples")
                self.train_example_history.pop(0)
            # backup history to a file
            # NB! the examples were collected using the model from the previous iteration, so (i-1)
            self.saveTrainExamples(i-1)

            # shuffle examlpes before training
            trainExamples = []
            for e in self.train_example_history:
                trainExamples.extend(e)
            shuffle(trainExamples)

            # training new network, keeping a copy of the old one
            self.nnet.save_checkpoint(
                folder=self.args.checkpoint, filename='temp.pth.tar')
            self.pnet.load_checkpoint(
                folder=self.args.checkpoint, filename='temp.pth.tar')
            pmcts = MCTS(self.game, self.pnet, self.args)

            self.nnet.train(trainExamples)
            nmcts = MCTS(self.game, self.nnet, self.args)

            print('PITTING AGAINST PREVIOUS VERSION')
            arena = Arena(lambda x: np.argmax(pmcts.getActionProb(x)),
                          lambda x: np.argmax(nmcts.getActionProb(x)), self.game)
            pwins, nwins, draws = arena.playGames(self.args.arena_compare)

            print('NEW/PREV WINS : %d / %d ; DRAWS : %d' %
                  (nwins, pwins, draws))
            if pwins+nwins > 0 and float(nwins)/(pwins+nwins) < self.args.update_threshold:
                print('REJECTING NEW MODEL')
                self.nnet.save_checkpoint(
                    folder=self.args.checkpoint, filename='rejected_'+self.getCheckpointFile(i))
                self.nnet.load_checkpoint(
                    folder=self.args.checkpoint, filename='temp.pth.tar')
            else:
                print('ACCEPTING NEW MODEL')
                self.nnet.save_checkpoint(
                    folder=self.args.checkpoint, filename=self.getCheckpointFile(i))
                self.nnet.save_checkpoint(
                    folder=self.args.checkpoint, filename='best.pth.tar')

    def getCheckpointFile(self, iteration):
        return 'checkpoint_' + str(iteration) + '.pth.tar'

    def saveTrainExamples(self, iteration):
        folder = self.args.checkpoint
        if not os.path.exists(folder):
            os.makedirs(folder)
        filename = self.getCheckpointFile(
            iteration) if self.args.save_all_examples else "latest"
        filepath = os.path.join(folder, filename+".examples")
        with open(filepath, "wb+") as f:
            Pickler(f).dump(self.train_example_history)
        f.closed

    def loadTrainExamples(self):
        modelFile = os.path.join(
            self.args.load_folder_file[0], self.args.load_folder_file[1])
        examplesFile = modelFile+".examples"
        if not os.path.isfile(examplesFile):
            print(examplesFile)
            r = input("File with trainExamples not found. Continue? [y|n]")
            if r != "y":
                sys.exit()
        else:
            print("File with trainExamples found. Read it.")
            with open(examplesFile, "rb") as f:
                self.train_example_history = Unpickler(f).load()
            f.closed
            # examples based on the model were already collected (loaded)
            # self.skip_first_self_play = True
