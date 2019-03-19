import keras
from sogo.keras.agz.SogoNNet import SogoNNet
from NeuralNet import NeuralNet
from utils import *
import os
import numpy as np
import sys
from Config import Config
from Game import Game


"""
NeuralNet wrapper class for the SogoNNet.

Author: Benedikt Nordhoff

Based on the NNet by SourKream and Surag Nair.
"""


class NNArgs(object):
    def __init__(self,
                 lr=0.001,
                 epochs=10,
                 batch_size=1024):
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size


class NNetWrapper(NeuralNet):
    def __init__(self, game: Game, config: Config):
        self.config = config
        self.args = config.nnet_args if config and config.nnet_args else NNArgs()
        self.nnet = SogoNNet(game, self.args)

        self.board_x, self.board_y, self.board_z = game.board_size()
        self.action_size = game.action_size()

    def train(self, examples):
        """
        examples: list of examples, each example is of form (board, pi, v)
        """
        input_boards, target_pis, target_vs = list(zip(*examples))
        input_boards = np.asarray(input_boards)
        target_pis = np.asarray(target_pis)
        target_vs = np.asarray(target_vs)
        tb_callback = keras.callbacks.TensorBoard(
            log_dir='./logs', histogram_freq=0, write_graph=True, write_images=True)

        self.nnet.model.fit(x=input_boards, y=[target_pis, target_vs],
                            batch_size=self.args.batch_size, epochs=self.args.epochs,
                            callbacks=[tb_callback])

    def predict(self, board):
        board = board[np.newaxis, :, :]
        pi, v = self.nnet.model.predict(board)
        return pi[0], v[0]

    def save_checkpoint(self, folder='checkpoint', filename='checkpoint.h5'):
        filepath = os.path.join(folder, filename)
        if not os.path.exists(folder):
            os.mkdir(folder)
        self.nnet.model.save_weights(filepath)

    def load_checkpoint(self, folder='checkpoint', filename='checkpoint.h5'):
        filepath = os.path.join(folder, filename)
        if not os.path.exists(filepath):
            raise Exception(f"No model in path '{filepath}'")
        self.nnet.model.load_weights(filepath)
