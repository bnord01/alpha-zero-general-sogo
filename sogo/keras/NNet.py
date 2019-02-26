import keras
from sogo.keras.SogoNNet import SogoNNet as onnet
from NeuralNet import NeuralNet
from utils import *
import os
import numpy as np
import sys
sys.path.append('..')


"""
NeuralNet wrapper class for the SogoNNet.

Author: Benedikt Nordhoff

Based on the NNet by SourKream and Surag Nair.
"""

args = dotdict({
    'lr': 0.001,
    'dropout': 0.3,
    'epochs': 5,
    'batch_size': 2*1024,
    'cuda': False,
    'num_channels': 512,
})


class NNetWrapper(NeuralNet):
    def __init__(self, game):
        self.nnet = onnet(game, args)
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
                            batch_size=args.batch_size, epochs=args.epochs, 
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
