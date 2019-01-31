from keras.models import *
from keras.layers import *
from keras.optimizers import *

"""
NeuralNet for the game of Sogo.
"""
class SogoNNet():
    def __init__(self, game, args):
        # game params
        self.board_x, self.board_y, self.board_z = game.getBoardSize()
        self.action_size = game.getActionSize()
        self.args = args

        num_4x1 = 32
        num_4x4 = 64

        self.input_boards = Input(shape=(self.board_x, self.board_y, self.board_z, 2))
        t = self.input_boards

        t144 = Flatten()(Conv3D(num_4x1, (self.board_x, 1, 1), padding='valid')(t))
        t414 = Flatten()(Conv3D(num_4x1, (1, self.board_y, 1), padding='valid')(t))
        t441 = Flatten()(Conv3D(num_4x1, (1, 1, self.board_z), padding='valid')(t))

        t114 = Flatten()(Conv3D(num_4x4, (self.board_x, self.board_y, 1), padding='valid')(t))
        t141 = Flatten()(Conv3D(num_4x4, (self.board_x, 1, self.board_z), padding='valid')(t))
        t411 = Flatten()(Conv3D(num_4x4, (1, self.board_y, self.board_z), padding='valid')(t))

        t = Concatenate()([t144, t414, t441, t114, t141, t411, Flatten()(t)])
        t = BatchNormalization()(t)
        t = Activation('relu')(t)

        t = Dense(2048)(t)
        t = BatchNormalization()(t)
        t = Activation('relu')(t)

        t = Dense(512)(t)
        t = BatchNormalization()(t)
        t = Activation('relu')(t)

        self.pi = Dense(self.action_size, activation='softmax', name='pi')(t)   
        self.v = Dense(1, activation='tanh', name='v')(t)

        self.model = Model(inputs=self.input_boards, outputs=[self.pi, self.v])
        self.model.compile(loss=['categorical_crossentropy','mean_squared_error'], optimizer=Adam(args.lr))
