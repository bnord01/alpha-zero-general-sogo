from sogo.keras.NNet import ModelBuilder
from keras.models import *
from keras.layers import *
from keras.optimizers import *
from sogo.keras.NNet import NNArgs
from Game import Game
from sogo.keras.util import *


class SimpleNetBuilder(ModelBuilder):
    def build_model(game:Game, args:NNArgs)->Model:
        # game params
        board_x, board_y, board_z = game.board_size()
        action_size = game.action_size()

        input_boards = Input(
            shape=(board_x, board_y, board_z, 2))
        t = input_boards

        num_4x1 = 32
        num_4x4 = 64

        t144 = Flatten()(Conv3D(num_4x1, (4, 1, 1), padding='valid')(t))
        t414 = Flatten()(Conv3D(num_4x1, (1, 4, 1), padding='valid')(t))
        t441 = Flatten()(Conv3D(num_4x1, (1, 1, 4), padding='valid')(t))

        t114 = Flatten()(Conv3D(num_4x4, (4, 4, 1), padding='valid')(t))
        t141 = Flatten()(Conv3D(num_4x4, (4, 1, 4), padding='valid')(t))
        t411 = Flatten()(Conv3D(num_4x4, (1, 4, 4), padding='valid')(t))

        t = Concatenate()([t144, t414, t441, t114, t141, t411, Flatten()(t)])
        t = BatchNormalization()(t)
        t = Activation('relu')(t)

        t = Dense(2048)(t)
        t = BatchNormalization()(t)
        t = Activation('relu')(t)

        t = Dense(512)(t)
        t = BatchNormalization()(t)
        t = Activation('relu')(t)

        pi = Dense(action_size, activation='softmax', name='pi')(t)
        v = Dense(1, activation='tanh', name='v')(t)        

        model = Model(inputs=input_boards, outputs=[pi, v])
        model.compile(
            loss=['categorical_crossentropy', 'mean_squared_error'], optimizer=adam_v2.Adam(args.lr))
        return model
