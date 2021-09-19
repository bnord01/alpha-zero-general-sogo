from sogo.keras.NNet import ModelBuilder
from keras.models import *
from keras.layers import *
from keras.optimizers import *
from sogo.keras.NNet import NNArgs
from Game import Game
from sogo.keras.util import *


class AGZSmallNetBuilder(ModelBuilder):
    def build_model(game:Game, args:NNArgs)->Model:
        # game params
        board_x, board_y, board_z = game.board_size()
        action_size = game.action_size()

        input_boards = Input(
            shape=(board_x, board_y, board_z, 2))
        x = input_boards

        # args
        filters3d = 32
        num3d = 5
        filters2d = 128
        num2d = 10
        v_filts = 128

        # Upsampling to 4 x 4 x 4 x filters3d
        x = conv3d_bn(x, filters3d, 3, 3, 3, name='in3d')

        # res blocks 3d
        for i in range(num3d):
            x = res_block_3d(x, filters3d, name=f'res3d{i}')

        # reshaping to 4 x 4 x 4*filters3d
        x = Reshape((4, 4, -1))(x)

        # upsampling to 4 x 4 x filters2d
        x = conv2d_bn(x, filters2d, 3, 3, name='in2d')

        # res blocks 2d
        for i in range(num2d):
            x = res_block_2d(x, filters2d, name=f'res2d{i}')

        # pi head
        y = conv2d_bn(x, 2, 1, 1, name='pi_conv')
        y = Flatten()(y)

        # v head
        z = conv2d_bn(x, 1, 1, 1, name='v_conv')
        z = Flatten()(z)
        z = Dense(v_filts, name='dense_v')(z)
        z = Activation('relu')(z)

        pi = Dense(action_size, activation='softmax', name='pi')(y)
        v = Dense(1, activation='tanh', name='v')(z)

        model = Model(inputs=input_boards, outputs=[pi, v])
        model.compile(
            loss=['categorical_crossentropy', 'mean_squared_error'], optimizer=adam_v2.Adam(args.lr))

        return model
