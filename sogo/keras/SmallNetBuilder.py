from sogo.keras.NNet import ModelBuilder
from keras.models import *
from keras.layers import *
from keras import layers
from keras.optimizers import *
from sogo.keras.NNet import NNArgs
from Game import Game
from sogo.keras.util import *


class SmallNetBuilder(ModelBuilder):
    def build_model(game:Game, args:NNArgs)->Model:
        # game params
        board_x, board_y, board_z = game.board_size()
        action_size = game.action_size()

        input_boards = Input(
            shape=(board_x, board_y, board_z, 2))
        x = input_boards

        filters1 = 48

        # Upsampling to 4 x 4 x 4 x filters1
        branch1 = conv3d_bn(x, filters1, 1, 1, 1)

        branch3 = conv3d_bn(x, filters1, 1, 1, 1)
        branch3 = conv3d_bn(branch3, filters1 * 4, 3, 3, 3)
        branch3 = conv3d_bn(branch3, filters1, 3, 3, 3)

        branch114 = conv3d_bn(x, filters1 // 2, 1, 1, 1)
        branch114 = conv3d_bn(branch114, filters1, 1, 1, 4)

        branch141 = conv3d_bn(x, filters1 // 2, 1, 1, 1)
        branch141 = conv3d_bn(branch141, filters1, 1, 4, 1)

        branch411 = conv3d_bn(x, filters1 // 2, 1, 1, 1)
        branch411 = conv3d_bn(branch411, filters1, 4, 1, 1)

        # 4 x 4 x 4 x filters1*5
        x = concatenate(
            [branch1, branch3, branch114, branch141, branch411],
            name='mixed0')

        # 4 x 4 x filters1*20
        x = Reshape((4, 4, -1))(x)

        branch1x1 = conv2d_bn(x, 64, 1, 1)

        branch4x4 = conv2d_bn(x, 48, 1, 1)
        branch4x4 = conv2d_bn(branch4x4, 64, 4, 4)

        branch3x3dbl = conv2d_bn(x, 64, 1, 1)
        branch3x3dbl = conv2d_bn(branch3x3dbl, 96, 3, 3)
        branch3x3dbl = conv2d_bn(branch3x3dbl, 96, 3, 3)

        branch_pool = layers.AveragePooling2D((3, 3),
                                              strides=(1, 1),
                                              padding='same')(x)
        branch_pool = conv2d_bn(branch_pool, 64, 1, 1)
        x = layers.concatenate(
            [branch1x1, branch4x4, branch3x3dbl, branch_pool],
            name='mixed1')

        # 4 x 4
        branch1x1 = conv2d_bn(x, 64, 1, 1)

        branch4x4 = conv2d_bn(x, 48, 1, 1)
        branch4x4 = conv2d_bn(branch4x4, 64, 1, 4)
        branch4x4 = conv2d_bn(branch4x4, 128, 4, 1)

        branch3x3dbl = conv2d_bn(x, 64, 1, 1)
        branch3x3dbl = conv2d_bn(branch3x3dbl, 96, 3, 3)
        branch3x3dbl = conv2d_bn(branch3x3dbl, 96, 3, 3)

        branch_pool = layers.AveragePooling2D((3, 3),
                                              strides=(1, 1),
                                              padding='same')(x)
        branch_pool = conv2d_bn(branch_pool, 64, 1, 1)
        x = layers.concatenate(
            [branch1x1, branch4x4, branch3x3dbl, branch_pool],
            name='mixed2')

        x = Flatten()(x)
        x = Dense(2048, name='dense1')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)

        x = Dense(1024, name='dense2')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)

        pi = Dense(action_size, activation='softmax', name='pi')(x)
        v = Dense(1, activation='tanh', name='v')(x)

        model = Model(inputs=input_boards, outputs=[pi, v])
        model.compile(
            loss=['categorical_crossentropy', 'mean_squared_error'], optimizer=adam_v2.Adam(args.lr))
        return model
