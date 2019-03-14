from keras.models import *
from keras.layers import *
from keras import layers
from keras.optimizers import *

"""
NeuralNet for the game of Sogo.
"""


def conv2d_bn(x,
              filters,
              num_row,
              num_col,
              padding='same',
              strides=(1, 1),
              name=None):
    """Utility function to apply conv + BN.
    # Arguments
        x: input tensor.
        filters: filters in `Conv2D`.
        num_row: height of the convolution kernel.
        num_col: width of the convolution kernel.
        padding: padding mode in `Conv2D`.
        strides: strides in `Conv2D`.
        name: name of the ops; will become `name + '_conv'`
            for the convolution and `name + '_bn'` for the
            batch norm layer.
    # Returns
        Output tensor after applying `Conv2D` and `BatchNormalization`.
    """
    if name is not None:
        bn_name = name + '_bn'
        conv_name = name + '_conv'
    else:
        bn_name = None
        conv_name = None
    x = Conv2D(
        filters, (num_row, num_col),
        strides=strides,
        padding=padding,
        use_bias=False,
        name=conv_name)(x)
    x = BatchNormalization(scale=False, name=bn_name,)(x)
    x = Activation('relu', name=name)(x)
    return x


def conv3d_bn(x,
              filters,
              num_row,
              num_col,
              num_lvl,
              padding='same',
              strides=(1, 1, 1),
              name=None):
    """Utility function to apply conv + BN.
    # Arguments
        x: input tensor.
        filters: filters in `Conv3D`.
        num_row: height of the convolution kernel.
        num_col: width of the convolution kernel.
        num_lvl: depth of the convolution kernel.
        padding: padding mode in `Conv3D`.
        strides: strides in `Conv3D`.
        name: name of the ops; will become `name + '_conv'`
            for the convolution and `name + '_bn'` for the
            batch norm layer.
    # Returns
        Output tensor after applying `Conv3D` and `BatchNormalization`.
    """
    if name is not None:
        bn_name = name + '_bn'
        conv_name = name + '_conv'
    else:
        bn_name = None
        conv_name = None
    x = Conv3D(
        filters, (num_row, num_col, num_lvl),
        strides=strides,
        padding=padding,
        use_bias=False,
        name=conv_name)(x)
    x = BatchNormalization(scale=False, name=bn_name)(x)
    x = Activation('relu', name=name)(x)
    return x


class SogoNNet():

    def __init__(self, game, args):
        # game params
        self.board_x, self.board_y, self.board_z = game.board_size()
        self.action_size = game.action_size()
        self.args = args

        self.input_boards = Input(
            shape=(self.board_x, self.board_y, self.board_z, 2))
        x = self.input_boards

        filters1 = 128

        # Upsampling to 4 x 4 x 4 x filters1
        branch1 = conv3d_bn(x, filters1, 1, 1, 1, name='branch1')

        branch3 = conv3d_bn(x, filters1, 1, 1, 1, name='branch3_1')
        branch3 = conv3d_bn(branch3, filters1 * 4, 3, 3, 3, name='branch3_2')
        branch3 = conv3d_bn(branch3, filters1, 3, 3, 3, name='branch3_3')

        branch114 = conv3d_bn(x, filters1 // 2, 1, 1, 1, name='branch114_1')
        branch114 = conv3d_bn(branch114, filters1, 1, 1, 4, name='branch114_2')

        branch141 = conv3d_bn(x, filters1 // 2, 1, 1, 1, name='branch141_1')
        branch141 = conv3d_bn(branch141, filters1, 1, 4, 1, name='branch141_2')

        branch411 = conv3d_bn(x, filters1 // 2, 1, 1, 1, name='branch411_1')
        branch411 = conv3d_bn(branch411, filters1, 4, 1, 1, name='branch411_2')

        # 4 x 4 x 4 x filters1*5
        x = concatenate(
            [branch1, branch3, branch114, branch141, branch411],
            name='mixed0')

        for i in range(1, 6):
            # 4 x 4 x filters1*20
            x = Reshape((4, 4, -1))(x)

            branch1x1 = conv2d_bn(x, 64, 1, 1, name=f'mix{i}_branch1x1')

            branch4x4 = conv2d_bn(x, 48, 1, 1, name=f'mix{i}_branch4x4_1')
            branch4x4 = conv2d_bn(branch4x4, 64, 4, 4,
                                  name=f'mix{i}_branch4x4_2')

            branch3x3dbl = conv2d_bn(
                x, 64, 1, 1, name=f'mix{i}_branch3x3dbl_1')
            branch3x3dbl = conv2d_bn(
                branch3x3dbl, 96, 3, 3, name=f'mix{i}_branch3x3dbl_2')
            branch3x3dbl = conv2d_bn(
                branch3x3dbl, 96, 3, 3, name=f'mix{i}_branch3x3dbl_3')

            branch_pool = layers.AveragePooling2D((3, 3),
                                                  strides=(1, 1),
                                                  padding='same')(x)
            branch_pool = conv2d_bn(branch_pool, 64, 1, 1, name=f'mix{i}_branch_pool')
            x = layers.concatenate(
                [branch1x1, branch4x4, branch3x3dbl, branch_pool],
                name=f'mixed{i}')

        x = Flatten()(x)
        y = Dense(2048, name='dense1_pi')(x)
        y = BatchNormalization()(y)
        y = Activation('relu')(y)

        y = Dense(1024, name='dense2_pi')(y)
        y = BatchNormalization()(y)
        y = Activation('relu')(y)

        z = Dense(2024, name='dense1_v')(x)
        z = BatchNormalization()(z)
        z = Activation('relu')(z)

        z = Dense(512, name='dense2_v')(z)
        z = BatchNormalization()(z)
        z = Activation('relu')(z)

        self.pi = Dense(self.action_size, activation='softmax', name='pi')(y)
        self.v = Dense(1, activation='tanh', name='v')(z)

        self.model = Model(inputs=self.input_boards, outputs=[self.pi, self.v])
        self.model.compile(
            loss=['categorical_crossentropy', 'mean_squared_error'], optimizer=SGD(args.lr))
