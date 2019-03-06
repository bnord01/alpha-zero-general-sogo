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

        self.pi = Dense(self.action_size, activation='softmax', name='pi')(x)
        self.v = Dense(1, activation='tanh', name='v')(x)

        self.model = Model(inputs=self.input_boards, outputs=[self.pi, self.v])
        self.model.compile(
            loss=['categorical_crossentropy', 'mean_squared_error'], optimizer=Adam(args.lr))
