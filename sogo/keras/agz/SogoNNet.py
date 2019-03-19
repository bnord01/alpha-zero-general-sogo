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
              name=None,
              with_activation=True):
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
    if with_activation:
        x = Activation('relu', name=name)(x)
    return x


def conv3d_bn(x,
              filters,
              num_row,
              num_col,
              num_lvl,
              padding='same',
              strides=(1, 1, 1),
              name=None,
              with_activation=True):
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
    if with_activation:
        x = Activation('relu', name=name)(x)
    return x

def res_block_2d(x, filters=256, name=None):
    y = conv2d_bn(x,filters, 3, 3, name=name, with_activation=False)    
    y = Add()([x,y])
    y = Activation('relu', name=name)(y)
    return y

def res_block_3d(x, filters=256, name=None):
    y = conv3d_bn(x,filters, 3, 3, 3, name=name, with_activation=False)    
    y = Add()([x,y])
    y = Activation('relu', name=name)(y)
    return y

class SogoNNet():

    def __init__(self, game, args):
        # game params
        self.board_x, self.board_y, self.board_z = game.board_size()
        self.action_size = game.action_size()
        self.args = args

        self.input_boards = Input(
            shape=(self.board_x, self.board_y, self.board_z, 2))
        x = self.input_boards

        # args
        filters3d = 128
        num3d = 10
        filters2d = 256
        num2d = 20
        v_filts = 256

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
        for i in range(num3d):
            x = res_block_3d(x, filters3d, name=f'res2d{i}')
        
        # pi head
        y = conv2d_bn(x, 2, 1, 1, name='pi_conv')
        y = Flatten()(y)
        
        # v head
        z = conv2d_bn(x, 1, 1, 1, name='v_conv')
        z = Flatten()(z)
        z = Dense(v_filts, name='dense_v')(x)        
        z = Activation('relu')(z)        

        self.pi = Dense(self.action_size, name='pi')(y)
        self.v = Dense(1, activation='tanh', name='v')(z)

        self.model = Model(inputs=self.input_boards, outputs=[self.pi, self.v])
        self.model.compile(
            loss=['categorical_crossentropy', 'mean_squared_error'], optimizer=Adam(args.lr))
