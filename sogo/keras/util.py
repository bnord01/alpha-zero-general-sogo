from keras.layers import *


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
    y = conv2d_bn(x, filters, 3, 3, name=name, with_activation=False)
    y = Add()([x, y])
    y = Activation('relu', name=name)(y)
    return y

def res_block_3d(x, filters=256, name=None):
    y = conv3d_bn(x, filters, 3, 3, 3, name=name, with_activation=False)
    y = Add()([x, y])
    y = Activation('relu', name=name)(y)
    return y