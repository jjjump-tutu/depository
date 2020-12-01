# -*- coding: utf-8 -*-
"""
Created on Wed Mar 13 10:21:54 2019

@author: Winham

网络搭建

"""

from keras import backend as K
import tensorflow as tf
from keras.models import Model
from keras.layers import Input
from keras.layers.core import Dense, Activation
from keras.layers import GlobalAveragePooling1D
import tensorflow_addons as tfa
import keras

def _bn_relu(layer, config, dropout=0, name=None):
    from keras.layers import BatchNormalization
    from keras.layers import Activation
    layer = BatchNormalization()(layer)
    layer = Activation(config.conv_activation, name=name)(layer)

    if dropout > 0:
        from keras.layers import Dropout
        layer = Dropout(config.conv_dropout)(layer)

    return layer


def add_conv_weight(
        layer,
        filter_length,
        num_filters,
        config,
        subsample_length=1):
    from keras.layers import Conv1D
    from keras import regularizers
    from keras.layers import Add
    layer1 = Conv1D(
        filters=num_filters,
        kernel_size=filter_length,
        strides=subsample_length,
        padding='same',
        kernel_initializer=config.conv_init,
        kernel_regularizer=regularizers.l2(0.001))(layer)
    # layer2 = Conv1D(
    #     filters=num_filters,
    #     kernel_size=int(filter_length * 2),
    #     strides=subsample_length,
    #     padding='same',
    #     kernel_initializer=config.conv_init,
    #     kernel_regularizer=regularizers.l2(0.001))(layer)
    # layer3 = Conv1D(
    #     filters=num_filters,
    #     kernel_size=int(filter_length * 3),
    #     strides=subsample_length,
    #     padding='same',
    #     kernel_initializer=config.conv_init,
    #     kernel_regularizer=regularizers.l2(0.001))(layer)
    # layer = Add()([layer1, layer2, layer3])
    layer = layer1
    return layer


def add_conv_layers(layer, config):
    for subsample_length in config.conv_subsample_lengths:
        layer = add_conv_weight(
                    layer,
                    config.conv_filter_length,
                    config.conv_num_filters_start,
                    config,
                    subsample_length=subsample_length)
        layer = _bn_relu(layer, config)
    return layer


def resnet_block(
        layer,
        num_filters,
        subsample_length,
        block_index,
        config):
    from keras.layers import Add
    from keras.layers import MaxPooling1D
    from keras.layers.core import Lambda

    def zeropad(x):
        y = K.zeros_like(x)
        return K.concatenate([x, y], axis=2)

    def zeropad_output_shape(input_shape):
        shape = list(input_shape)
        assert len(shape) == 3
        shape[2] *= 2
        return tuple(shape)

    shortcut = MaxPooling1D(pool_size=subsample_length)(layer)
    zero_pad = (block_index % config.conv_increase_channels_at) == 0 \
        and block_index > 0
    if zero_pad is True:
        shortcut = Lambda(zeropad, output_shape=zeropad_output_shape)(shortcut)

    for i in range(config.conv_num_skip):
        if not (block_index == 0 and i == 0):
            layer = _bn_relu(
                layer,
                config,
                dropout=config.conv_dropout if i > 0 else 0
                )
        layer = add_conv_weight(
            layer,
            config.conv_filter_length,
            num_filters,
            config,
            subsample_length if i == 0 else 1)
    layer = Add()([shortcut, layer])
    return layer


def get_num_filters_at_index(index, num_start_filters, config):
    return 2**int(index / config.conv_increase_channels_at) \
        * num_start_filters


def add_resnet_layers(layer, config):
    layer = add_conv_weight(
        layer,
        config.conv_filter_length,
        config.conv_num_filters_start,
        config,
        subsample_length=1)
    layer = _bn_relu(layer, config)
    for index, subsample_length in enumerate(config.conv_subsample_lengths):
        num_filters = get_num_filters_at_index(
            index, config.conv_num_filters_start, config)
        layer = resnet_block(
            layer,
            num_filters,
            subsample_length,
            index,
            config)
    layer = _bn_relu(layer, config)
    return layer


def add_output_layer(layer, config):
    # layer = GlobalAveragePooling1D()(layer)
    # layer = tf.keras.layers.AveragePooling1D(pool_size=32)(layer)
    # layer = tf.keras.layers.Reshape((576,))(layer)
    layer = tfa.layers.AdaptiveAveragePooling1D(output_size=4)(layer)
    from keras.layers import Dropout
    layer = Dropout(0.3)(layer)
    layer = keras.layers.Flatten()(layer)
    layer = Dense(192)(layer)
    layer = Dropout(0.3)(layer)
    layer = Dense(96)(layer)
    layer = Dense(config.num_categories)(layer)
    return Activation('softmax')(layer)


def add_compile(model, config):
    from keras.optimizers import SGD
    optimizer = SGD(lr=config.lr_schedule(0), momentum=0.9)
    model.compile(loss=['categorical_crossentropy'],
                  optimizer=optimizer,
                  metrics=['categorical_accuracy'])


def build_network(config):
    inputs = Input(shape=config.input_shape,
                   dtype='float32',
                   name='inputs')
    # inputs = tf.keras.layers.UpSampling1D(
    # size=2)(inputs)
    layer = add_resnet_layers(inputs, config)
    output = add_output_layer(layer, config)
    model = Model(inputs=[inputs], outputs=[output])
    add_compile(model, config)
    return model

if __name__ == '__main__':
    from Config import Config
    config = Config()
    model = build_network(config)
    model.summary()
