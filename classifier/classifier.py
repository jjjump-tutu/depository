import keras
import tensorflow_addons as tfa

def build_classifier(config, net_arch):
    inputs = keras.Input(shape=config.input_shape,
                   dtype='float32',
                   name='inputs')
    layer = inputs
    if 'gap' in net_arch.keys() and net_arch['gap']:
        layer = keras.layers.GlobalAveragePooling1D()(layer)
        if 'ap' in net_arch.keys():
            net_arch['ap'] = False
        if 'conv' in net_arch.keys():
            net_arch['conv'] = []
    if 'ap' in net_arch.keys() and net_arch['ap']:
        # layer = keras.layers.AveragePooling1D(net_arch['ap'], padding='same')(layer)
        layer = tfa.layers.AdaptiveAveragePooling1D(output_size = net_arch['ap'])(layer)
    if 'conv' in net_arch.keys():
        for conv_kernel in net_arch['conv']:
            layer = keras.layers.Conv1D(conv_kernel[0], conv_kernel[1], padding='same', activation='relu')(layer)
            layer = keras.layers.BatchNormalization()(layer)
            layer = keras.layers.Dropout(net_arch['dropout'])(layer)
            layer = keras.layers.AveragePooling1D(2, padding='same')(layer)
    layer = keras.layers.Flatten()(layer)
    for fc_cfg in net_arch['fc']:
        layer = keras.layers.Dense(fc_cfg, activation='relu')(layer)
        layer = keras.layers.Dropout(net_arch['dropout'])(layer)
    output = keras.layers.Dense(config.num_categories ,activation='softmax')(layer)
    model = keras.Model(inputs=[inputs], outputs=[output])
    return model
