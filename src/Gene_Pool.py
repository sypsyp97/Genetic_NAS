import tensorflow as tf
import tensorflow_addons as tfa

from keras import layers


def conv_block(x, filters=16, kernel_size=3, strides=2, normalization='BatchNormalization', activation='silu6'):
    x = layers.Conv2D(filters, kernel_size, strides=strides, padding="same")(x)

    try:
        normalization_layer = {'BatchNormalization': layers.BatchNormalization(epsilon=1e-6),
                               'LayerNormalization': layers.LayerNormalization(epsilon=1e-6)}[normalization]
        x = normalization_layer(x)

    except KeyError:
        print(f"{normalization} not found in the list of normalization layers.")
        x = x

    try:
        activation_function = {'relu': tf.nn.relu,
                               'relu6': tf.nn.relu6,
                               'silu': tf.nn.silu,
                               'silu6': lambda x: tf.math.minimum(tf.nn.silu(x), 6)}[activation]
        x = activation_function(x)

    except KeyError:
        print(f"{activation} not found in the list of activation functions.")
        x = x

    return x


def inverted_residual_block(x, expansion_factor, output_channels, strides=1, kernel_size=3,
                            normalization='BatchNormalization', activation='silu6', residual='Concatenate'):
    m = layers.Conv2D(expansion_factor * output_channels, 1, padding="same")(x)
    try:
        normalization_layer = {'BatchNormalization': layers.BatchNormalization(epsilon=1e-6),
                               'LayerNormalization': layers.LayerNormalization(epsilon=1e-6)}[normalization]
        m = normalization_layer(m)

    except KeyError:
        print(f"{normalization} not found in the list of normalization layers.")
        m = m

    m = layers.DepthwiseConv2D(kernel_size, strides=strides, padding="same")(m)
    try:
        normalization_layer = {'BatchNormalization': layers.BatchNormalization(epsilon=1e-6),
                               'LayerNormalization': layers.LayerNormalization(epsilon=1e-6)}[normalization]
        m = normalization_layer(m)

    except KeyError:
        print(f"{normalization} not found in the list of normalization layers.")
        m = m

    try:
        activation_function = {'relu': tf.nn.relu,
                               'relu6': tf.nn.relu6,
                               'silu': tf.nn.silu,
                               'silu6': lambda x: tf.math.minimum(tf.nn.silu(x), 6)}[activation]
        m = activation_function(m)

    except KeyError:
        print(f"{activation} not found in the list of activation functions.")
        m = m

    m = layers.Conv2D(output_channels, 1, padding="same")(m)

    try:
        normalization_layer = {'BatchNormalization': layers.BatchNormalization(epsilon=1e-6),
                               'LayerNormalization': layers.LayerNormalization(epsilon=1e-6)}[normalization]
        m = normalization_layer(m)

    except KeyError:
        print(f"{normalization} not found in the list of normalization layers.")
        m = m

    if strides == 1 and residual == 'Concatenate':
        m = layers.Concatenate(axis=-1)([m, x])

    elif tf.math.equal(x.shape[-1], output_channels) and strides == 1:

        if residual == 'Concatenate':
            m = layers.Concatenate(axis=-1)([m, x])
        elif residual == 'StochasticDepth':
            m = tfa.layers.StochasticDepth(0.5)([m, x])
        elif residual == 'Add':
            m = layers.Add()([m, x])
        else:
            m = m

    elif strides == 2 and residual == 'Concatenate':
        x = layers.Conv2D(output_channels, kernel_size=(2, 2), strides=2, padding="same")(x)
        m = layers.Concatenate(axis=-1)([m, x])

    elif tf.math.equal(x.shape[-1], output_channels) and strides == 2:
        x = layers.Conv2D(output_channels, kernel_size=(2, 2), strides=2, padding="same")(x)
        try:
            normalization_layer = {'BatchNormalization': layers.BatchNormalization(epsilon=1e-6),
                                   'LayerNormalization': layers.LayerNormalization(epsilon=1e-6)}[normalization]
            x = normalization_layer(x)

        except KeyError:
            print(f"{normalization} not found in the list of normalization layers.")
            x = x

        if residual == 'Concatenate':
            m = layers.Concatenate(axis=-1)([m, x])
        elif residual == 'StochasticDepth':
            m = tfa.layers.StochasticDepth(0.5)([m, x])
        elif residual == 'Add':
            m = layers.Add()([m, x])
        else:
            m = m

    else:
        m = m

    return m


def ffn(x, hidden_units, dropout_rate, use_bias=False):
    a = tf.reshape(x, (-1, 1, x.shape[1], x.shape[-1]))

    for hidden_unit in hidden_units:
        a = tf.keras.layers.Conv2D(filters=hidden_unit, kernel_size=1, padding='same', use_bias=use_bias)(a)
        a = layers.LayerNormalization(epsilon=1e-6)(a)
        a = tf.math.minimum(tf.nn.silu(a), 6)
        a = layers.Dropout(dropout_rate)(a)

    x = tf.reshape(a, (-1, x.shape[1], x.shape[-1]))

    return x


def transformer_block(encoded_patches, transformer_layers, projection_dim, num_heads=2):
    for i in range(transformer_layers):
        # Layer normalization 1.
        t1 = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
        # Create a multi-head attention layer.
        attention_output = layers.MultiHeadAttention(num_heads=num_heads,
                                                     key_dim=projection_dim)(t1, t1)
        # Skip connection 1.
        t2 = tfa.layers.StochasticDepth()([attention_output, encoded_patches])
        # Layer normalization 2.
        t3 = layers.LayerNormalization(epsilon=1e-6)(t2)
        t3 = ffn(t3, hidden_units=[projection_dim * 2, projection_dim], dropout_rate=0.1)

        # Skip connection 2.
        encoded_patches = tfa.layers.StochasticDepth()([t3, t2])

    encoded_patches = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)

    return encoded_patches


'''This function defines a mobilevit block, which is a type of block that combines the features of convolutional 
neural networks and transformer networks. The block applies a series of convolutional layers, normalization layers, 
transformer layers and skip connections. The input x is passed through a convolutional block to extract local 
features, then another convolutional block to further extract local features. These local features are then reshaped 
and passed through a transformer block to extract global features. The global features are then reshaped and passed 
through another convolutional block to produce folded features. Finally, the folded features are concatenated or 
added to the input x using a skip connection with the specified residual. The output is passed through another 
convolutional block with filter size of projection_dim and kernel size of 1. The function returns the output of the 
mobilevit block.'''


def mobilevit_block(x, num_blocks, projection_dim, strides=1, kernel_size=3, num_heads=2,
                    residual='Concatenate', activation='silu6', normalization='BatchNormalization'):
    local_features = conv_block(x, filters=projection_dim, kernel_size=kernel_size,
                                strides=strides, activation=activation, normalization=normalization)
    local_features = conv_block(local_features, filters=projection_dim, kernel_size=1,
                                strides=1, activation=activation, normalization=normalization)

    non_overlapping_patches = tf.reshape(local_features, (
        -1, tf.shape(local_features)[1] * tf.shape(local_features)[2], tf.shape(local_features)[-1]))

    global_features = transformer_block(non_overlapping_patches, num_blocks, projection_dim, num_heads=num_heads)

    folded_feature_map = tf.reshape(global_features, (
        -1, tf.shape(local_features)[1], tf.shape(local_features)[2], tf.shape(local_features)[-1]))

    folded_feature_map = conv_block(folded_feature_map, filters=x.shape[-1], kernel_size=kernel_size,
                                    strides=1, activation=activation, normalization=normalization)

    if strides == 1:

        if residual == 'Concatenate':
            local_global_features = layers.Concatenate(axis=-1)([folded_feature_map, x])
        elif residual == 'StochasticDepth':
            local_global_features = tfa.layers.StochasticDepth(0.5)([folded_feature_map, x])
        elif residual == 'Add':
            local_global_features = layers.Add()([folded_feature_map, x])
        else:
            local_global_features = folded_feature_map

    elif strides == 2:
        x = layers.Conv2D(x.shape[-1], kernel_size=(2, 2), strides=2, padding="same")(x)

        if residual == 'Concatenate':
            local_global_features = layers.Concatenate(axis=-1)([folded_feature_map, x])
        elif residual == 'StochasticDepth':
            local_global_features = tfa.layers.StochasticDepth(0.5)([folded_feature_map, x])
        elif residual == 'Add':
            local_global_features = layers.Add()([folded_feature_map, x])
        else:
            local_global_features = folded_feature_map

    else:
        local_global_features = folded_feature_map

    local_global_features = conv_block(local_global_features, filters=projection_dim, kernel_size=1,
                                       strides=1, activation=activation, normalization=normalization)

    return local_global_features
