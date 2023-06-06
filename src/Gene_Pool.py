import tensorflow as tf
import tensorflow_addons as tfa
from keras import layers


def conv_block(
    x,
    filters=16,
    kernel_size=3,
    strides=2,
    normalization="BatchNormalization",
    activation="silu6",
):
    """Defines a convolutional block with a given layer of inputs `x`.

    Parameters
    ----------
    x : tf.Tensor
        Input tensor for convolutional block.
    filters : int, optional
        The dimensionality of the output space for the Conv2D layer.
    kernel_size : int, optional
        The height and width of the 2D convolution window.
    strides : int, optional
        The strides of the convolution along the height and width.
    normalization : str, optional
        The type of normalization layer. Supports 'BatchNormalization' and 'LayerNormalization'.
    activation : str, optional
        The activation function to use. Supports 'relu', 'relu6', 'silu', and 'silu6'.

    Returns
    -------
    x : tf.Tensor
        Output tensor after applying convolution, normalization, and activation.
    """

    # Apply Conv2D layer
    x = layers.Conv2D(filters, kernel_size, strides=strides, padding="same")(x)

    # Apply normalization
    try:
        normalization_layer = {
            "BatchNormalization": layers.BatchNormalization(epsilon=1e-6),
            "LayerNormalization": layers.LayerNormalization(epsilon=1e-6),
        }[normalization]
        x = normalization_layer(x)
    except KeyError:
        print(f"{normalization} not found in the list of normalization layers.")
        x = x

    # Apply activation function
    try:
        activation_function = {
            "relu": tf.nn.relu,
            "relu6": tf.nn.relu6,
            "silu": tf.nn.silu,
            "silu6": lambda x: tf.math.minimum(tf.nn.silu(x), 6),
        }[activation]
        x = activation_function(x)
    except KeyError:
        print(f"{activation} not found in the list of activation functions.")
        x = x

    return x


def inverted_residual_block(
    x,
    expansion_factor,
    output_channels,
    strides=1,
    kernel_size=3,
    normalization="BatchNormalization",
    activation="silu6",
    residual="Concatenate",
):
    """Defines an inverted residual block with a given layer of inputs `x`.

    Parameters
    ----------
    x : tf.Tensor
        Input tensor for the inverted residual block.
    expansion_factor : int
        Determines the number of output channels for the first Conv2D layer in the block.
    output_channels : int
        The number of output channels for the last Conv2D layer in the block.
    strides : int, optional
        The strides of the convolution along the height and width.
    kernel_size : int, optional
        The height and width of the 2D convolution window.
    normalization : str, optional
        The type of normalization layer. Supports 'BatchNormalization' and 'LayerNormalization'.
    activation : str, optional
        The activation function to use. Supports 'relu', 'relu6', 'silu', and 'silu6'.
    residual : str, optional
        The type of residual connection to use. Supports 'Concatenate', 'StochasticDepth', and 'Add'.

    Returns
    -------
    m : tf.Tensor
        Output tensor after applying the inverted residual block operations.
    """

    # Apply 1x1 Conv2D layer for channel expansion
    m = layers.Conv2D(expansion_factor * output_channels, 1, padding="same")(x)

    # Apply normalization
    try:
        normalization_layer = {
            "BatchNormalization": layers.BatchNormalization(epsilon=1e-6),
            "LayerNormalization": layers.LayerNormalization(epsilon=1e-6),
        }[normalization]
        m = normalization_layer(m)
    except KeyError:
        print(f"{normalization} not found in the list of normalization layers.")
        m = m

    # Apply depthwise convolution
    m = layers.DepthwiseConv2D(kernel_size, strides=strides, padding="same")(m)

    # Apply normalization again
    try:
        normalization_layer = {
            "BatchNormalization": layers.BatchNormalization(epsilon=1e-6),
            "LayerNormalization": layers.LayerNormalization(epsilon=1e-6),
        }[normalization]
        m = normalization_layer(m)
    except KeyError:
        print(f"{normalization} not found in the list of normalization layers.")
        m = m

    # Apply activation function
    try:
        activation_function = {
            "relu": tf.nn.relu,
            "relu6": tf.nn.relu6,
            "silu": tf.nn.silu,
            "silu6": lambda x: tf.math.minimum(tf.nn.silu(x), 6),
        }[activation]
        m = activation_function(m)
    except KeyError:
        print(f"{activation} not found in the list of activation functions.")
        m = m

    # Apply 1x1 Conv2D layer for channel reduction
    m = layers.Conv2D(output_channels, 1, padding="same")(m)

    # Apply normalization
    try:
        normalization_layer = {
            "BatchNormalization": layers.BatchNormalization(epsilon=1e-6),
            "LayerNormalization": layers.LayerNormalization(epsilon=1e-6),
        }[normalization]
        m = normalization_layer(m)
    except KeyError:
        print(f"{normalization} not found in the list of normalization layers.")
        m = m

    # Apply the appropriate residual connection based on the provided parameters The specific residual connection
    # used will depend on the strides, residual connection type, and input/output channel dimensions The
    # 'Concatenate' option concatenates the output of the block with the original input, 'StochasticDepth' randomly
    # drops some of the residual paths during training, and 'Add' performs an element-wise addition of the output and
    # input.

    # If strides are equal to 1 and residual connection type is 'Concatenate'
    if strides == 1 and residual == "Concatenate":
        m = layers.Concatenate(axis=-1)([m, x])
    # If input and output channels are equal and strides are equal to 1
    elif tf.math.equal(x.shape[-1], output_channels) and strides == 1:
        if residual == "Concatenate":
            m = layers.Concatenate(axis=-1)([m, x])
        elif residual == "StochasticDepth":
            m = tfa.layers.StochasticDepth(0.5)([m, x])
        elif residual == "Add":
            m = layers.Add()([m, x])
        else:
            m = m
    # If strides are equal to 2 and residual connection type is 'Concatenate'
    elif strides == 2 and residual == "Concatenate":
        x = layers.Conv2D(
            output_channels, kernel_size=(2, 2), strides=2, padding="same"
        )(x)
        m = layers.Concatenate(axis=-1)([m, x])
    # If input and output channels are equal and strides are equal to 2
    elif tf.math.equal(x.shape[-1], output_channels) and strides == 2:
        x = layers.Conv2D(
            output_channels, kernel_size=(2, 2), strides=2, padding="same"
        )(x)
        try:
            normalization_layer = {
                "BatchNormalization": layers.BatchNormalization(epsilon=1e-6),
                "LayerNormalization": layers.LayerNormalization(epsilon=1e-6),
            }[normalization]
            x = normalization_layer(x)
        except KeyError:
            print(f"{normalization} not found in the list of normalization layers.")
            x = x
        if residual == "Concatenate":
            m = layers.Concatenate(axis=-1)([m, x])
        elif residual == "StochasticDepth":
            m = tfa.layers.StochasticDepth(0.5)([m, x])
        elif residual == "Add":
            m = layers.Add()([m, x])
        else:
            m = m
    else:
        m = m

    # Returns the output tensor after applying the inverted residual block operations.
    return m


def ffn(x, hidden_units, dropout_rate, use_bias=False):
    """
    Implements a Feed-Forward Network (FFN), which is an essential component
    of various deep learning architectures.

    Parameters
    ----------
    x : tensor
        The input tensor to the FFN.
    hidden_units : list
        A list containing the number of hidden units for each dense layer in the FFN.
    dropout_rate : float
        The dropout rate used by the dropout layers in the FFN.
    use_bias : bool, default=False
        If True, the layers in the FFN will use bias vectors.

    Returns
    -------
    x : tensor
        The output tensor from the FFN.

    """

    # Reshape the input tensor to (-1, 1, x.shape[1], x.shape[-1]) so that it can
    # be passed into the Conv2D layer
    a = tf.reshape(x, (-1, 1, x.shape[1], x.shape[-1]))

    # For each number of hidden units, add a new dense layer to the FFN
    for hidden_unit in hidden_units:
        # Apply Conv2D operation, the number of filters is defined by the current hidden unit
        # the kernel size is 1, and padding is 'same'
        a = tf.keras.layers.Conv2D(
            filters=hidden_unit, kernel_size=1, padding="same", use_bias=use_bias
        )(a)

        # Apply layer normalization to standardize the inputs (layer outputs)
        a = layers.LayerNormalization(epsilon=1e-6)(a)

        # Apply activation function (SiLU - Sigmoid Linear Unit) with a cap at 6 for the output
        a = tf.math.minimum(tf.nn.silu(a), 6)

        # Apply dropout for regularization (prevent overfitting)
        a = layers.Dropout(dropout_rate)(a)

    # Reshape the output tensor back to its original shape
    x = tf.reshape(a, (-1, x.shape[1], x.shape[-1]))

    return x


def transformer_block(encoded_patches, transformer_layers, projection_dim, num_heads=2):
    """
    Creates a Transformer block, which contains multiple layers of multi-head
    self-attention followed by a feed-forward network (FFN). Each of these
    operations is followed by a stochastic depth skip connection, and layer normalization.

    Parameters
    ----------
    encoded_patches : tensor
        The input tensor to the Transformer block.
    transformer_layers : int
        The number of layers in the Transformer block.
    projection_dim : int
        The number of output dimensions for the Transformer block.
    num_heads : int, default=2
        The number of attention heads for each self-attention layer in the Transformer block.

    Returns
    -------
    encoded_patches : tensor
        The output tensor from the Transformer block.
    """
    for i in range(transformer_layers):
        # First Layer Normalization operation
        t1 = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)

        # Create a multi-head self-attention layer
        attention_output = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=projection_dim
        )(t1, t1)

        # Skip connection 1 - Stochastic Depth is a type of regularization technique
        # that randomly drops entire layers from the neural network during training.
        t2 = tfa.layers.StochasticDepth()([attention_output, encoded_patches])

        # Second Layer Normalization operation
        t3 = layers.LayerNormalization(epsilon=1e-6)(t2)

        # Feed Forward Network (FFN) layer
        t3 = ffn(
            t3, hidden_units=[projection_dim * 2, projection_dim], dropout_rate=0.1
        )

        # Skip connection 2
        encoded_patches = tfa.layers.StochasticDepth()([t3, t2])

    # Final Layer Normalization operation
    encoded_patches = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)

    return encoded_patches


def mobilevit_block(
    x,
    num_blocks,
    projection_dim,
    strides=1,
    kernel_size=3,
    num_heads=2,
    residual="Concatenate",
    activation="silu6",
    normalization="BatchNormalization",
):
    """
    Constructs a MobileViT block which consists of local feature extraction,
    global feature extraction (via transformer block), and merging of local and global features.

    Parameters
    ----------
    x : tensor
        Input tensor.
    num_blocks : int
        Number of transformer layers to use in the transformer block.
    projection_dim : int
        Output dimensions for the Convolution and Transformer blocks.
    strides : int, default=1
        Stride length for the Convolution blocks.
    kernel_size : int, default=3
        Kernel size for the Convolution blocks.
    num_heads : int, default=2
        Number of attention heads for the MultiHeadAttention layer in the Transformer block.
    residual : str, default='Concatenate'
        Type of residual connection. Options are 'Concatenate', 'StochasticDepth', and 'Add'.
    activation : str, default='silu6'
        Activation function to use in the Convolution blocks.
    normalization : str, default='BatchNormalization'
        Normalization layer to use in the Convolution blocks.

    Returns
    -------
    local_global_features : tensor
        Output tensor after the MobileViT block.
    """
    # Local feature extraction via Convolutional blocks
    local_features = conv_block(
        x,
        filters=projection_dim,
        kernel_size=kernel_size,
        strides=strides,
        activation=activation,
        normalization=normalization,
    )
    local_features = conv_block(
        local_features,
        filters=projection_dim,
        kernel_size=1,
        strides=1,
        activation=activation,
        normalization=normalization,
    )

    # Reshaping local features into non-overlapping patches for the Transformer block
    non_overlapping_patches = tf.reshape(
        local_features,
        (
            -1,
            tf.shape(local_features)[1] * tf.shape(local_features)[2],
            tf.shape(local_features)[-1],
        ),
    )

    # Global feature extraction via Transformer block
    global_features = transformer_block(
        non_overlapping_patches, num_blocks, projection_dim, num_heads=num_heads
    )

    # Reshape the global features to match the dimensions of the local features
    folded_feature_map = tf.reshape(
        global_features,
        (
            -1,
            tf.shape(local_features)[1],
            tf.shape(local_features)[2],
            tf.shape(local_features)[-1],
        ),
    )

    # Apply another Convolutional block to the folded_feature_map
    folded_feature_map = conv_block(
        folded_feature_map,
        filters=x.shape[-1],
        kernel_size=kernel_size,
        strides=1,
        activation=activation,
        normalization=normalization,
    )

    # Merge local and global features depending on the stride and residual type
    if strides == 1:
        if residual == "Concatenate":
            local_global_features = layers.Concatenate(axis=-1)([folded_feature_map, x])
        elif residual == "StochasticDepth":
            local_global_features = tfa.layers.StochasticDepth(0.5)(
                [folded_feature_map, x]
            )
        elif residual == "Add":
            local_global_features = layers.Add()([folded_feature_map, x])
        else:
            local_global_features = folded_feature_map

    elif strides == 2:
        # Apply 2D convolution on x if stride is 2
        x = layers.Conv2D(x.shape[-1], kernel_size=(2, 2), strides=2, padding="same")(x)

        if residual == "Concatenate":
            local_global_features = layers.Concatenate(axis=-1)([folded_feature_map, x])
        elif residual == "StochasticDepth":
            local_global_features = tfa.layers.StochasticDepth(0.5)(
                [folded_feature_map, x]
            )
        elif residual == "Add":
            local_global_features = layers.Add()([folded_feature_map, x])
        else:
            local_global_features = folded_feature_map

    else:
        local_global_features = folded_feature_map

    # Pass the merged features through a final Convolution block
    local_global_features = conv_block(
        local_global_features,
        filters=projection_dim,
        kernel_size=1,
        strides=1,
        activation=activation,
        normalization=normalization,
    )

    return local_global_features
