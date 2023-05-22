from src.Gene_Pool import conv_block, inverted_residual_block, mobilevit_block
from src.Search_Space import kernel_size_space, stride_space, filters_space, \
    expansion_factor_space, residual_space, normalization_space, activation_space, transformer_space, head_space


def decoded_block(x, layer_array):
    """
    This function defines a block in a neural network architecture, based on an input array specifying the
    characteristics of the layers within the block.

    Parameters:
    x : tensor
        Input tensor to the block.
    layer_array : np.ndarray
        An array of binary digits that encode the specifications of the block's layers.

    Returns:
    tensor
        The output tensor after applying the block to the input.
    """

    # Each index corresponds to the binary encoding of different aspects of the layer
    layer_type_index = int(str(layer_array[0]) + str(layer_array[1]), 2)
    kernel_size_index = int(str(layer_array[2]) + str(layer_array[3]), 2)
    stride_index = layer_array[4]
    filters_index = int(str(layer_array[5]) + str(layer_array[6]) + str(layer_array[7]), 2)
    expansion_factor_index = layer_array[8]
    residual_index = int(str(layer_array[9]) + str(layer_array[10]), 2)
    normalization_index = layer_array[11]
    activation_index = int(str(layer_array[12]) + str(layer_array[13]), 2)
    transformer_index = int(str(layer_array[14]) + str(layer_array[15]), 2)
    head_index = int(str(layer_array[16]) + str(layer_array[17]), 2)

    # Mapping for the index to actual layer type
    layer_type_dict = {
        # Convolutional block
        0: lambda x: conv_block(x, filters=filters_space[filters_index],
                                kernel_size=kernel_size_space[kernel_size_index],
                                strides=stride_space[stride_index],
                                normalization=normalization_space[normalization_index],
                                activation=activation_space[activation_index]),
        # Inverted residual block
        1: lambda x: inverted_residual_block(x, expansion_factor=expansion_factor_space[expansion_factor_index],
                                             kernel_size=kernel_size_space[kernel_size_index],
                                             output_channels=filters_space[filters_index],
                                             strides=stride_space[stride_index],
                                             normalization=normalization_space[normalization_index],
                                             activation=activation_space[activation_index],
                                             residual=residual_space[residual_index]),
        # MobileViT block
        2: lambda x: mobilevit_block(x, num_blocks=transformer_space[transformer_index],
                                     projection_dim=filters_space[filters_index],
                                     strides=stride_space[stride_index],
                                     normalization=normalization_space[normalization_index],
                                     kernel_size=kernel_size_space[kernel_size_index],
                                     num_heads=head_space[head_index],
                                     activation=activation_space[activation_index],
                                     residual=residual_space[residual_index]),
        # None block
        3: lambda x: x
    }
    # The function defined by layer_type_dict[layer_type_index] is applied to the input x and returned
    return layer_type_dict[layer_type_index](x)
