from src.Gene_Pool import conv_block, inverted_residual_block, mobilevit_block
from src.Search_Space import kernel_size_space, stride_space, filters_space, \
    expansion_factor_space, residual_space, normalization_space, activation_space, transformer_space, head_space

"""Function Signature:
def decoded_block(x: tf.Tensor, layer_array: np.ndarray) -> tf.Tensor

Parameters:
x: A tensor representing the input to the block of layers.
layer_array: A binary array representing the sequence of layers to be applied.

Returns:
A tensor representing the output after applying the layer operations. Description: The "decoded_block" function 
decodes a binary array representing a sequence of layers and applies the corresponding layer operations to an input 
tensor "x". The function returns the tensor "x" after applying the layer operations."""


def decoded_block(x, layer_array):
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

    layer_type_dict = {

        0: lambda x: conv_block(x, filters=filters_space[filters_index],
                                kernel_size=kernel_size_space[kernel_size_index],
                                strides=stride_space[stride_index],
                                normalization=normalization_space[normalization_index],
                                activation=activation_space[activation_index]),

        1: lambda x: inverted_residual_block(x, expansion_factor=expansion_factor_space[expansion_factor_index],
                                             kernel_size=kernel_size_space[kernel_size_index],
                                             output_channels=filters_space[filters_index],
                                             strides=stride_space[stride_index],
                                             normalization=normalization_space[normalization_index],
                                             activation=activation_space[activation_index],
                                             residual=residual_space[residual_index]),

        2: lambda x: mobilevit_block(x, num_blocks=transformer_space[transformer_index],
                                     projection_dim=filters_space[filters_index],
                                     strides=stride_space[stride_index],
                                     normalization=normalization_space[normalization_index],
                                     kernel_size=kernel_size_space[kernel_size_index],
                                     num_heads=head_space[head_index],
                                     activation=activation_space[activation_index],
                                     residual=residual_space[residual_index]),

        3: lambda x: x
    }
    return layer_type_dict[layer_type_index](x)
