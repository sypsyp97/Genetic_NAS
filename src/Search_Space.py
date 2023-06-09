"""These are a set of possible values for different hyperparameters that can be used to
define different types of layers, kernel size, stride, filters, expansion factor,
residual connections, normalization, activation function, number of transformer layers,
and number of heads in a transformer block.

These values can be used to perform a grid search or random search over the
hyperparameters to find the best configuration for a given task.
"""

layer_type_space = ["conv_block", "inverted_residual_block", "mobilevit_block", "None"]
kernel_size_space = [1, 3, 5, 7]
stride_space = [1, 2]
filters_space = [16, 24, 32, 48, 64, 96, 128, 192]
expansion_factor_space = [2, 4]
residual_space = ["None", "Add", "StochasticDepth", "Concatenate"]
normalization_space = ["BatchNormalization", "LayerNormalization"]
activation_space = ["relu", "relu6", "silu", "silu6"]
transformer_space = [2, 3, 4, 5]
head_space = [2, 4, 6, 8]
