from keras import Model


def check_large_model(model):
    for layer in model.layers:
        # Check if the layer is a MultiHeadAttention layer
        if 'multi_head_attention' in str(layer):
            # Get the output shape of the layer
            output_shape = layer.output.shape
            # Get the size of the second dimension
            size = output_shape[1]
            if size > 1024:
                return True
    return False
