"""
Function Signature:
def check_model(model: keras.Model) -> bool

Parameters:
model: A Keras model object to be checked.

Returns:
A boolean value, True if the model does not contain a multi-head attention layer or if the output shape of the layer
is less than or equal to 1024, False otherwise. Description: The "check_model" function checks if a Keras model
contains a layer of type "multi_head_attention" and if the output shape of the layer is greater than 1024. The
function returns True if the model does not contain a multi-head attention layer or if the output shape of the layer
is less than or equal to 1024, otherwise it returns False."""


def check_model(model):
    contains_multi_head_attention = False
    for layer in model.layers:
        if 'multi_head_attention' in str(layer):
            contains_multi_head_attention = True
            break

    if contains_multi_head_attention:
        for layer in model.layers:
            if 'multi_head_attention' in str(layer):
                output_shape = layer.output.shape
                size = output_shape[1]
                if size > 1024:
                    return True
        return False

    else:
        return True
