"""This function takes a model as an input and checks if the model has any MultiHeadAttention layers with an output
size greater than 1024. If it finds such a layer, it returns True, otherwise it returns False. It does this by
iterating through all the layers of the model, checking if the string 'multi_head_attention' is present in the string
representation of the layer. If it is, it gets the output shape of the layer and checks the size of the second
dimension. If it's greater than 1024, it returns True. Otherwise, it continues to check the next layer. If it doesn't
find any such layer, it returns False."""


# def check_model(model):
#     for layer in model.layers:
#         # Check if the layer is a MultiHeadAttention layer
#         if 'multi_head_attention' in str(layer):
#             # Get the output shape of the layer
#             output_shape = layer.output.shape
#             # Get the size of the second dimension
#             size = output_shape[1]
#             if size > 1024:
#                 return True
#     return False


def check_model(model):
    # Check if model contains a layer with the name 'multi_head_attention'
    contains_multi_head_attention = False
    for layer in model.layers:
        if 'multi_head_attention' in str(layer):
            contains_multi_head_attention = True
            break

    # If model contains a layer with the name 'multi_head_attention', check if output size > 1024
    if contains_multi_head_attention:
        for layer in model.layers:
            if 'multi_head_attention' in str(layer):
                output_shape = layer.output.shape
                size = output_shape[1]
                if size > 1024:
                    return True
        return False

    # If model does not contain a layer with the name 'multi_head_attention', return False
    else:
        return False
