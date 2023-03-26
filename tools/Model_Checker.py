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

#
# def check_model(model):
#     contains_multi_head_attention = False
#     for layer in model.layers:
#         if 'multi_head_attention' in str(layer):
#             contains_multi_head_attention = True
#             break
#
#     if contains_multi_head_attention:
#         for layer in model.layers:
#             if 'multi_head_attention' in str(layer):
#                 output_shape = layer.output.shape
#                 size = output_shape[1]
#                 if size > 1024:
#                     return True
#         return False
#
#     else:
#         return True


import os
from tools.TFLITE_Converter import convert_to_tflite
from tools.Compile_Edge_TPU import compile_edgetpu


def is_edge_tpu_compatible(keras_model):
    try:
        # Convert the Keras model to a TFLite model
        _, tflite_path = convert_to_tflite(keras_model, "compatibility_check", 0, "compatibility_check")

        # Try to compile the TFLite model for the Edge TPU
        edgetpu_model_name = compile_edgetpu(tflite_path)

        # Check if the compilation was successful
        if os.path.exists(edgetpu_model_name):
            compatible = True
        else:
            compatible = False

        # Clean up the temporary files
        os.remove(tflite_path)
        if os.path.exists(edgetpu_model_name):
            os.remove(edgetpu_model_name)

        return compatible
    except Exception as e:
        print(f"Error during Edge TPU compatibility check: {e}")
        return False


def model_has_problem(model):
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
                else:
                    if not is_edge_tpu_compatible(model):
                        return True
        return False

    else:
        return True
