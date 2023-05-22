from tools.TFLITE_Converter import convert_to_tflite
from tools.Compile_Edge_TPU import compile_edgetpu

import os


def is_edge_tpu_compatible(model):
    """
    Checks if a given Keras model is compatible with Edge TPU.

    Parameters:
    model : tensorflow.python.keras.engine.functional.Functional
        The Keras model to be checked for Edge TPU compatibility.

    Returns:
    bool
        True if the model is compatible with Edge TPU, False otherwise.
    """

    try:
        # Convert the Keras model to a TFLite model
        # Also retrieves the path of the converted model
        _, tflite_path = convert_to_tflite(model)

        # Try to compile the TFLite model for the Edge TPU
        # Retrieves the name of the Edge TPU compiled model
        edgetpu_model_name = compile_edgetpu(tflite_path)

        # Check if the Edge TPU compiled model file exists, which indicates successful compilation
        if os.path.exists(edgetpu_model_name):
            compatible = True
        else:
            compatible = False

        # Clean up the temporary files: the TFLite model and the Edge TPU compiled model (if it exists)
        os.remove(tflite_path)
        if os.path.exists(edgetpu_model_name):
            os.remove(edgetpu_model_name)

        # Return the compatibility status
        return compatible
    except Exception as e:
        # Print the error message and return False in case of any exceptions
        print(f"Error during Edge TPU compatibility check: {e}")
        return False


def model_has_attention(model):
    """
    Checks if a given model contains multi head attention layers and whether they meet certain conditions.

    Parameters:
    model : tensorflow.python.keras.engine.functional.Functional
        The Keras model to be checked for multi head attention layers.

    Returns:
    bool
        True if the model contains multi head attention layers and all these layers output shapes are less than or equal to 256,
        False otherwise.
    """
    # Initialize the flag as False
    contains_multi_head_attention = False

    # Iterate over all the layers in the model
    for layer in model.layers:
        # Check if the current layer is a multi head attention layer
        if 'multi_head_attention' in str(layer):
            # If it is, set the flag to True and break the loop
            contains_multi_head_attention = True
            break

    # If the model contains at least one multi head attention layer
    if contains_multi_head_attention:
        # Check all multi head attention layers in the model
        for layer in model.layers:
            if 'multi_head_attention' in str(layer):
                # Retrieve the output shape of the current layer
                output_shape = layer.output.shape
                # The second dimension of the output shape represents the size
                size = output_shape[1]
                # If the size is greater than 256, return False
                if size > 256:
                    return False
        # If all multi head attention layers have a size less than or equal to 256, return True
        return True

    # If the model does not contain any multi head attention layers, return False
    else:
        return False


def model_has_problem(model):
    """
    Checks if a given model contains any issues related to multi head attention layers and Edge TPU compatibility.

    Parameters:
    model : tensorflow.python.keras.engine.functional.Functional
        The Keras model to be checked.

    Returns:
    bool
        True if the model has an issue, False otherwise.
    """
    # Check if the model contains multi head attention layers and if they meet certain conditions
    if model_has_attention(model):
        # If the model contains multi head attention layers, check if it is compatible with Edge TPU
        if is_edge_tpu_compatible(model):
            # If the model is compatible with Edge TPU, it does not have any issues, return False
            return False
        else:
            # If the model is not compatible with Edge TPU, it has an issue, return True
            return True
    else:
        # If the model does not contain multi head attention layers, it has an issue, return True
        return True

