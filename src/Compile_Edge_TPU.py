import os


def compile_edgetpu(path):
    """Compiles a TensorFlow Lite model for the Edge TPU (Tensor Processing Unit).

    Parameters
    ----------
    path: str
        The path to the TensorFlow Lite model file that needs to be compiled for Edge TPU.

    Returns
    -------
    edgetpu_model_name : str or None
        The filename of the compiled Edge TPU model, or None if the initial TensorFlow Lite model file was not found.
    """

    # Check if the specified path exists, print an error message and return None if it does not
    if not os.path.exists(path):
        print(f"{path} not found")
        return None

    # Determine the filename of the Edge TPU model by replacing '.tflite' with '_edgetpu.tflite' in the original
    # filename
    edgetpu_model_name = path.replace(".tflite", "_edgetpu.tflite")

    # Compile the TensorFlow Lite model for Edge TPU using the edgetpu_compiler tool
    os.system("edgetpu_compiler -sa {}".format(path))

    return edgetpu_model_name
