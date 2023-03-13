import os

"""Overview: The "compile_edgetpu" function compiles a TensorFlow Lite (TFLite) model for Edge TPU, a type of 
specialized hardware for machine learning inference. This function takes a file path as input and returns the name of 
the compiled TFLite model for Edge TPU.

Function Signature:
def compile_edgetpu(path: str) -> Optional[str]

Parameters：
path: A string representing the path of the TFLite model to be compiled for Edge TPU.
Returns:
edgetpu_model_name: A string representing the name of the compiled TFLite model for Edge TPU. 

Returns:
The name of the compiled TFLite model for Edge TPU.
"""


def compile_edgetpu(path):
    if not os.path.exists(path):
        print(f"{path} not found")
        return None

    edgetpu_model_name = path.replace('.tflite', '_edgetpu.tflite')
    os.system('edgetpu_compiler -sa {}'.format(path))

    return edgetpu_model_name
