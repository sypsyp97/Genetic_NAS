import os

"""Overview: The "compile_edgetpu" function compiles a TensorFlow Lite (TFLite) model for Edge TPU, a type of 
specialized hardware for machine learning inference. This function takes a file path as input and returns the name of 
the compiled TFLite model for Edge TPU.

Function Signature:
def compile_edgetpu(path: str) -> Optional[str]

Parameters:

path: A string representing the path of the TFLite model to be compiled for Edge TPU.
Returns:

edgetpu_model_name: A string representing the name of the compiled TFLite model for Edge TPU. Example Usage: Suppose 
we have a TFLite model saved at "/home/user/mymodel.tflite" and we want to compile it for Edge TPU. We can call the 
"compile_edgetpu" function as follows:

edgetpu_model = compile_edgetpu("/home/user/mymodel.tflite")

If the compilation is successful, the "edgetpu_model" variable will contain the name of the compiled TFLite model (
i.e., "/home/user/mymodel_edgetpu.tflite").

Description: The "compile_edgetpu" function first checks if the TFLite model file exists at the given path using the 
"os.path.exists" function. If the file does not exist, an error message is printed and the function returns None.

If the file exists, the function creates the name for the compiled TFLite model for Edge TPU by replacing the 
".tflite" extension in the input path with "_edgetpu.tflite".

Next, the function runs the "edgetpu_compiler" command-line tool using the "os.system" function to compile the TFLite 
model for Edge TPU. The "-sa" option is used to enable quantization-aware training.

Finally, the function returns the name of the compiled TFLite model for Edge TPU.

Dependencies:

os: This module provides a way of using operating system dependent functionality.
Note:

The "edgetpu_compiler" command-line tool must be installed on the system for this function to work.
The compiled TFLite model for Edge TPU can be executed on Edge TPU devices such as the Coral Dev Board or Coral USB Accelerator.

"""


def compile_edgetpu(path):
    if not os.path.exists(path):
        print(f"{path} not found")
        return None

    edgetpu_model_name = path.replace('.tflite', '_edgetpu.tflite')
    os.system('edgetpu_compiler -sa {}'.format(path))

    return edgetpu_model_name
