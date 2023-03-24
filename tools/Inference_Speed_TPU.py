import gc
from pycoral.utils.edgetpu import make_interpreter
import numpy as np
from PIL import Image
import time

image_file = 'test.jpg'
image = Image.open(image_file).convert('RGB')
image = np.array(image)

"""Function Signature:
def inference_time_tpu(edgetpu_model_name: str) -> float

Parameters:
edgetpu_model_name: A string representing the name of the TensorFlow Lite model compiled for the Edge TPU.
Returns:

A float representing the inference time of the model on the Edge TPU in milliseconds. Description: The 
"inference_time_tpu" function measures the inference time of a TensorFlow Lite model running on an Edge TPU device. 
The function returns the inference time of the model on the Edge TPU device in milliseconds."""


def inference_time_tpu(edgetpu_model_name):
    interpreter = make_interpreter(edgetpu_model_name)

    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()[0]

    input_tensor = np.expand_dims(image, axis=0).astype(input_details['dtype'])
    interpreter.set_tensor(input_details['index'], input_tensor)

    start_time = time.monotonic()
    interpreter.invoke()
    tpu_inference_time = (time.monotonic() - start_time) * 1000

    gc.collect()

    return tpu_inference_time
