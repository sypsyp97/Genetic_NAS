import gc
from pycoral.utils.edgetpu import make_interpreter
import numpy as np
from PIL import Image
import time

image_file = 'test.jpg'
image = Image.open(image_file).convert('RGB')
image = np.array(image)


def inference_time_tpu(edgetpu_model_name):
    # Create the TPU model interpreter using the provided model name.
    interpreter = make_interpreter(edgetpu_model_name)

    # Allocate memory for tensors.
    interpreter.allocate_tensors()

    # Get the details of the input tensor.
    input_details = interpreter.get_input_details()[0]

    # Prepare the input tensor by expanding dimensions and converting to the appropriate data type.
    input_tensor = np.expand_dims(image, axis=0).astype(input_details['dtype'])

    # Set the input tensor to the interpreter.
    interpreter.set_tensor(input_details['index'], input_tensor)

    # Measure the start time for inference.
    start_time = time.monotonic()

    # Invoke the interpreter to perform inference.
    interpreter.invoke()

    # Calculate the TPU inference time by subtracting the start time from the current time and converting to milliseconds.
    tpu_inference_time = (time.monotonic() - start_time) * 1000

    # Perform garbage collection to free up memory.
    gc.collect()

    return tpu_inference_time
