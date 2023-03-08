from pycoral.utils.edgetpu import make_interpreter
import numpy as np
from PIL import Image
import time


image_file = 'test.jpg'
image = Image.open(image_file).convert('RGB')
image = np.array(image)


def inference_time_tpu(edgetpu_model_name):
    interpreter = make_interpreter(edgetpu_model_name)
    del edgetpu_model_name
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()[0]

    input_tensor = np.expand_dims(image, axis=0).astype(input_details['dtype'])
    interpreter.set_tensor(input_details['index'], input_tensor)

    del input_details
    del input_tensor

    start_time = time.monotonic()
    interpreter.invoke()
    tpu_inference_time = (time.monotonic() - start_time) * 1000

    return tpu_inference_time
