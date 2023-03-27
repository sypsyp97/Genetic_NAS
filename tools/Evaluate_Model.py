from get_datasets.Data_for_TFLITE import x_test, y_test
import time
import numpy as np
import tensorflow as tf
import tflite_runtime.interpreter as tflite
import usb.core
import usb.util

VENDOR_ID = 0x1a6e  # Global Unichip Corp.
PRODUCT_ID = 0x089a  # Edge TPU USB Accelerator


def find_device(vendor_id, product_id):
    return usb.core.find(idVendor=vendor_id, idProduct=product_id)


def disconnect_device(device):
    if device is not None:
        usb.util.dispose_resources(device)
        print("Device disconnected")


def reconnect_device(vendor_id, product_id):
    device = find_device(vendor_id, product_id)
    if device is not None:
        return device
    print("Device not found")


def handle_error():
    device = find_device(VENDOR_ID, PRODUCT_ID)
    if device is not None:
        disconnect_device(device)
        time.sleep(1)  # Adjust this delay if necessary
        device = reconnect_device(VENDOR_ID, PRODUCT_ID)
        if device is not None:
            print("Device reconnected")
    else:
        print("Device not found")


def model_evaluation(trained_model, test_ds):
    _, raw_model_accuracy = trained_model.evaluate(test_ds)

    return raw_model_accuracy


"""Function Signature:
def evaluate_tflite_model(tflite_model: bytes, tfl_int8: bool = True) -> float

Parameters:
tflite_model: TensorFlow Lite model object to be evaluated.
tfl_int8: A boolean value indicating whether the model was quantized to int8 or not. Defaults to True.

Returns: A float representing the accuracy of the TensorFlow Lite model. 

Description: The "evaluate_tflite_model" 
function evaluates a TensorFlow Lite model by comparing its predictions to the ground truth labels of a test dataset. 
The function returns the accuracy of the model.

"""


def evaluate_tflite_model(tflite_model, tfl_int8=True):
    try:
        interpreter = tflite.Interpreter(model_path=tflite_model,
                                         experimental_delegates=[tflite.load_delegate('libedgetpu.so.1')])
        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        input_index = input_details[0]["index"]
        output_index = output_details[0]["index"]
        scale_in, zero_point_in = input_details[0]['quantization']
        scale_out, zero_point_out = output_details[0]['quantization']

        prediction_labels = []
        test_labels = []
        inference_speeds = []

        for i in range(x_test.shape[0]):
            if tfl_int8:
                test_image = x_test[i] / scale_in + zero_point_in
                test_image = np.expand_dims(test_image, axis=0).astype(np.uint8)
            else:
                test_image = np.expand_dims(x_test[i], axis=0).astype(np.float32)

            interpreter.set_tensor(input_index, test_image)
            start_time = time.monotonic()
            interpreter.invoke()
            tpu_inference_time = (time.monotonic() - start_time) * 1000
            inference_speeds.append(tpu_inference_time)

            output = interpreter.get_tensor(output_index)
            if tfl_int8:
                output = output.astype(np.float32)
                output = (output - zero_point_out) * scale_out
            digit = np.argmax(output[0])
            prediction_labels.append(digit)
            test_labels.append(np.argmax(y_test[i], ))

        prediction_labels = np.array(prediction_labels)
        test_labels = np.array(test_labels)
        tflite_accuracy = tf.keras.metrics.Accuracy()
        tflite_accuracy(prediction_labels, test_labels)

        del interpreter
        del input_details
        del output_details

    except Exception as e:
        tflite_accuracy = tf.keras.metrics.Accuracy()
        inference_speeds = 9999
        print(e)
        handle_error()

    return float(tflite_accuracy.result()), np.average(inference_speeds)
