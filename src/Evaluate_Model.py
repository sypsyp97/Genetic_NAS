import time

import numpy as np
import tensorflow as tf
import tflite_runtime.interpreter as tflite

from get_datasets.Data_for_TFLITE import x_test, y_test


def model_evaluation(trained_model, test_ds):
    """Evaluate the model performance.

    Parameters
    ----------
    trained_model : keras.Model
        The trained model to be evaluated.
    test_ds : tensorflow.data.Dataset
        Test dataset.

    Returns
    -------
    raw_model_accuracy : float
        Model accuracy on the test dataset.
    """
    _, raw_model_accuracy = trained_model.evaluate(test_ds)
    return raw_model_accuracy


def evaluate_tflite_model(tflite_model, tfl_int8=True):
    """Evaluate the performance of a TFLite model.

    Parameters
    ----------
    tflite_model : str
        Path to the TFLite model.
    tfl_int8 : bool, optional
        If True, use int8 quantized TFLite model. If False, use float32 TFLite model.

    Returns
    -------
    tflite_accuracy : float
        Accuracy of the TFLite model on the test dataset.
    inference_speed : float
        Average inference speed.
    """
    try:
        # Initialize the TFLite interpreter with Edge TPU delegate.
        interpreter = tflite.Interpreter(
            model_path=tflite_model,
            experimental_delegates=[tflite.load_delegate("libedgetpu.so.1")],
        )

        # Allocate tensors for the TFLite interpreter.
        interpreter.allocate_tensors()

        # Get the input and output tensor details from the interpreter.
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        input_index = input_details[0]["index"]
        output_index = output_details[0]["index"]

        # Get the scale and zero point for quantization from the input and output tensor details.
        scale_in, zero_point_in = input_details[0]["quantization"]
        scale_out, zero_point_out = output_details[0]["quantization"]

        # Initialize lists to store prediction labels, test labels and inference speeds.
        prediction_labels = []
        test_labels = []
        inference_speeds = []

        # Loop through all the test samples.
        for i in range(x_test.shape[0]):
            # Prepare the test image for inference depending on whether the TFLite model is quantized or not.
            if tfl_int8:
                # If the model is quantized, dequantize the image.
                test_image = x_test[i] / scale_in + zero_point_in
                test_image = np.expand_dims(test_image, axis=0).astype(np.uint8)
            else:
                test_image = np.expand_dims(x_test[i], axis=0).astype(np.float32)

            # Set the input tensor for the interpreter.
            interpreter.set_tensor(input_index, test_image)

            # Record the start time, invoke the interpreter and calculate the inference time.
            start_time = time.monotonic()
            interpreter.invoke()
            tpu_inference_time = (time.monotonic() - start_time) * 1000
            inference_speeds.append(tpu_inference_time)

            # Get the output tensor from the interpreter.
            output = interpreter.get_tensor(output_index)

            # If the model is quantized, dequantize the output.
            if tfl_int8:
                output = output.astype(np.float32)
                output = (output - zero_point_out) * scale_out

            # Get the digit with the highest probability.
            digit = np.argmax(output[0])
            prediction_labels.append(digit)

            # Get the true label.
            test_labels.append(
                np.argmax(
                    y_test[i],
                )
            )

        # Convert prediction and test labels to NumPy arrays.
        prediction_labels = np.array(prediction_labels)
        test_labels = np.array(test_labels)

        # Initialize the accuracy metric.
        tflite_accuracy = tf.keras.metrics.Accuracy()

        # Compute the accuracy of the TFLite model.
        tflite_accuracy(prediction_labels, test_labels)

        # Clean up resources.
        del interpreter
        del input_details
        del output_details

    except Exception as e:
        # If an exception occurs during the evaluation, initialize the accuracy to a default metric,
        # set the average inference speed to a very high value, and print the exception.
        tflite_accuracy = tf.keras.metrics.Accuracy()
        tflite_accuracy.update_state([[1], [2], [3], [4]], [[5], [6], [7], [8]])
        inference_speeds = 9999
        print(e)

    # Return the accuracy and the average inference speed.
    return float(tflite_accuracy.result()), np.average(inference_speeds)
