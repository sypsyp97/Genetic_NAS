import numpy as np
import tensorflow as tf



def model_evaluation(trained_model, test_ds):
    _, raw_model_accuracy = trained_model.evaluate(test_ds)

    return raw_model_accuracy


def evaluate_tflite_model(tflite_model, x_test, y_test, tfl_int8=True):
    interpreter = tf.lite.Interpreter(model_content=tflite_model)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    input_index = input_details[0]["index"]
    output_index = output_details[0]["index"]
    scale_in, zero_point_in = input_details[0]['quantization']
    scale_out, zero_point_out = output_details[0]['quantization']

    prediction_labels = []
    test_labels = []

    for i in range(x_test.shape[0]):
        if tfl_int8:
            test_image = x_test[i] / scale_in + zero_point_in
            test_image = np.expand_dims(test_image, axis=0).astype(np.uint8)
        else:
            test_image = np.expand_dims(x_test[i], axis=0).astype(np.float32)

        interpreter.set_tensor(input_index, test_image)
        interpreter.invoke()

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
    print("Quant TF Lite accuracy: {:.3%}".format(tflite_accuracy.result()))

    return float(tflite_accuracy.result())

