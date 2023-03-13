import tensorflow as tf
from get_datasets.Data_for_TFLITE import x_test


def representative_data_gen():
    for data in tf.data.Dataset.from_tensor_slices(x_test).batch(1).take(100):
        yield [(tf.dtypes.cast(data, tf.float32))]


"""Function Signature:
def convert_to_tflite(keras_model: keras.Model, generation: int, i: int, time: str) -> Tuple[bytes, str]

Parameters:
keras_model: A Keras model object representing the TensorFlow model to be converted to TensorFlow Lite.
generation: An integer representing the generation number of the model.
i: An integer representing the index number of the model within the generation.
time: A string representing the timestamp of the conversion.

Returns:
A tuple containing the converted TensorFlow Lite model as a byte string and the path to the saved model file. 
Description: The "convert_to_tflite" function converts the input Keras model to a TensorFlow Lite model with 
quantization enabled. The function returns a tuple containing the converted model as a byte string and the path to 
the saved model file. The saved model file name includes the generation number, index number, and timestamp.
"""


def convert_to_tflite(keras_model, generation, i, time):
    converter = tf.lite.TFLiteConverter.from_keras_model(keras_model)

    # This enables quantization
    converter.optimizations = [tf.lite.Optimize.DEFAULT]

    # This sets the representative dataset for quantization
    converter.representative_dataset = representative_data_gen

    # For full integer quantization, though supported types defaults to int8 only, we explicitly declare it for clarity.
    converter.target_spec.supported_types = [tf.int8]

    # These set the input and output tensors to uint8 (added in r2.3)
    converter.inference_input_type = tf.uint8
    converter.inference_output_type = tf.uint8

    converter.allow_custom_ops = True
    converter.experimental_new_converter = True
    converter.experimental_new_quantizer = True

    tflite_model = converter.convert()
    # tf.lite.experimental.Analyzer.analyze(model_content=tflite_model, gpu_compatibility=True)
    path = f"model_{i}_gen_{generation}_time_{time}.tflite"

    with open(path, 'wb') as f:
        f.write(tflite_model)

    del converter

    return tflite_model, path
