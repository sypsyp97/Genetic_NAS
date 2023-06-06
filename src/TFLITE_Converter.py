import tensorflow as tf

from get_datasets.Data_for_TFLITE import x_test


def representative_data_gen():
    """
    Generator function for the representative dataset required by the TFLite converter for quantization.

    Yields:
    ---------------
    list
        List containing a single batch of data. In this case, the batch size is 1.
    """
    for data in tf.data.Dataset.from_tensor_slices(x_test).batch(1).take(100):
        yield [(tf.dtypes.cast(data, tf.float32))]


def convert_to_tflite(keras_model, generation=0, i=0, time=0):
    """
    Convert a TensorFlow Keras model to TensorFlow Lite format and save it to a file. This function also applies
    optimization and quantization to the model during the conversion process.

    Parameters:
    -----------
    keras_model : keras.Model
        The TensorFlow Keras model to be converted.
    generation : int, optional
        The generation number of the model, used in the filename of the saved file.
    i : int, optional
        An index used in the filename of the saved file.
    time : datetime or str, optional
        A timestamp used in the filename of the saved file.

    Returns:
    --------
    tflite_model, path : tuple
        A tuple containing the converted TensorFlow Lite model and the path of the saved file.
    """
    # Create a TFLiteConverter object from the Keras model
    converter = tf.lite.TFLiteConverter.from_keras_model(keras_model)

    # Enable model optimization
    converter.optimizations = [tf.lite.Optimize.DEFAULT]

    # Set the representative dataset for quantization
    converter.representative_dataset = representative_data_gen

    # Set the target specification for full integer quantization
    converter.target_spec.supported_types = [tf.int8]

    # Set the input and output tensors to uint8
    converter.inference_input_type = tf.uint8
    converter.inference_output_type = tf.uint8

    # Allow custom operations in the model
    converter.allow_custom_ops = True

    # Use the experimental new converter and quantizer
    converter.experimental_new_converter = True
    converter.experimental_new_quantizer = True

    # Convert the Keras model to TFLite format
    tflite_model = converter.convert()

    # Define the path of the saved file
    path = f"model_{i}_gen_{generation}_time_{time}.tflite"

    # Save the TFLite model to the file
    with open(path, "wb") as f:
        f.write(tflite_model)

    # Delete the converter to free up memory
    del converter

    return tflite_model, path
