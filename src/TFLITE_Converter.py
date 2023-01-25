import tensorflow as tf

'''The convert_to_tflite function takes in two arguments: x_test and keras_model.
x_test is a test dataset that is used to generate representative data for the quantization process.
keras_model is the model that you want to convert to TFLite format.

The function first defines a representative_data_gen() function that yields a batch of data from x_test, 
which is used to generate representative data for the quantization process.

It then creates a TFLiteConverter object from the keras_model and sets the optimizations to the default value. The 
representative dataset is set as the output of the representative_data_gen() function.

It sets the target_spec.supported_types to int8, which enables full integer quantization.

It also set the input and output tensors to uint8, which is necessary for full integer quantization.

The allow_custom_ops, experimental_new_converter, and experimental_new_quantizer are set to true to enable some new 
features.

Then it calls the convert() method on the converter object which returns the TFLite model in bytes.

The tflite_model is returned by the function, which can be written to a file or used in other ways as needed.'''


def convert_to_tflite(x_test, keras_model):
    def representative_data_gen():
        for data in tf.data.Dataset.from_tensor_slices(x_test).batch(1).take(100):
            yield [(tf.dtypes.cast(data, tf.float32))]

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

    # with open('test.tflite', 'wb') as f:
    #     f.write(tflite_model)

    return tflite_model
