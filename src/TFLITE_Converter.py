import tensorflow as tf


def convert_to_tflite(x_test, model):
    def representative_data_gen():
        for data in tf.data.Dataset.from_tensor_slices(x_test).batch(1).take(100):
            yield [(tf.dtypes.cast(data, tf.float32))]

    converter = tf.lite.TFLiteConverter.from_keras_model(model)

    # This enables quantization
    converter.optimizations = [tf.lite.Optimize.DEFAULT]

    # This sets the representative dataset for quantization
    converter.representative_dataset = representative_data_gen

    # This ensures that if any ops can't be quantized, the converter throws an error

    # converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8,
    #                                        tf.lite.OpsSet.SELECT_TF_OPS]

    # For full integer quantization, though supported types defaults to int8 only, we explicitly declare it for clarity.
    converter.target_spec.supported_types = [tf.int8]

    # These set the input and output tensors to uint8 (added in r2.3)
    converter.inference_input_type = tf.uint8
    converter.inference_output_type = tf.uint8

    converter.allow_custom_ops = True
    converter.experimental_new_converter = True
    converter.experimental_new_quantizer = True

    tflite_model = converter.convert()
    tf.lite.experimental.Analyzer.analyze(model_content=tflite_model, gpu_compatibility=True)

    # with open('test.tflite', 'wb') as f:
    #     f.write(tflite_model)

    return tflite_model
