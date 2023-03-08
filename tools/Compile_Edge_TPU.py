import os


def compile_edgetpu(path):
    if not os.path.exists(path):
        print(f"{path} not found")
        return

    edgetpu_model_name = path.replace('.tflite', '_edgetpu.tflite')
    os.system('edgetpu_compiler -sa {}'.format(path))

    return edgetpu_model_name

