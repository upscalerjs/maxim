import json
import shutil
import tempfile
import matplotlib.pyplot as plt
from PIL import Image
import tensorflow as tf
from tensorflow import keras
import pathlib
import subprocess
import numpy as np
import sys
sys.path.append('./maxim_tf')
from maxim_tf.maxim.configs import MAXIM_CONFIGS
from maxim_tf.convert_to_tf import _MODEL_VARIANT_DICT, port_jax_params


def do(cmd, cwd='./'):
    cmd = [str(part) for part in cmd if part]
    # print('Running shell command:', ' '.join(cmd))
    try:
        return subprocess.check_output(cmd, cwd=cwd).decode()
    except subprocess.CalledProcessError as e:
        raise Exception(e.output.decode())
        
def run_node(model, image_path):
    with tempfile.NamedTemporaryFile() as f:
        output = do([
            'node',
            './node.js',
            model / 'model.json',
            image_path,
            f.name,
        ])
        print(output)
        
        loaded = json.load(f)
        shape = loaded['shape']
        data = np.array(loaded['data']).reshape(shape)
        return tf.convert_to_tensor(data, dtype=tf.float32)

def convert_to_tfjs(input, output, quantization=''):
    do([
        'tensorflowjs_converter',
        '--input_format=tf_saved_model',
        '--output_format=tfjs_graph_model',
        '--skip_op_check',
        quantization,
        input,
        output
    ])

def download_python_model(task, ckpt_path, output):
    do([
        'python3',
        'download_python_model.py',
        '--task',
        task,
        '--ckpt_path',
        ckpt_path,
        '--output',
        output,
    ])

    
def show_image(image_path, final_pred_image):
    plt.figure(figsize=(15, 15))

    plt.subplot(1, 2, 1)
    input_image = np.asarray(Image.open(image_path).convert("RGB"), np.float32) / 255.0
    imshow(input_image, "Input Image")

    plt.subplot(1, 2, 2)
    imshow(final_pred_image, "Predicted Image")
    plt.show()    

# Based on https://www.tensorflow.org/lite/examples/style_transfer/overview#visualize_the_inputs
def imshow(image, title=None):
    if len(image.shape) > 3:
        image = tf.squeeze(image, axis=0)

    plt.imshow(image)
    if title:
        plt.title(title)

def resize_image(image, target_dim):
    # Resize the image so that the shorter dimension becomes `target_dim`.
    shape = tf.cast(tf.shape(image)[1:-1], tf.float32)
    short_dim = min(shape)
    scale = target_dim / short_dim
    new_shape = tf.cast(shape * scale, tf.int32)
    image = tf.image.resize(image, new_shape)

    # Central crop the image.
    image = tf.image.resize_with_crop_or_pad(image, target_dim, target_dim)

    return image

def download_and_save_image(output_path, image_url):
    image_path = tf.keras.utils.get_file(origin=image_url)
    input_img = np.asarray(Image.open(image_path).convert("RGB"), np.float32) / 255.0
    input_img = tf.expand_dims(input_img, axis=0)
    input_img = resize_image(input_img, 256) # Images are hardcoded to 256
    pixels = (tf.squeeze(input_img).numpy() * 255).astype(np.uint8)
    im = Image.fromarray(pixels)
    im.save(output_path)
    return input_img

def evaluate_models(python_model_path, tfjs_model_path, image_path, sample_image_url):
    preprocessed_image = download_and_save_image(image_path, sample_image_url)
    python_model = tf.keras.models.load_model(python_model_path)
    python_prediction = python_model.predict(preprocessed_image)
    tfjs_prediction = run_node(tfjs_model_path, image_path)

    print('Python output')
    show_image(image_path, np.array((np.clip(python_prediction, 0.0, 1.0)).astype(np.float32)))
    print('Node output')
    show_image(image_path, np.array((np.clip(tfjs_prediction, 0.0, 1.0)).astype(np.float32)))

    return python_prediction, tfjs_prediction
