import json
import time
import os
import flax
import tempfile
import matplotlib.pyplot as plt
from PIL import Image
from tensorflow import keras
import pathlib
import base64
import subprocess
import numpy as np

import tensorflow as tf
tf.config.experimental.set_visible_devices([], 'GPU')

CACHE_DIR = pathlib.Path('.cache')
CACHE_DIR.mkdir(exist_ok=True, parents=True)

def get_cache_name(*parts):
    return base64.b64encode(' '.join(parts).encode("ascii")).decode("utf-8")

def do(cmd, cwd='./'):
    cmd = [str(part) for part in cmd if part]
    print('Running shell command:', ' '.join(cmd))
    try:
        return subprocess.check_output(cmd, cwd=cwd).decode()
    except subprocess.CalledProcessError as e:
        raise Exception(e.output.decode())
        
def run_node(model, image_path, use_cache = True):
    cache_name = get_cache_name(str(model), image_path)
    if os.path.exists(CACHE_DIR / cache_name) is False or use_cache is False:
        with tempfile.NamedTemporaryFile() as f:
            output = do([
                'node',
                './node/index.js',
                os.path.join(os.path.dirname(__file__), str(model), 'model.json'),
                image_path,
                f.name,
            ])
            print(output)
            
            loaded = json.load(f)
            shape = loaded['shape']
            data = np.array(loaded['data']).reshape(shape)
            result = tf.convert_to_tensor(data, dtype=tf.float32)
        with open(CACHE_DIR / cache_name, 'wb') as f:
            result.numpy().tofile(f)
    else:
        with open(CACHE_DIR / cache_name, 'wb') as f:
            result = np.load(f)
    return result

def run_jax_model(model, params, preprocessed_image):
    start = time.time()
    result = model.apply(params, preprocessed_image)
    return result, time.time() - start

def run_tf_model(model, preprocessed_image, cache_name = None):
    cache_path = None
    if cache_name is not None:
        cache_path = CACHE_DIR / f'{cache_name}.npy'
    if cache_path is None or os.path.exists(cache_path) is False:
        result = model(preprocessed_image)
        if cache_path is not None:
            np.save(cache_path, np.array(result))
    else:
        result = np.load(cache_path)
    return result

def convert_to_tfjs(input, output, quantization=''):
    do([
        'tensorflowjs_converter',
        '--input_format=tf_saved_model',
        '--output_format=tfjs_graph_model',
        '--skip_op_check',
        quantization or '',
        input,
        output
    ])

def download_keras_model(task, ckpt_path, output, input_resolution = None):
    do([
        'python3',
        'download_keras_model.py',
        '--task',
        task,
        '--ckpt_path',
        ckpt_path,
        *([ '--input_resolution', input_resolution, ] if input_resolution else []),
        '--output',
        output,
    ])

    
def show_image(final_pred_image, image_path=None):
    plt.figure(figsize=(15, 15))

    plt.subplot(1, 2, 1)
    if image_path:
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
