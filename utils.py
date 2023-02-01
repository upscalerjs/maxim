import json
import os
import flax
from skimage.metrics import structural_similarity
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

def run_jax_model(model, params, cache_name, preprocessed_image):
    cache_path = CACHE_DIR / f'{cache_name}.npy'
    if os.path.exists(cache_path) is False:
        result = model.apply(params, preprocessed_image)
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

def download_keras_model(task, ckpt_path, output):
    do([
        'python3',
        'download_keras_model.py',
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

def compare_images(python_prediction, tfjs_prediction, image_path):
    try:
        print('Python output')
        if len(python_prediction.shape) == 4:
            python_prediction = np.squeeze(python_prediction)
        python_img = np.array((np.clip(python_prediction, 0.0, 1.0)).astype(np.float32))
        show_image(image_path, python_img)

        print('Node output')
        if len(tfjs_prediction.shape) == 4:
            tfjs_prediction = np.squeeze(tfjs_prediction)
        tfjs_img = np.array((np.clip(tfjs_prediction, 0.0, 1.0)).astype(np.float32))
        show_image(image_path, tfjs_img)

        ssim = structural_similarity(python_img, tfjs_img, channel_axis = -1)

        print(f'SSIM: {ssim}')
        return (python_prediction, python_img), (tfjs_prediction, tfjs_img), ssim
    except Exception as e:
        print('Failed SSIM', e)
        pass

    return (python_prediction, None), (tfjs_prediction, None), None

def evaluate_models(python_model_path, tfjs_model_path, sample_image_url):
    with tempfile.NamedTemporaryFile(suffix='.png') as f:
        preprocessed_image = download_and_save_image(f.name, sample_image_url)
        python_model = tf.keras.models.load_model(python_model_path)
        python_prediction = python_model.predict(preprocessed_image)
        tfjs_prediction = run_node(tfjs_model_path, f.name)
        return compare_images(python_prediction, tfjs_prediction, f.name)


def evaluate_jax_models(model, params, tfjs_model_path, sample_image_url, ckpt_path):
    with tempfile.NamedTemporaryFile(suffix='.png') as f:
        preprocessed_image = download_and_save_image(f.name, sample_image_url)
        tfjs_prediction = run_node(tfjs_model_path, f.name)
        cache_name = get_cache_name(ckpt_path, sample_image_url)
        python_prediction = run_jax_model(model, params, cache_name, preprocessed_image.numpy())
        return compare_images(python_prediction, tfjs_prediction, f.name)
