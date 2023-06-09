import time
import tensorflow as tf
import tempfile
import numpy as np
from utils import run_node, run_jax_model, show_image
from PIL import Image
from skimage.metrics import structural_similarity

def tensor_to_image(tensor):
    if len(tensor.shape) == 4:
        tensor = np.squeeze(tensor)
    # return np.array((np.clip(tensor, 0.0, 1.0)).astype(np.float32))
    return (np.array((np.clip(tensor, 0.0, 1.0)) * 255).astype(np.int16))

def compare_images(python_prediction, tfjs_prediction, image_path):
    try:
        print('Python output')
        python_img = tensor_to_image(python_prediction)
        show_image(python_img, image_path)

        print('Node output')
        tfjs_img = tensor_to_image(tfjs_prediction)
        show_image(tfjs_img, image_path)

        ssim = structural_similarity(python_img, tfjs_img, channel_axis = -1)

        print(f'SSIM: {ssim}')
        return (python_prediction, python_img), (tfjs_prediction, tfjs_img), ssim
    except Exception as e:
        print('Failed SSIM', e)
        pass

    return (python_prediction, None), (tfjs_prediction, None), None

def evaluate_models(python_model_path, tfjs_model_path, sample_image_url, input_resolution = 256):
    with tempfile.NamedTemporaryFile(suffix='.png') as f:
        preprocessed_image = download_and_save_image(f.name, sample_image_url, input_resolution)
        python_model = tf.keras.models.load_model(python_model_path)
        python_prediction = python_model.predict(preprocessed_image)
        tfjs_prediction = run_node(tfjs_model_path, f.name)
        return compare_images(python_prediction, tfjs_prediction, f.name)


def evaluate_jax_models(model, params, tfjs_model_path, sample_image_url, input_resolution=None):
    with tempfile.NamedTemporaryFile(suffix='.png') as f:
        preprocessed_image = download_and_save_image(f.name, sample_image_url, input_resolution=input_resolution)
        tfjs_prediction = run_node(tfjs_model_path, f.name)
        python_prediction, duration = run_jax_model(model, params, preprocessed_image.numpy())
        print('Python Prediction took', duration)
        return compare_images(python_prediction, tfjs_prediction, f.name)


def download_and_save_image(output_path, image_url, input_resolution = None, crop=True):
    image_path = tf.keras.utils.get_file(origin=image_url)
    input_img = np.asarray(Image.open(image_path).convert("RGB"), np.float32) / 255.0
    input_img = tf.expand_dims(input_img, axis=0)
    if crop and input_resolution is not None:
        input_img = tf.image.resize_with_crop_or_pad(input_img, input_resolution[0], input_resolution[1])
    # input_img = resize_image(input_img, input_resolution)
    # pixels = (tf.squeeze(input_img).numpy() * 255).astype(np.uint8)
    tf.keras.utils.save_img(output_path, tf.squeeze(input_img))

    # im = Image.fromarray(pixels)
    # im.save(output_path)
    

    return input_img
