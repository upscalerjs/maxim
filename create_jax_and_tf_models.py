import os
import argparse
import sys
sys.path.append('./maxim')
from jax_model import get_jax_model, convert_jax_to_tfjs
from utils import evaluate_jax_models
import tensorflow as tf
tf.config.experimental.set_visible_devices([], 'GPU')

def create_jax_and_tf_models(task, dataset, tfjs_output_folder, quantization_settings):
    checkpoint_path = f'gs://gresearch/maxim/ckpt/{task}/{dataset}/checkpoint.npz' # path to the checkpoint on google storage

    print(f'Creating a jax model for task "{task}" and dataset "{dataset}"')
    jax_model, params = get_jax_model(task, checkpoint_path)

    # Convert the Python model into a TFJS model, if we have not already done so
    if os.path.exists(f'{tfjs_output_folder}/model.json') is False:
        print(f'Creating a Tensorflow.js model for task "{task}" and dataset "{dataset}" at path {tfjs_output_folder}')
        convert_jax_to_tfjs(jax_model, tfjs_output_folder, params, quantization_settings)
    else:
        print(f'Tensorflow.js model already exists for task "{task}" and dataset "{dataset}" at path {tfjs_output_folder}')
        

def parse_args():
    parser = argparse.ArgumentParser(
        description=""
    )
    parser.add_argument(
        "-t",
        "--task",
        default="Denoising",
        type=str,
    )
    parser.add_argument(
        "-d",
        "--dataset",
        type=str,
    )
    parser.add_argument(
        '-o',
        '--output',
        type=str,
    )
    parser.add_argument(
        '-q',
        '--quantization_settings',
        type=str,
    )
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    create_jax_and_tf_models(args.task, args.dataset, args.output, args.quantization_settings)
