import os
import argparse
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'cloned-code/maxim_tf'))
from utils import download_keras_model, convert_to_tfjs, evaluate_models
import tensorflow as tf
tf.config.experimental.set_visible_devices([], 'GPU')

def create_tf_models(task, dataset, python_output_folder, tfjs_output_folder, quantization_settings, input_resolution=None):
    checkpoint_path = f'gs://gresearch/maxim/ckpt/{task}/{dataset}/checkpoint.npz' # path to the checkpoint on google storage
    
    # Download a MAXIM model and save it locally, if we have not already done so
    if os.path.exists(python_output_folder) is False:
        print(f'Creating a python model for task "{task}" and dataset "{dataset}"')
        download_keras_model(task, checkpoint_path, python_output_folder, input_resolution)
        
    # Convert the Python model into a TFJS model, if we have not already done so
    if os.path.exists(f'{tfjs_output_folder}/model.json') is False:
        print(f'Creating a Tensorflow.js model for task "{task}" and dataset "{dataset}"')
        if quantization_settings == '"float16"':
            quantization_settings = '--quantize_float16=*'
        elif quantization_settings == '"uint16"':
            quantization_settings = '--quantize_uint16=*'
        elif quantization_settings == '"uint8"':
            quantization_settings = '--quantize_uint8=*'
        elif quantization_settings == '' or quantization_settings is None:
            pass
        else:
            raise Exception(f'Unsupported quantization settings: {quantization_settings}')
        convert_to_tfjs(python_output_folder, tfjs_output_folder, quantization_settings)
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
        '-j',
        '--tfjs_output',
        type=str,
    )
    parser.add_argument(
        '-p',
        '--python_output',
        type=str,
    )
    parser.add_argument(
        "-i",
        "--input_resolution",
        type=int,
        help="Optional input resolution",
    )
    parser.add_argument(
        '-q',
        '--quantization_settings',
        type=str,
    )
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    create_tf_models(args.task, args.dataset, args.python_output, args.tfjs_output, args.quantization_settings, input_resolution = args.input_resolution)

