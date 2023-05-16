import pathlib
import argparse
import sys
sys.path.append('./maxim')
from jax_model import get_quantization_dtype_map, Wrapper, Params
from fixing_ensure_shapes import remove_ensure_shape_nodes_from_model_json
from typing import Optional, Literal
import tensorflow as tf
import tensorflowjs as tfjs
tf.config.experimental.set_visible_devices([], 'GPU')

IMAGE_DIVISIBLE_BY = 64

def create_jax_and_tf_models(
    jax_model: Wrapper, 
    params: Params, 
    tfjs_output_folder: str | pathlib.Path, 
    quantization_settings: Optional[Literal['float16', 'uint16', 'uint8']], 
    input_size: Optional[int]=None,
    remove_ensure_shape_nodes=True,
):
    # Convert the Python model into a TFJS model, if we have not already done so
        # convert_jax_to_tfjs(tf_output_folder, jax_model, tfjs_output_folder, params, quantization_settings, use_cache=use_cache)
    apply_fn=jax_model.apply
    if input_size is None:
        input_signatures=[tf.TensorSpec([None, None, None, 3], tf.float32)]
        polymorphic_shapes=[f"(b, h * {IMAGE_DIVISIBLE_BY}, w * {IMAGE_DIVISIBLE_BY}, ...)"]
    else:
        input_signatures=[tf.TensorSpec([None, input_size, input_size, 3], tf.float32)]
        polymorphic_shapes=[f"(b, ...)"]

    tfjs_output_folder = str(tfjs_output_folder)

    tfjs.converters.convert_jax(
        apply_fn,
        params, 
        model_dir=tfjs_output_folder, 
        input_signatures=input_signatures, 
        polymorphic_shapes=polymorphic_shapes,
        quantization_dtype_map=get_quantization_dtype_map(quantization_settings),
    )

    if remove_ensure_shape_nodes:
        remove_ensure_shape_nodes_from_model_json(f'{tfjs_output_folder}/model.json')
        

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
        '-f',
        '--tf_output',
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
    create_jax_and_tf_models(args.task, args.dataset, args.tf_output, args.output, args.quantization_settings)
