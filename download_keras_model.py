import argparse
from tensorflow import keras
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'cloned-code/maxim_tf'))
from maxim.configs import MAXIM_CONFIGS
from convert_to_tf import _MODEL_VARIANT_DICT, port_jax_params
from create_maxim_model import Model


def download_python_model(task, ckpt_path, input_resolution = None):
    # From https://github.com/google-research/maxim/blob/main/maxim/run_eval.py#L55
    variant = _MODEL_VARIANT_DICT[task]
    configs = MAXIM_CONFIGS.get(variant)
    configs.update(
        {
            "variant": variant,
            "dropout_rate": 0.0,
            "num_outputs": 3,
            "use_bias": True,
            "num_supervision_scales": 3,
        }
    )

    _, orig_model = port_jax_params(configs, ckpt_path)

    if input_resolution:
        configs.update({"input_resolution": [input_resolution, input_resolution]})
        model = Model(**configs)
        model.set_weights(orig_model.get_weights())
    else:
        model = orig_model

    print("Model porting successful.")
    return keras.Model(inputs=model.inputs, outputs=model.layers[-1].output)

def parse_args():
    parser = argparse.ArgumentParser(
        description=""
    )
    parser.add_argument(
        "-t",
        "--task",
        default="Denoising",
        type=str,
        help="Name of the task on which the corresponding checkpoints were derived.",
    )
    parser.add_argument(
        "-c",
        "--ckpt_path",
        type=str,
        help="Checkpoint to port.",
    )
    parser.add_argument(
        "-i",
        "--input_resolution",
        type=int,
        help="Optional input resolution",
    )
    parser.add_argument(
        '-o',
        '--output',
        type=str,
    )
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    model = download_python_model(args.task, args.ckpt_path, args.input_resolution)
    model.save(args.output)
