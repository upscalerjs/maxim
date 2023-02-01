import sys
sys.path.append('./cloned-code/maxim')
sys.path.append('./cloned-code/tfjs/tfjs-converter/python')

import collections
import io
from maxim.models.maxim import Model


from flax import linen as nn
import ml_collections
import numpy as np
import jax.numpy as jnp
from tqdm import tqdm
import flax
import jax.numpy as jnp
import tensorflow as tf
import tempfile
from tensorflowjs.converters.jax_conversion import _TF_SERVING_KEY, _ReusableSavedModelWrapper
from tensorflowjs.converters import tf_saved_model_conversion_v2 as saved_model_conversion
from tensorflowjs.converters.converter import _parse_quantization_dtype_map
from jax.experimental import jax2tf

class Wrapper(nn.Module):
    backbone: nn.Module

    @nn.compact
    def __call__(self, x):
        return self.backbone(x)[-1][-1]



_MODEL_VARIANT_DICT = {
    'Denoising': 'S-3',
    'Deblurring': 'S-3',
    'Deraining': 'S-2',
    'Dehazing': 'S-2',
    'Enhancement': 'S-2',
}

_MODEL_CONFIGS = {
    'variant': '',
    'dropout_rate': 0.0,
    'num_outputs': 3,
    'use_bias': True,
    'num_supervision_scales': 3,
}



def build_model(task = "Denoising"):
    model_configs = ml_collections.ConfigDict(_MODEL_CONFIGS)

    model_configs.variant = _MODEL_VARIANT_DICT[task]

    model = Model(**model_configs)
    return model


def recover_tree(keys, values):
  """Recovers a tree as a nested dict from flat names and values.

  This function is useful to analyze checkpoints that are saved by our programs
  without need to access the exact source code of the experiment. In particular,
  it can be used to extract an reuse various subtrees of the scheckpoint, e.g.
  subtree of parameters.
  Args:
    keys: a list of keys, where '/' is used as separator between nodes.
    values: a list of leaf values.
  Returns:
    A nested tree-like dict.
  """
  tree = {}
  sub_trees = collections.defaultdict(list)
  for k, v in zip(keys, values):
    if '/' not in k:
      tree[k] = v
    else:
      k_left, k_right = k.split('/', 1)
      sub_trees[k_left].append((k_right, v))
  for k, kv_pairs in sub_trees.items():
    k_subtree, v_subtree = zip(*kv_pairs)
    tree[k] = recover_tree(k_subtree, v_subtree)
  return tree


def mod_padding_symmetric(image, factor=64):
  """Padding the image to be divided by factor."""
  height, width = image.shape[0], image.shape[1]
  height_pad, width_pad = ((height + factor) // factor) * factor, (
      (width + factor) // factor) * factor
  padh = height_pad - height if height % factor != 0 else 0
  padw = width_pad - width if width % factor != 0 else 0
  image = jnp.pad(
      image, [(padh // 2, padh // 2), (padw // 2, padw // 2), (0, 0)],
      mode='reflect')
  return image


def get_params(ckpt_path):
    with tf.io.gfile.GFile(ckpt_path, 'rb') as f:
        data = f.read()
    values = np.load(io.BytesIO(data))
    params = recover_tree(*zip(*values.items()))
    params = params['opt']['target']

    return params

def get_jax_model(task, ckpt_path):
    model_params = get_params(ckpt_path) # Parse the config
    model = build_model(task=task) # Build Model
    wrapper = Wrapper(backbone=model)

    params = {
        'params': {
            'backbone': flax.core.freeze(model_params),
        }
    }

    return wrapper, params

def convert_jax_to_tfjs(model, target_path, params, quantization_settings):
    convert_jax_with_quantization(
        apply_fn=model.apply,
        params=params,
        input_signatures=[tf.TensorSpec([1, 256, 256, 3], tf.float32)],
        model_dir=str(target_path),
        quantization_settings=quantization_settings
    )




def convert_jax_with_quantization(
    apply_fn,
    params,
    input_signatures,
    model_dir,
    quantization_settings,
    polymorphic_shapes=None,
):
    if (quantization_settings and quantization_settings not in ['float16', 'uint16', 'uint8']):
        raise Exception('Invalid quantization, must be uint8, uint16, or float16')
    pbar = tqdm(total = 11)
    def update(msg):
        pbar.update(1)
        pbar.set_description(msg)

    update('Setting polymorphic shapes')
    if polymorphic_shapes is not None:
        # If polymorphic shapes are provided, add a polymorphic spec for the
        # first argument to `apply_fn`, which are the parameters.
        polymorphic_shapes = [None, *polymorphic_shapes]

    update('converting with jax2f')
    tf_fn = jax2tf.convert(
        apply_fn,
        # Gradients must be included as 'PreventGradient' is not supported.
        with_gradient=True,
        polymorphic_shapes=polymorphic_shapes,
        # Do not use TFXLA Ops because these aren't supported by TFjs, but use
        # workarounds instead. More information:
        # https://github.com/google/jax/tree/main/jax/experimental/jax2tf#tensorflow-xla-ops
        enable_xla=False)

    update('mapping nested structure')
    # Create tf.Variables for the parameters. If you want more useful variable
    # names, you can use `tree.map_structure_with_path` from the `dm-tree`
    # package.
    param_vars = tf.nest.map_structure(lambda param: tf.Variable(param, trainable=True), params)

    update('calling tf.function')
    # Do not use TF's jit compilation on the function.
    tf_graph = tf.function(lambda *xs: tf_fn(param_vars, *xs), autograph=False, jit_compile=False)

    update('getting concrete signature')
    # This signature is needed for TensorFlow Serving use.
    signatures = {
        _TF_SERVING_KEY: tf_graph.get_concrete_function(*input_signatures)
    }

    update('getting reusable saved model wrapper')
    wrapper = _ReusableSavedModelWrapper(tf_graph, param_vars)
    update('saving options')
    saved_model_options = tf.saved_model.SaveOptions( experimental_custom_gradients=True)

    update('setting up temp directory')
    with tempfile.TemporaryDirectory() as saved_model_dir:
        update('saving tf model')
        tf.saved_model.save(
            wrapper,
            saved_model_dir,
            signatures=signatures,
            options=saved_model_options)
        update('converting tf model')
        saved_model_conversion.convert_tf_saved_model(saved_model_dir, model_dir, quantization_dtype_map=_parse_quantization_dtype_map(
            float16=None if quantization_settings != 'float16' else True, 
            uint8=None if quantization_settings != 'uint8' else True, 
            uint16=None if quantization_settings != 'uint16' else True, 
            quantization_bytes=None,
        ))
    update('done')

    pbar.close()
