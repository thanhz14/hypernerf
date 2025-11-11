# @title Define imports and utility functions.

import jax
# from jax.config import config as jax_config
import jax.numpy as jnp
from jax import grad, jit, vmap
from jax import random

import flax
import flax.linen as nn
from flax import jax_utils
import optax as optim
from flax.metrics import tensorboard
from flax.training import checkpoints
from flax.training import train_state

from absl import logging
from io import BytesIO
import random as pyrandom
import numpy as np
import PIL
import IPython
import tempfile
import imageio
import mediapy
from IPython.display import display, HTML
from base64 import b64encode


# Monkey patch logging.
def myprint(msg, *args, **kwargs):
 print(msg % args)

logging.info = myprint 
logging.warn = myprint
logging.error = myprint

# @title Model and dataset configuration
# @markdown Change the directories to where you saved your capture and experiment.


from pathlib import Path
from pprint import pprint
import gin
from IPython.display import display, Markdown

from hypernerf import models
from hypernerf import modules
from hypernerf import warping
from hypernerf import datasets
from hypernerf import configs


# @markdown The working directory where the trained model is.
train_dir = r'D:\Acer\Code\Dat_pr\HYPERNERF\nerfies\hypernerf_experiments\capture1\exp1'  # @param {type: "string"}
# @markdown The directory to the dataset capture.
data_dir = r'D:\Acer\Code\Dat_pr\HYPERNERF\nerfies\captures\capture1'  # @param {type: "string"}

checkpoint_dir = Path(train_dir, 'checkpoints')
checkpoint_dir.mkdir(exist_ok=True, parents=True)

config_path = Path(train_dir, 'config.gin')
with open(config_path, 'r') as f:
  logging.info('Loading config from %s', config_path)
  config_str = f.read()
gin.parse_config(config_str)

config_path = Path(train_dir, 'config.gin')
with open(config_path, 'w') as f:
  logging.info('Saving config to %s', config_path)
  f.write(config_str)

exp_config = configs.ExperimentConfig()
train_config = configs.TrainConfig()
eval_config = configs.EvalConfig()

display(Markdown(
    gin.config.markdown(gin.config_str())))




# @title Create datasource and show an example.

from hypernerf import datasets
from hypernerf import image_utils

dummy_model = models.NerfModel({}, 0, 0)
datasource = exp_config.datasource_cls(
    image_scale=exp_config.image_scale,
    random_seed=exp_config.random_seed,
    # Enable metadata based on model needs.
    use_warp_id=dummy_model.use_warp,
    use_appearance_id=(
        dummy_model.nerf_embed_key == 'appearance'
        or dummy_model.hyper_embed_key == 'appearance'),
    use_camera_id=dummy_model.nerf_embed_key == 'camera',
    use_time=dummy_model.warp_embed_key == 'time')

mediapy.show_image(datasource.load_rgb(datasource.train_ids[0]))



# @title Load model
# @markdown Defines the model and initializes its parameters.

from flax.training import checkpoints
from hypernerf import models
from hypernerf import model_utils
from hypernerf import schedules
from hypernerf import training

rng = random.PRNGKey(exp_config.random_seed)
np.random.seed(exp_config.random_seed + jax.process_index())
devices_to_use = jax.devices()

learning_rate_sched = schedules.from_config(train_config.lr_schedule)
nerf_alpha_sched = schedules.from_config(train_config.nerf_alpha_schedule)
warp_alpha_sched = schedules.from_config(train_config.warp_alpha_schedule)
elastic_loss_weight_sched = schedules.from_config(
train_config.elastic_loss_weight_schedule)
hyper_alpha_sched = schedules.from_config(train_config.hyper_alpha_schedule)
hyper_sheet_alpha_sched = schedules.from_config(
    train_config.hyper_sheet_alpha_schedule)

rng, key = random.split(rng)
params = {}
model, params['model'] = models.construct_nerf(
      key,
      batch_size=train_config.batch_size,
      embeddings_dict=datasource.embeddings_dict,
      near=datasource.near,
      far=datasource.far)

#optimizer_def = optim.adam(learning_rate_sched(0))
#optimizer = optimizer_def.create(params)

import optax
from flax.training import train_state
from flax import struct


import optax
from flax import struct
import jax
import jax.numpy as jnp

# ================= OptimizerWrapper =================
@struct.dataclass
class OptimizerWrapper:
    apply_fn: callable      # Hàm forward của model
    params: any             # Tham số model
    tx: optax.GradientTransformation  # Optax optimizer
    opt_state: any          # Trạng thái nội bộ optimizer
    step: int = 0           # Step hiện tại

    @classmethod
    def create(cls, apply_fn, params, optimizer_def):
        tx = optimizer_def
        opt_state = tx.init(params)
        return cls(apply_fn=apply_fn, params=params, tx=tx, opt_state=opt_state)

    def apply_gradients(self, grads):
        updates, new_opt_state = self.tx.update(grads, self.opt_state, self.params)
        new_params = optax.apply_updates(self.params, updates)
        return self.replace(
            params=new_params,
            opt_state=new_opt_state,
            step=self.step + 1
        )

    def replace(self, **kwargs):
        return self.__class__(**{**self.__dict__, **kwargs})


# ================= TrainState =================
@struct.dataclass
class TrainState:
    optimizer: OptimizerWrapper
    nerf_alpha: float
    warp_alpha: float
    hyper_alpha: float
    hyper_sheet_alpha: float


# ==================== Khởi tạo ====================
# Learning rate schedule (ví dụ)
def learning_rate_sched(step):
    return 1e-3

optimizer_def = optax.adam(learning_rate_sched(0))

# Giả sử model của bạn là model = ...
optimizer = OptimizerWrapper.create(
    apply_fn=model.apply,  # model.apply
    params=params,         # params model
    optimizer_def=optimizer_def
)

state = TrainState(
    optimizer=optimizer,
    nerf_alpha=nerf_alpha_sched(0),
    warp_alpha=warp_alpha_sched(0),
    hyper_alpha=hyper_alpha_sched(0),
    hyper_sheet_alpha=hyper_sheet_alpha_sched(0)
)

# Scalar params
scalar_params = training.ScalarParams(
    learning_rate=learning_rate_sched(0),
    elastic_loss_weight=elastic_loss_weight_sched(0),
    warp_reg_loss_weight=train_config.warp_reg_loss_weight,
    warp_reg_loss_alpha=train_config.warp_reg_loss_alpha,
    warp_reg_loss_scale=train_config.warp_reg_loss_scale,
    background_loss_weight=train_config.background_loss_weight,
    hyper_reg_loss_weight=train_config.hyper_reg_loss_weight
)

# ==================== Checkpoint ====================
import flax.training.checkpoints as checkpoints
from flax import jax_utils

logging.info('Restoring checkpoint from %s', checkpoint_dir)
state = checkpoints.restore_checkpoint(checkpoint_dir, state)
step = state.optimizer.step + 1
state = jax_utils.replicate(state, devices=devices_to_use)

logging.info('Restoring checkpoint from %s', checkpoint_dir)
state = checkpoints.restore_checkpoint(checkpoint_dir, state)
step = state.tx.state.step + 1
state = jax_utils.replicate(state, devices=devices_to_use)
del params



# @title Define pmapped render function.

import functools
from hypernerf import evaluation

devices = jax.devices()


def _model_fn(key_0, key_1, params, rays_dict, extra_params):
  out = model.apply({'params': params},
                    rays_dict,
                    extra_params=extra_params,
                    rngs={
                        'coarse': key_0,
                        'fine': key_1
                    },
                    mutable=False)
  return jax.lax.all_gather(out, axis_name='batch')

pmodel_fn = jax.pmap(
    # Note rng_keys are useless in eval mode since there's no randomness.
    _model_fn,
    in_axes=(0, 0, 0, 0, 0),  # Only distribute the data input.
    devices=devices_to_use,
    axis_name='batch',
)

render_fn = functools.partial(evaluation.render_image,
                              model_fn=pmodel_fn,
                              device_count=len(devices),
                              chunk=eval_config.chunk)

# @title Load cameras.

from hypernerf import utils

camera_path = 'camera-paths/orbit-mild'  # @param {type: 'string'}

camera_dir = Path(data_dir, camera_path)
print(f'Loading cameras from {camera_dir}')
test_camera_paths = datasource.glob_cameras(camera_dir)
test_cameras = utils.parallel_map(datasource.load_camera, test_camera_paths, show_pbar=True)


# @title Render video frames.
from hypernerf import visualization as viz


rng = rng + jax.process_index()  # Make random seed separate across hosts.
keys = random.split(rng, len(devices))

results = []
for i in range(len(test_cameras)):
  print(f'Rendering frame {i+1}/{len(test_cameras)}')
  camera = test_cameras[i]
  batch = datasets.camera_to_rays(camera)
  batch['metadata'] = {
      'appearance': jnp.zeros_like(batch['origins'][..., 0, jnp.newaxis], jnp.uint32),
      'warp': jnp.zeros_like(batch['origins'][..., 0, jnp.newaxis], jnp.uint32),
  }

  render = render_fn(state, batch, rng=rng)
  rgb = np.array(render['rgb'])
  depth_med = np.array(render['med_depth'])
  results.append((rgb, depth_med))
  depth_viz = viz.colorize(depth_med.squeeze(), cmin=datasource.near, cmax=datasource.far, invert=True)
  mediapy.show_images([rgb, depth_viz])


  # @title Show rendered video.

fps = 30  # @param {type:'number'}

frames = []
for rgb, depth in results:
  depth_viz = viz.colorize(depth.squeeze(), cmin=datasource.near, cmax=datasource.far, invert=True)
  frame = np.concatenate([rgb, depth_viz], axis=1)
  frames.append(image_utils.image_to_uint8(frame))

mediapy.show_video(frames, fps=fps)