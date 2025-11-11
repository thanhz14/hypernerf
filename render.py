# @title Define imports and utility functions.

import jax
import jax.numpy as jnp
from jax import grad, jit, vmap
from jax import random

import flax
import flax.linen as nn
from flax import jax_utils
import optax  # ← THAY ĐỔI: Dùng optax thay vì flax.optim
from flax.training import train_state  # ← THÊM
from flax.metrics import tensorboard
from flax.training import checkpoints

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

# Monkey patch logging
def myprint(msg, *args, **kwargs):
    print(msg % args)

logging.info = myprint 
logging.warn = myprint
logging.error = myprint

# @title Model and dataset configuration

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

exp_config = configs.ExperimentConfig()
train_config = configs.TrainConfig()
eval_config = configs.EvalConfig()

display(Markdown(gin.config.markdown(gin.config_str())))

# @title Create datasource and show an example.

from hypernerf import datasets
from hypernerf import image_utils

dummy_model = models.NerfModel({}, 0, 0)
datasource = exp_config.datasource_cls(
    image_scale=exp_config.image_scale,
    random_seed=exp_config.random_seed,
    use_warp_id=dummy_model.use_warp,
    use_appearance_id=(
        dummy_model.nerf_embed_key == 'appearance'
        or dummy_model.hyper_embed_key == 'appearance'),
    use_camera_id=dummy_model.nerf_embed_key == 'camera',
    use_time=dummy_model.warp_embed_key == 'time')

mediapy.show_image(datasource.load_rgb(datasource.train_ids[0]))

# @title Load model

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
elastic_loss_weight_sched = schedules.from_config(train_config.elastic_loss_weight_schedule)
hyper_alpha_sched = schedules.from_config(train_config.hyper_alpha_schedule)
hyper_sheet_alpha_sched = schedules.from_config(train_config.hyper_sheet_alpha_schedule)

print('DATASOURCE: ',datasource.embeddings_dict)
rng, key = random.split(rng)
params = {}
model, params['model'] = models.construct_nerf(
    key,
    batch_size=train_config.batch_size,
    embeddings_dict=datasource.embeddings_dict,
    near=datasource.near,
    far=datasource.far)

# ===== THAY ĐỔI: Dùng optax thay vì flax.optim =====
tx = optax.adam(learning_rate=learning_rate_sched(0))

# Tạo TrainState với optax
state = model_utils.TrainState.create(
    apply_fn=model.apply,
    params=params,
    tx=tx,
    nerf_alpha=nerf_alpha_sched(0),
    warp_alpha=warp_alpha_sched(0),
    hyper_alpha=hyper_alpha_sched(0),
    hyper_sheet_alpha=hyper_sheet_alpha_sched(0)
)

scalar_params = training.ScalarParams(
    learning_rate=learning_rate_sched(0),
    elastic_loss_weight=elastic_loss_weight_sched(0),
    warp_reg_loss_weight=train_config.warp_reg_loss_weight,
    warp_reg_loss_alpha=train_config.warp_reg_loss_alpha,
    warp_reg_loss_scale=train_config.warp_reg_loss_scale,
    background_loss_weight=train_config.background_loss_weight,
    hyper_reg_loss_weight=train_config.hyper_reg_loss_weight)

logging.info('Restoring checkpoint from %s', checkpoint_dir)
state = checkpoints.restore_checkpoint(checkpoint_dir, state)

step = state.step + 1
# Không replicate cho local single device
del params

# @title Define pmapped render function.

import functools
from hypernerf import evaluation

devices = jax.devices()
print(f'✅ Devices: {len(devices)} ({"GPU" if jax.devices()[0].platform == "gpu" else "CPU"})')

def _model_fn(key_0, key_1, params, rays_dict, extra_params):
    out = model.apply({'params': params},
                      rays_dict,
                      extra_params=extra_params,
                      rngs={
                          'coarse': key_0,
                          'fine': key_1
                      },
                      mutable=False)
    return out

# Dùng jit cho local, không dùng pmap
pmodel_fn = jax.jit(_model_fn)

render_fn = functools.partial(evaluation.render_image,
                              model_fn=pmodel_fn,
                              device_count=1,
                              chunk=eval_config.chunk)

print('✅ Render function ready')

# @title Load cameras.

from hypernerf import utils

camera_path = 'camera-paths/orbit-mild'  # @param {type: 'string'}

camera_dir = Path(data_dir, camera_path)
print(f'Loading cameras from {camera_dir}')
test_camera_paths = datasource.glob_cameras(camera_dir)
test_cameras = utils.parallel_map(datasource.load_camera, test_camera_paths, show_pbar=True)
#===================================================================================
# @title Render video frames.
from hypernerf import visualization as viz
from hypernerf import image_utils

print('✅ Starting render...')
rng = rng + jax.process_index()
chunk_size = 4096
results = []

for i in range(len(test_cameras)):
    print(f'Rendering frame {i+1}/{len(test_cameras)}')
    camera = test_cameras[i]
    batch = datasets.camera_to_rays(camera)
    
    # SAVE ORIGINAL SHAPE
    orig_h, orig_w = batch['origins'].shape[0], batch['origins'].shape[1]
    print(f" Original shape: {orig_h}x{orig_w}")
    
    # ===== FLATTEN ĐÚNG =====
    origins = batch['origins'].reshape(-1, 3)  # (H*W, 3)
    directions = batch['directions'].reshape(-1, 3)  # (H*W, 3)
    num_rays = origins.shape[0]
    
    print(f" After flatten: origins {origins.shape}, directions {directions.shape}")
    
    rgb_parts = []
    depth_parts = []
    
    # Render in chunks
    for chunk_start in range(0, num_rays, chunk_size):
        chunk_end = min(chunk_start + chunk_size, num_rays)
        chunk_size_actual = chunk_end - chunk_start
        
        # ===== metadata TRONG rays_dict với SHAPE ĐÚNG (batch, 1) =====
        # ===== rays_dict KHÔNG CÓ metadata =====
        # ===== rays_dict CÓ metadata BÊN TRONG =====
        rays_dict = {
            'origins': origins[chunk_start:chunk_end],
            'directions': directions[chunk_start:chunk_end],
            'metadata': {
                'appearance': jnp.full((chunk_size_actual, 1), 0, dtype=jnp.uint32),
                'warp': jnp.full((chunk_size_actual, 1), 0, dtype=jnp.uint32),
            }
        }

        rng, key_0, key_1 = random.split(rng, 3)

        # ===== GỌI MODEL - metadata ĐÃ TRONG rays_dict =====
        render_output = model.apply(
            {'params': state.params['model']},
            rays_dict,
            extra_params=state.extra_params(),
            rngs={'coarse': key_0, 'fine': key_1},
            mutable=False,
            use_warp=False
        )


        
        rgb_parts.append(np.array(render_output['coarse']['rgb']))
        depth_parts.append(np.array(render_output['coarse']['med_depth']))
    
    # Concatenate và reshape
    rgb = np.concatenate(rgb_parts, axis=0).reshape(orig_h, orig_w, 3)
    depth_med = np.concatenate(depth_parts, axis=0).reshape(orig_h, orig_w)
    
    results.append((rgb, depth_med))
    
    # Visualize
    depth_viz = viz.colorize(depth_med, cmin=datasource.near, 
                             cmax=datasource.far, invert=True)
    mediapy.show_images([rgb, depth_viz])

print('✅ Rendering complete!')

# Save video
fps = 30
frames = []
for rgb, depth in results:
    depth_viz = viz.colorize(depth, cmin=datasource.near, 
                             cmax=datasource.far, invert=True)
    frame = np.concatenate([rgb, depth_viz], axis=1)
    frames.append(image_utils.image_to_uint8(frame))

mediapy.show_video(frames, fps=fps)
