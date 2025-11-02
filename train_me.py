# @title Define imports and utility functions.

import jax
import jax.numpy as jnp
from jax import grad, jit, vmap, random

import flax
import flax.linen as nn
from flax import jax_utils
import optax
from flax.training import train_state
from flax.metrics import tensorboard
from flax.training import checkpoints

from absl import logging
from io import BytesIO
import random as pyrandom
import numpy as np
import PIL
import IPython

# Monkey patch logging.
def myprint(msg, *args, **kwargs):
    print(msg % args)

logging.info = myprint
logging.warn = myprint
logging.error = myprint

def show_image(image, fmt='png'):
    from hypernerf import image_utils
    image = image_utils.image_to_uint8(image)
    f = BytesIO()
    PIL.Image.fromarray(image).save(f, fmt)
    IPython.display.display(IPython.display.Image(data=f.getvalue()))

########################
# @title Model and dataset configuration
import sys
sys.path.append('D:/Acer/Code/Dat_pr/hypernerf')

import hypernerf

from pathlib import Path
from pprint import pprint
import gin
from IPython.display import display, Markdown

from hypernerf import models
from hypernerf import modules
from hypernerf import warping
from hypernerf import datasets
from hypernerf import configs

# @markdown The working directory.
train_dir = 'hypernerf/nerfies/hypernerf_experiments/capture1/exp1'  # @param {type: "string"}
# @markdown The directory to the dataset capture.
data_dir = r'D:/Acer/Code/Dat_pr/hypernerf/nerfies/captures/capture1'  # @param {type: "string"}

# @markdown Training configuration.
max_steps = 100000  # @param {type: 'number'}
batch_size = 4096  # @param {type: 'number'}
image_scale = 8  # @param {type: 'number'}

# @markdown Model configuration.
use_viewdirs = True  #@param {type: 'boolean'}
use_appearance_metadata = True  #@param {type: 'boolean'}
num_coarse_samples = 64  # @param {type: 'number'}
num_fine_samples = 64  # @param {type: 'number'}

# @markdown Deformation configuration.
use_warp = True  #@param {type: 'boolean'}
warp_field_type = '@SE3Field'  #@param['@SE3Field', '@TranslationField']
warp_min_deg = 0  #@param{type:'number'}
warp_max_deg = 6  #@param{type:'number'}

# @markdown Hyper-space configuration.
hyper_num_dims = 8  #@param{type:'number'}
hyper_point_min_deg = 0  #@param{type:'number'}
hyper_point_max_deg = 1  #@param{type:'number'}
hyper_slice_method = 'bendy_sheet'  #@param['none', 'axis_aligned_plane', 'bendy_sheet']

checkpoint_dir = Path(train_dir, 'checkpoints')
checkpoint_dir.mkdir(exist_ok=True, parents=True)

config_str = f"""
DELAYED_HYPER_ALPHA_SCHED = {{
  'type': 'piecewise',
  'schedules': [
    (1000, ('constant', 0.0)),
    (0, ('linear', 0.0, %hyper_point_max_deg, 10000))
  ],
}}

ExperimentConfig.image_scale = {image_scale}
ExperimentConfig.datasource_cls = @NerfiesDataSource
NerfiesDataSource.data_dir = '{data_dir}'
NerfiesDataSource.image_scale = {image_scale}

NerfModel.use_viewdirs = {int(use_viewdirs)}
NerfModel.use_rgb_condition = {int(use_appearance_metadata)}
NerfModel.num_coarse_samples = {num_coarse_samples}
NerfModel.num_fine_samples = {num_fine_samples}

NerfModel.use_viewdirs = True
NerfModel.use_stratified_sampling = True
NerfModel.use_posenc_identity = False
NerfModel.nerf_trunk_width = 128
NerfModel.nerf_trunk_depth = 8

TrainConfig.max_steps = {max_steps}
TrainConfig.batch_size = {batch_size}
TrainConfig.print_every = 100
TrainConfig.use_elastic_loss = False
TrainConfig.use_background_loss = False

# Warp configs.
warp_min_deg = {warp_min_deg}
warp_max_deg = {warp_max_deg}
NerfModel.use_warp = {use_warp}
SE3Field.min_deg = %warp_min_deg
SE3Field.max_deg = %warp_max_deg
SE3Field.use_posenc_identity = False
NerfModel.warp_field_cls = @SE3Field

TrainConfig.warp_alpha_schedule = {{
    'type': 'linear',
    'initial_value': {warp_min_deg},
    'final_value': {warp_max_deg},
    'num_steps': {int(max_steps*0.8)},
}}

# Hyper configs.
hyper_num_dims = {hyper_num_dims}
hyper_point_min_deg = {hyper_point_min_deg}
hyper_point_max_deg = {hyper_point_max_deg}

NerfModel.hyper_embed_cls = @hyper/GLOEmbed
hyper/GLOEmbed.num_dims = %hyper_num_dims
NerfModel.hyper_point_min_deg = %hyper_point_min_deg
NerfModel.hyper_point_max_deg = %hyper_point_max_deg

TrainConfig.hyper_alpha_schedule = %DELAYED_HYPER_ALPHA_SCHED

hyper_sheet_min_deg = 0
hyper_sheet_max_deg = 6
HyperSheetMLP.min_deg = %hyper_sheet_min_deg
HyperSheetMLP.max_deg = %hyper_sheet_max_deg
HyperSheetMLP.output_channels = %hyper_num_dims

NerfModel.hyper_slice_method = '{hyper_slice_method}'
NerfModel.hyper_sheet_mlp_cls = @HyperSheetMLP
NerfModel.hyper_use_warp_embed = True

TrainConfig.hyper_sheet_alpha_schedule = ('constant', %hyper_sheet_max_deg)
"""

gin.parse_config(config_str)

config_path = Path(train_dir, 'config.gin')
with open(config_path, 'w') as f:
  logging.info('Saving config to %s', config_path)
  f.write(config_str)

exp_config = configs.ExperimentConfig()
train_config = configs.TrainConfig()
eval_config = configs.EvalConfig()

display(Markdown(gin.config.markdown(gin.config_str())))

#########################
# @title Load dataset
# THÊM PHẦN NÀY: Khởi tạo datasource
logging.info('Loading dataset from %s', data_dir)
# MỚI - Enable warp/appearance/camera IDs
datasource = exp_config.datasource_cls(
    data_dir=data_dir,
    image_scale=exp_config.image_scale,
    shuffle_pixels=True,
    camera_type='json',
    test_camera_trajectory='orbit-mild',
    use_warp_id=True,           # ← BẬT CÁI NÀY
    use_appearance_id=True,     # ← BẬT CÁI NÀY
    use_camera_id=True,         # ← BẬT CÁI NÀY
)


#########################
# @title Create training iterators
# ===== FIX EMBEDDINGS =====
# ===== FIX EMBEDDINGS - DICT BYPASS =====
import numpy as np

logging.info('Fixing embeddings dictionary...')

all_ids = sorted(list(set(datasource.train_ids + datasource.val_ids)))
logging.info('Total images: %d (train: %d, val: %d)', 
             len(all_ids), len(datasource.train_ids), len(datasource.val_ids))

# Create embeddings
embeddings_dict = {
    'warp': {img_id: idx for idx, img_id in enumerate(all_ids)},
    'appearance': {img_id: idx for idx, img_id in enumerate(all_ids)},
    'camera': {img_id: 0 for img_id in all_ids}
}

# Bypass property by modifying __dict__ directly
datasource.__dict__['_embeddings_dict'] = embeddings_dict
# Also try the direct name in case
datasource.__dict__['embeddings_dict'] = embeddings_dict

logging.info('✅ Embeddings: warp=%d, appearance=%d, camera=%d',
             len(embeddings_dict['warp']),
             len(embeddings_dict['appearance']),
             len(embeddings_dict['camera']))

# Verify it works
try:
    test = datasource.embeddings_dict
    logging.info('✅ Verification: embeddings_dict accessible with %d keys', len(test))
except Exception as e:
    logging.warn('⚠️ Warning: embeddings_dict access failed: %s', e)
# ===== END FIX =====


devices = jax.local_devices()

train_iter = datasource.create_iterator(
    datasource.train_ids,
    flatten=True,
    shuffle=True,
    batch_size=train_config.batch_size,
    prefetch_size=3,
    shuffle_buffer_size=train_config.shuffle_buffer_size,
    devices=devices,
)

def shuffled(l):
  import random as r
  import copy
  l = copy.copy(l)
  r.shuffle(l)
  return l

train_eval_iter = datasource.create_iterator(
    shuffled(datasource.train_ids), batch_size=0, devices=devices)
val_eval_iter = datasource.create_iterator(
    shuffled(datasource.val_ids), batch_size=0, devices=devices)

#Training###############
# @title Initialize model
# @markdown Defines the model and initializes its parameters.

from flax.training import checkpoints
from hypernerf import models
from hypernerf import model_utils
from hypernerf import schedules
from hypernerf import training

# @markdown Restore a checkpoint if one exists.
restore_checkpoint = False  # @param{type:'boolean'}

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

if restore_checkpoint:
  logging.info('Restoring checkpoint from %s', checkpoint_dir)
  state = checkpoints.restore_checkpoint(checkpoint_dir, state)

# ===== THAY ĐỔI: state.step thay vì state.optimizer.state.step =====
step = state.step + 1
state = jax_utils.replicate(state, devices=devices)
del params

#eval############
# @title Define pmapped functions
# @markdown This parallelizes the training and evaluation step functions using `jax.pmap`.

import functools
from hypernerf import evaluation

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
    _model_fn,
    in_axes=(0, 0, 0, 0, 0),
    devices=devices_to_use,
    axis_name='batch',
)

render_fn = functools.partial(evaluation.render_image,
                              model_fn=pmodel_fn,
                              device_count=len(devices),
                              chunk=eval_config.chunk)

train_step = functools.partial(
    training.train_step,
    model,
    elastic_reduce_method=train_config.elastic_reduce_method,
    elastic_loss_type=train_config.elastic_loss_type,
    use_elastic_loss=train_config.use_elastic_loss,
    use_background_loss=train_config.use_background_loss,
    use_warp_reg_loss=train_config.use_warp_reg_loss,
    use_hyper_reg_loss=train_config.use_hyper_reg_loss,
)

ptrain_step = jax.pmap(
    train_step,
    axis_name='batch',
    devices=devices,
    in_axes=(0, 0, 0, None),
    donate_argnums=(2,),
)

##########visual########

import mediapy
from hypernerf import utils
from hypernerf import visualization as viz

print_every_n_iterations = 100  # @param{type:'number'}
visualize_results_every_n_iterations = 500  # @param{type:'number'}
save_checkpoint_every_n_iterations = 1000  # @param{type:'number'}

logging.info('Starting training')
rng = rng + jax.process_index()
keys = random.split(rng, len(devices))
time_tracker = utils.TimeTracker()
time_tracker.tic('data', 'total')

# ===== PROGRESS TRACKING =====
import time
from datetime import datetime, timedelta

start_time = time.time()
losses_history = []
checkpoint_losses = {}
# ===== END SETUP =====

for step, batch in zip(range(step, train_config.max_steps + 1), train_iter):
  time_tracker.toc('data')
  
  scalar_params = scalar_params.replace(
      learning_rate=learning_rate_sched(step),
      elastic_loss_weight=elastic_loss_weight_sched(step))
  
  nerf_alpha = jax_utils.replicate(nerf_alpha_sched(step), devices)
  warp_alpha = jax_utils.replicate(warp_alpha_sched(step), devices)
  hyper_alpha = jax_utils.replicate(hyper_alpha_sched(step), devices)
  hyper_sheet_alpha = jax_utils.replicate(
      hyper_sheet_alpha_sched(step), devices)
  
  state = state.replace(
      nerf_alpha=nerf_alpha,
      warp_alpha=warp_alpha,
      hyper_alpha=hyper_alpha,
      hyper_sheet_alpha=hyper_sheet_alpha)

  with time_tracker.record_time('train_step'):
    state, stats, keys, _ = ptrain_step(keys, state, batch, scalar_params)
    time_tracker.toc('total')

  # ===== PROGRESS TRACKING =====
  elapsed_time = time.time() - start_time
  steps_completed = step - (step - 1)  # Always 1 per iteration
  progress_percent = (step / train_config.max_steps) * 100
  
  # Average time per step
  avg_time_per_step = elapsed_time / step
  remaining_steps = train_config.max_steps - step
  eta_seconds = remaining_steps * avg_time_per_step
  eta_time = datetime.now() + timedelta(seconds=eta_seconds)
  
  # Track loss
  total_loss = stats['coarse'].get('loss/total', 0.0).mean() if isinstance(stats['coarse'].get('loss/total'), jnp.ndarray) else stats['coarse'].get('loss/total', 0.0)
  losses_history.append(float(total_loss))
  
  # ===== END TRACKING =====

  if step % print_every_n_iterations == 0:
    # ===== PROGRESS DISPLAY =====
    progress_bar = "█" * int(progress_percent / 2) + "░" * (50 - int(progress_percent / 2))
    logging.info(
        '[%s] Step %d/%d (%.1f%%) | Loss: %.4f | ETA: %s (%s remaining)',
        progress_bar,
        step,
        train_config.max_steps,
        progress_percent,
        float(losses_history[-1]) if losses_history else 0.0,
        eta_time.strftime('%H:%M'),
        str(timedelta(seconds=int(eta_seconds))).split('.')[0]
    )
    # ===== END PROGRESS DISPLAY =====
    
    logging.info(
        'step=%d, warp_alpha=%.04f, hyper_alpha=%.04f, hyper_sheet_alpha=%.04f, %s',
        step, 
        warp_alpha_sched(step), 
        hyper_alpha_sched(step), 
        hyper_sheet_alpha_sched(step), 
        time_tracker.summary_str('last'))
    coarse_metrics_str = ', '.join(
        [f'{k}={v.mean():.04f}' for k, v in stats['coarse'].items()])
    fine_metrics_str = ', '.join(
        [f'{k}={v.mean():.04f}' for k, v in stats['fine'].items()])
    logging.info('\tcoarse metrics: %s', coarse_metrics_str)
    if 'fine' in stats:
      logging.info('\tfine metrics: %s', fine_metrics_str)
  
  if step % visualize_results_every_n_iterations == 0:
    print(f'[step={step}] Training set visualization')
    eval_batch = next(train_eval_iter)
    render = render_fn(state, eval_batch, rng=rng)
    rgb = render['rgb']
    acc = render['acc']
    depth_exp = render['depth']
    depth_med = render['med_depth']
    rgb_target = eval_batch['rgb']
    depth_med_viz = viz.colorize(depth_med, cmin=datasource.near, cmax=datasource.far)
    mediapy.show_images([rgb_target, rgb, depth_med_viz],
                        titles=['GT RGB', 'Pred RGB', 'Pred Depth'])

    print(f'[step={step}] Validation set visualization')
    eval_batch = next(val_eval_iter)
    render = render_fn(state, eval_batch, rng=rng)
    rgb = render['rgb']
    acc = render['acc']
    depth_exp = render['depth']
    depth_med = render['med_depth']
    rgb_target = eval_batch['rgb']
    depth_med_viz = viz.colorize(depth_med, cmin=datasource.near, cmax=datasource.far)
    mediapy.show_images([rgb_target, rgb, depth_med_viz],
                       titles=['GT RGB', 'Pred RGB', 'Pred Depth'])

  if step % save_checkpoint_every_n_iterations == 0:
    training.save_checkpoint(checkpoint_dir, state)
    checkpoint_losses[step] = float(losses_history[-1]) if losses_history else 0.0
    logging.info('✅ Checkpoint saved at step %d (loss: %.4f)', step, checkpoint_losses[step])

  time_tracker.tic('data', 'total')

# ===== TRAINING SUMMARY =====
total_time = time.time() - start_time
logging.info('=' * 80)
logging.info('🎉 TRAINING COMPLETED!')
logging.info('Total time: %s', str(timedelta(seconds=int(total_time))).split('.')[0])
logging.info('Average loss: %.4f', np.mean(losses_history) if losses_history else 0.0)
logging.info('Final loss: %.4f', losses_history[-1] if losses_history else 0.0)
logging.info('Checkpoints saved: %s', list(checkpoint_losses.keys()))
logging.info('=' * 80)
# ===== END SUMMARY =====

logging.info('Training completed!')
