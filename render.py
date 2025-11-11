import sys
import os
import jax
import jax.numpy as jnp
from flax.training import checkpoints
from flax import jax_utils
import optax
import numpy as np
from pathlib import Path
from absl import logging

import hypernerf
from hypernerf import models, model_utils, datasets, configs, schedules, evaluation, visualization as viz, image_utils, utils

import mediapy
import gin

# ----- ĐƯỜNG DẪN -----
train_dir = 'hypernerf/nerfies/hypernerf_experiments/capture1/exp1'
data_dir = r'D:/Acer/Code/Dat_pr/hypernerf/nerfies/captures/capture1'
checkpoint_dir = Path(train_dir, 'checkpoints')
config_path = Path(train_dir, 'config.gin')

# ----- LOAD CONFIG -----
with open(config_path, 'r') as f:
    gin.parse_config(f.read())

exp_config = configs.ExperimentConfig()
train_config = configs.TrainConfig()
eval_config = configs.EvalConfig()

# ----- TÁI TẠO DATASOURCE + EMBEDDINGS_DICT EXACTLY NHƯ TRAIN -----
datasource = exp_config.datasource_cls(
    data_dir=data_dir,
    image_scale=exp_config.image_scale,
    shuffle_pixels=True,
    camera_type='json',
    test_camera_trajectory='orbit-mild',
    use_warp_id=True,
    use_appearance_id=True,
    use_camera_id=True,
)

# (Tuỳ training script, nếu bạn đã custom phần này như sau thì phải LÀM Y HỆT)
all_ids = sorted(list(set(datasource.train_ids + datasource.val_ids)))
embeddings_dict = {
    'warp': {img_id: idx for idx, img_id in enumerate(all_ids)},
    'appearance': {img_id: idx for idx, img_id in enumerate(all_ids)},
    'camera': {img_id: 0 for img_id in all_ids}
}
datasource.__dict__['_embeddings_dict'] = embeddings_dict
datasource.__dict__['embeddings_dict'] = embeddings_dict

# ----- BUILD MODEL + RESTORE CHECKPOINT -----
rng = jax.random.PRNGKey(exp_config.random_seed)
rng, key = jax.random.split(rng)

learning_rate_sched = schedules.from_config(train_config.lr_schedule)
nerf_alpha_sched = schedules.from_config(train_config.nerf_alpha_schedule)
warp_alpha_sched = schedules.from_config(train_config.warp_alpha_schedule)
hyper_alpha_sched = schedules.from_config(train_config.hyper_alpha_schedule)
hyper_sheet_alpha_sched = schedules.from_config(train_config.hyper_sheet_alpha_schedule)

params = {}
model, params['model'] = models.construct_nerf(
    key,
    batch_size=train_config.batch_size,
    embeddings_dict=embeddings_dict,
    near=datasource.near,
    far=datasource.far,
)
tx = optax.adam(learning_rate=learning_rate_sched(0))

state = model_utils.TrainState.create(
    apply_fn=model.apply,
    params=params,
    tx=tx,
    nerf_alpha=nerf_alpha_sched(0),
    warp_alpha=warp_alpha_sched(0),
    hyper_alpha=hyper_alpha_sched(0),
    hyper_sheet_alpha=hyper_sheet_alpha_sched(0)
)
state = checkpoints.restore_checkpoint(checkpoint_dir, state)
del params
print('✅ Checkpoint loaded')

# ----- LOAD CAMERA PATH -----
camera_dir = Path(data_dir, 'camera-paths/orbit-mild') # chỉnh lại nếu camera path khác!
test_camera_paths = datasource.glob_cameras(camera_dir)
test_cameras = utils.parallel_map(datasource.load_camera, test_camera_paths, show_pbar=True)

# ----- RENDER CONFIG -----
chunk_size = getattr(eval_config, 'chunk', 4096)
render_fn = jax.jit(lambda key0, key1, params, rays_dict, extra_params:
    model.apply({'params': params}, rays_dict, extra_params=extra_params, rngs={'coarse': key0, 'fine': key1}, mutable=False))
print('✅ Render function ready')

# ----- RENDERING LOOP -----
rng = rng + jax.process_index()
results = []

for i, camera in enumerate(test_cameras):
    print(f'Rendering frame {i+1}/{len(test_cameras)}...')
    batch = datasets.camera_to_rays(camera)
    orig_h, orig_w = batch['origins'].shape[0], batch['origins'].shape[1]
    origins = batch['origins'].reshape(-1, 3)
    directions = batch['directions'].reshape(-1, 3)
    num_rays = origins.shape[0]
    rgb_parts = []
    depth_parts = []
    for start in range(0, num_rays, chunk_size):
        end = min(start + chunk_size, num_rays)
        rays_dict = {
            'origins': origins[start:end],
            'directions': directions[start:end],
            'metadata': {
                'appearance': jnp.full((end-start, 1), 0, dtype=jnp.uint32),
                'warp': jnp.full((end-start, 1), 0, dtype=jnp.uint32),
            }
        }
        rng, key0, key1 = jax.random.split(rng, 3)
        out = render_fn(key0, key1,
                        state.params['model'],
                        rays_dict,
                        state.extra_params())
        rgb_parts.append(np.array(out['coarse']['rgb']))
        depth_parts.append(np.array(out['coarse']['med_depth']))
    rgb = np.concatenate(rgb_parts, axis=0).reshape(orig_h, orig_w, 3)
    depth = np.concatenate(depth_parts, axis=0).reshape(orig_h, orig_w)
    results.append((rgb, depth))

    # --- Visualize ---
    depth_viz = viz.colorize(depth, cmin=datasource.near, cmax=datasource.far, invert=True)
    mediapy.show_images([rgb, depth_viz], titles=['RGB', 'Depth'])
     # ==== Lưu ảnh ====
    import imageio
    import os

    output_dir = "outputs/frames"
    os.makedirs(output_dir, exist_ok=True)

    imageio.imwrite(os.path.join(output_dir, f"frame_{i:03d}.png"), image_utils.image_to_uint8(rgb))
    imageio.imwrite(os.path.join(output_dir, f"depth_{i:03d}.png"), image_utils.image_to_uint8(depth_viz))
    print(f"✅ Saved {os.path.join(output_dir, f'frame_{i:03d}.png')} and depth_{i:03d}.png")

# ----- SAVE VIDEO -----
fps = 30
frames = []
for rgb, depth in results:
    depth_viz = viz.colorize(depth, cmin=datasource.near, cmax=datasource.far, invert=True)
    frame = np.concatenate([rgb, depth_viz], axis=1)
    frames.append(image_utils.image_to_uint8(frame))
mediapy.show_video(frames, fps=fps)
print('✅ Rendering complete!')