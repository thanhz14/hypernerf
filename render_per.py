import orbax.checkpoint
import jax
import jax.numpy as jnp
import numpy as np
from PIL import Image
import imageio
from flax.training import train_state
from hypernerf import models

# --- 1. Load model và checkpoint ---
ckpt_dir = "hypernerf/nerfies/hypernerf_experiments/capture1/exp1/checkpoints"

# Embeddings_dict mặc định cho render (giả sử chỉ cần 1 embedding mỗi loại)
embeddings_dict = {
    'warp': [0],
    'appearance': [0],
    'camera': [0],
    'time': [0]
}
near = 2.0
far = 6.0
batch_size = 480 * 640  # số pixel ảnh

# Khởi tạo random key
key = jax.random.PRNGKey(0)

# Khởi tạo model và params đúng config lúc training
model, params = models.construct_nerf(
    key,
    batch_size=batch_size,
    embeddings_dict=embeddings_dict,
    near=near,
    far=far
)
state = train_state.TrainState.create(apply_fn=model.apply, params=params, tx=None)
checkpointer = orbax.checkpoint.PyTreeCheckpointer()
state_restored = checkpointer.restore(ckpt_dir, item=state)  # Sẽ tự phát hiện manifest.ocdbt
params = state_restored.params

# --- 2. Tạo các poses cho video quay quanh scene ---
def create_poses_for_video(num_frames=120, radius=4.0, height=0.0):
    angles = np.linspace(0, 2*np.pi, num_frames, endpoint=False)
    poses = []
    for theta in angles:
        pose = np.eye(4)
        pose[0, 3] = radius * np.cos(theta)
        pose[2, 3] = radius * np.sin(theta)
        pose[1, 3] = height
        poses.append(pose)
    return poses

# --- 3. Hàm tạo rays và metadata như lúc train ---
def create_rays_and_metadata(height, width, focal, pose, embeddings_dict):
    i, j = np.meshgrid(np.arange(width), np.arange(height), indexing='xy')
    dirs = np.stack([(i - width * 0.5)/focal,
                     -(j - height * 0.5)/focal,
                     -np.ones_like(i)], axis=-1)
    rays_d = dirs @ pose[:3, :3].T   # (H,W,3)
    rays_o = np.broadcast_to(pose[:3, 3], rays_d.shape)
    rays = {
        'origins': jnp.array(rays_o.reshape((-1, 3))),
        'directions': jnp.array(rays_d.reshape((-1, 3))),
        'viewdirs': jnp.array(rays_d.reshape((-1, 3))) / (
            np.linalg.norm(rays_d.reshape((-1, 3)), axis=-1, keepdims=True) + 1e-8),
    }
    metadata = {k: jnp.zeros((height*width, 1), dtype=jnp.float32 if k=='time' else jnp.uint32)
                for k in embeddings_dict}
    return rays, metadata

# --- 4. Render và xuất video ---
img_height, img_width = 480, 640
focal = 0.5 * img_width / np.tan(0.5 * 50 * np.pi / 180)
poses = create_poses_for_video(num_frames=120, radius=4.0, height=0.0)

writer = imageio.get_writer('rendered_video.mp4', fps=30)

for pose in poses:
    rays, metadata = create_rays_and_metadata(img_height, img_width, focal, pose, embeddings_dict)
    extra_params = {'nerf_alpha': 0.0, 'warp_alpha': 0.0, 'hyper_alpha': 0.0, 'hyper_sheet_alpha': 0.0}
    out = model.apply({'params': params}, {**rays, 'metadata': metadata}, extra_params=extra_params)
    rgb = np.array(out['fine']['rgb']).reshape((img_height, img_width, 3))
    frame = np.clip(rgb * 255, 0, 255).astype(np.uint8)
    writer.append_data(frame)
    print("Rendered frame from camera pose")

writer.close()
print("Video đã lưu tại rendered_video.mp4")
