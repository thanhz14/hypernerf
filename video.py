import imageio
import os

output_dir = "outputs/frames"
video_path = "outputs/rendered_video.gif"  # GIF không cần ffmpeg
fps = 30

frame_files = sorted([
    os.path.join(output_dir, fname)
    for fname in os.listdir(output_dir)
    if fname.startswith("frame_") and fname.endswith(".png")
])

frames = [imageio.imread(fname) for fname in frame_files]
imageio.mimsave(video_path, frames, fps=fps)
print(f"✅ Saved GIF video to {video_path}")