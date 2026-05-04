#!/usr/bin/env python3
# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Render trajectory video with step-synced value curve overlay.

This script loads one trajectory from an RLinf replay buffer and exports an MP4
where each frame contains:
1) environment image (top)
2) value-vs-step curve with current step highlighted in red (bottom)

Usage:
    python RLinf/script/render_value_curve_video.py \
        --replay_dir RLinf/logs/<run>/replay_buffer/rank_0 \
        --output RLinf/script/value_curve.mp4
"""

import argparse
from pathlib import Path
from typing import Optional

import imageio
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch

from rlinf.data.replay_buffer import TrajectoryReplayBuffer

matplotlib.use("Agg")


def _to_numpy(x):
    if torch.is_tensor(x):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def _extract_image(obs_dict: dict, step_idx: int, batch_idx: int, camera_key: str) -> np.ndarray:
    img = obs_dict.get(camera_key)
    if img is None:
        for fallback in ["main_images", "wrist_images", "extra_view_images"]:
            img = obs_dict.get(fallback)
            if img is not None:
                break
    if img is None:
        raise ValueError(f"No camera image found for key={camera_key}")

    img = _to_numpy(img)

    if img.ndim == 5:  # [T, B, C, H, W] or [T, B, H, W, C]
        img = img[step_idx, batch_idx]
    elif img.ndim == 4:  # [T, C, H, W] or [T, H, W, C]
        img = img[step_idx]
    elif img.ndim != 3:
        raise ValueError(f"Unsupported image shape: {img.shape}")

    if img.ndim == 3 and img.shape[0] in (1, 3, 4):
        img = np.transpose(img, (1, 2, 0))

    if img.dtype in (np.float32, np.float64):
        if img.max() <= 1.0:
            img = (img * 255.0).clip(0, 255).astype(np.uint8)
        else:
            img = img.clip(0, 255).astype(np.uint8)

    if img.ndim == 3 and img.shape[-1] == 1:
        img = np.repeat(img, 3, axis=-1)
    if img.ndim == 2:
        img = np.stack([img, img, img], axis=-1)

    return img


def _extract_values(traj, T: int, batch_idx: int, gamma: float) -> tuple[np.ndarray, str]:
    if traj.prev_values is not None:
        v = _to_numpy(traj.prev_values)
        if v.ndim == 3:
            v = v[:, batch_idx, 0]
        elif v.ndim == 2:
            v = v[:, batch_idx]
        else:
            raise ValueError(f"Unsupported prev_values shape: {v.shape}")
        if len(v) < T:
            raise ValueError(f"prev_values length {len(v)} < trajectory length {T}")
        return v[:T].astype(np.float32), "prev_values"

    if traj.rewards is None:
        raise ValueError("Neither prev_values nor rewards found in trajectory")

    r = _to_numpy(traj.rewards)
    if r.ndim == 3:
        r = r[:, batch_idx, 0]
    elif r.ndim == 2:
        r = r[:, batch_idx]
    else:
        raise ValueError(f"Unsupported rewards shape: {r.shape}")
    r = r[:T].astype(np.float32)

    returns = np.zeros_like(r)
    running = 0.0
    for t in range(T - 1, -1, -1):
        running = float(r[t]) + gamma * running
        returns[t] = running
    return returns, f"mc_return(gamma={gamma})"


def _resize_to_width(img: np.ndarray, target_w: int) -> np.ndarray:
    if img.shape[1] == target_w:
        return img
    scale = target_w / img.shape[1]
    target_h = max(1, int(round(img.shape[0] * scale)))
    fig = plt.figure(figsize=(target_w / 100, target_h / 100), dpi=100)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.imshow(img)
    ax.axis("off")
    fig.canvas.draw()
    buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
    out = buf.reshape(fig.canvas.get_width_height()[1], fig.canvas.get_width_height()[0], 4)[..., :3]
    plt.close(fig)
    return out


def _render_curve_panel(values: np.ndarray, step_idx: int, title: str, width_px: int = 960, height_px: int = 320) -> np.ndarray:
    fig, ax = plt.subplots(figsize=(width_px / 100, height_px / 100), dpi=100)
    x = np.arange(len(values))

    ax.plot(x, values, color="#2C7FB8", linewidth=2.0)
    ax.axvline(step_idx, color="#D62728", linestyle="--", linewidth=1.8)
    ax.scatter([step_idx], [values[step_idx]], color="#D62728", s=42, zorder=3)

    ax.set_xlabel("Step")
    ax.set_ylabel("Value")
    ax.set_title(title)
    ax.grid(alpha=0.25, linestyle="--", linewidth=0.7)

    fig.tight_layout()
    fig.canvas.draw()
    buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
    panel = buf.reshape(fig.canvas.get_width_height()[1], fig.canvas.get_width_height()[0], 4)[..., :3]
    plt.close(fig)
    return panel


def render_video(
    replay_dir: str,
    output_path: str,
    trajectory_id: Optional[int],
    trajectory_index: int,
    batch_idx: int,
    camera_key: str,
    fps: int,
    gamma: float,
):
    replay_path = Path(replay_dir)
    if not replay_path.exists():
        raise ValueError(f"Replay buffer directory not found: {replay_dir}")

    buffer = TrajectoryReplayBuffer(
        auto_save=True,
        auto_save_path=str(replay_path),
        enable_cache=True,
        cache_size=3,
    )
    buffer.load_checkpoint(str(replay_path))

    if buffer.size == 0:
        raise ValueError("Replay buffer is empty")

    if trajectory_id is not None:
        if trajectory_id not in buffer._trajectory_id_list:
            raise ValueError(
                f"trajectory_id={trajectory_id} not found, valid head={buffer._trajectory_id_list[:10]}"
            )
        traj_id = trajectory_id
    else:
        if trajectory_index < 0 or trajectory_index >= buffer.size:
            raise ValueError(f"trajectory_index must be in [0, {buffer.size - 1}]")
        traj_id = buffer._trajectory_id_list[trajectory_index]

    traj_info = buffer._trajectory_index[traj_id]
    traj_shape = traj_info["shape"]
    T, B = traj_shape[:2]
    if batch_idx < 0 or batch_idx >= B:
        raise ValueError(f"batch_idx must be in [0, {B - 1}] for this trajectory")

    traj = buffer._load_trajectory(traj_id, traj_info["model_weights_id"])

    values, source = _extract_values(traj, T=T, batch_idx=batch_idx, gamma=gamma)

    output_path = str(Path(output_path))
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    print(f"Loaded replay buffer: {replay_dir}")
    print(f"Trajectory id: {traj_id}, shape: {traj_shape}, batch_idx: {batch_idx}")
    print(f"Value source: {source}")
    print(f"Writing video to: {output_path}")

    with imageio.get_writer(output_path, fps=fps, codec="libx264", quality=8) as writer:
        for step_idx in range(T):
            frame = _extract_image(traj.curr_obs, step_idx, batch_idx, camera_key)
            curve = _render_curve_panel(
                values,
                step_idx,
                title=f"Trajectory {traj_id} | step {step_idx}/{T - 1} | {source}",
                width_px=max(frame.shape[1], 640),
                height_px=320,
            )

            frame = _resize_to_width(frame, curve.shape[1])
            merged = np.concatenate([frame, curve], axis=0)
            writer.append_data(merged)

    print("Done.")


def main():
    parser = argparse.ArgumentParser(description="Render value-curve synchronized trajectory video")
    parser.add_argument("--replay_dir", type=str, required=True, help="Replay buffer directory, e.g. logs/.../replay_buffer/rank_0")
    parser.add_argument("--output", type=str, required=True, help="Output video path (.mp4)")
    parser.add_argument("--trajectory_id", type=int, default=None, help="Exact trajectory ID to render")
    parser.add_argument("--trajectory_index", type=int, default=0, help="Fallback: index in trajectory list when --trajectory_id not set")
    parser.add_argument("--batch_idx", type=int, default=0, help="Batch index inside trajectory")
    parser.add_argument(
        "--camera",
        type=str,
        default="main_images",
        choices=["main_images", "wrist_images", "extra_view_images"],
        help="Camera key in observation dict",
    )
    parser.add_argument("--fps", type=int, default=20, help="Output video FPS")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount used only when fallback to Monte-Carlo returns")

    args = parser.parse_args()

    render_video(
        replay_dir=args.replay_dir,
        output_path=args.output,
        trajectory_id=args.trajectory_id,
        trajectory_index=args.trajectory_index,
        batch_idx=args.batch_idx,
        camera_key=args.camera,
        fps=args.fps,
        gamma=args.gamma,
    )


if __name__ == "__main__":
    main()
