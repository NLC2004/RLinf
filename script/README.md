# RLinf Script Utilities

## `render_value_curve_video.py`

把 replay buffer 里的单条轨迹渲染成视频，并在每一帧下方同步显示 value 曲线（当前 step 用红色竖线 + 红点高亮）。

---

### 功能

- 从 `TrajectoryReplayBuffer` 读取轨迹
- 提取相机图像（默认 `main_images`）
- 优先使用 `trajectory.prev_values` 作为 value 序列
- 若没有 `prev_values`，自动回退为根据 reward 计算 MC return
- 输出拼接视频（上：环境画面，下：value 曲线）

---

### 依赖

通常 RLinf 环境已有这些依赖：

- `numpy`
- `torch`
- `matplotlib`
- `imageio`
- `ffmpeg`（用于 mp4 编码）

若缺失可安装：

```bash
pip install imageio imageio-ffmpeg matplotlib
```

---

### 用法

#### 1) 按轨迹索引（默认）

```bash
python RLinf/script/render_value_curve_video.py \
  --replay_dir RLinf/logs/<run_name>/replay_buffer/rank_0 \
  --output RLinf/script/value_curve.mp4 \
  --trajectory_index 0 \
  --batch_idx 0 \
  --camera main_images
```

#### 2) 按轨迹 ID 指定

```bash
python RLinf/script/render_value_curve_video.py \
  --replay_dir RLinf/logs/<run_name>/replay_buffer/rank_0 \
  --output RLinf/script/value_curve.mp4 \
  --trajectory_id 123
```

---

### 参数说明

- `--replay_dir`：replay buffer 目录（必填）
- `--output`：输出视频路径 `.mp4`（必填）
- `--trajectory_id`：轨迹 ID（可选，优先级高）
- `--trajectory_index`：轨迹列表下标（默认 0）
- `--batch_idx`：轨迹内 batch 维索引（默认 0）
- `--camera`：相机 key，可选：
  - `main_images`（默认）
  - `wrist_images`
  - `extra_view_images`
- `--fps`：输出视频帧率（默认 20）
- `--gamma`：当回退到 MC return 时使用的折扣因子（默认 0.99）

---

### 输出效果

每一帧由两部分组成：

1. 当前环境图像（轨迹 step 对应）
2. 整体 value 曲线 + 当前 step 红色高亮

便于观察“动作进展”与“价值变化”的对应关系。

---

### 常见问题

#### 1) 报错找不到图像 key
确认轨迹的 `curr_obs` 中是否包含 `main_images/wrist_images/extra_view_images`，或者切换 `--camera`。

#### 2) 没有 `prev_values`
这是正常情况，脚本会自动用 reward 计算 MC return 作为替代 value 序列。

#### 3) mp4 编码失败
通常是 ffmpeg 不可用。可安装系统 ffmpeg，或安装 `imageio-ffmpeg`。
