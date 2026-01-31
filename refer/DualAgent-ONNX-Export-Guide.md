# Dual Agent ONNX Export Guide

本文档介绍 `dual_agent_export_onnx.py` 的工作原理、导出内容及部署使用方法。

## 1. 概述

### 1.1 背景

Dual Agent 是一个双策略架构的人形机器人控制系统：
- **上半身策略 (Upper Body Policy)**：控制手臂（14个关节），用于抱箱子等操作任务
- **下半身策略 (Lower Body Policy)**：控制腿部和腰部（15个关节），用于运动跟踪或速度跟踪

两个策略独立训练，但在部署时合并为一个 ONNX 模型，输出完整的 29 自由度动作。

### 1.2 导出模式

| 模式 | 输出文件 | 使用场景 |
|------|----------|----------|
| **Basic Mode** | `dual_agent_combined.onnx` | 速度跟踪任务，或需要外部提供轨迹的场景 |
| **Motion Mode** | `dual_agent_motion.onnx` | 运动跟踪任务，轨迹嵌入模型内部 |

---

## 2. 模型架构

### 2.1 Basic Mode (`dual_agent_combined.onnx`)

```
┌─────────────────────────────────────────────────────────────────┐
│                    dual_agent_combined.onnx                     │
├─────────────────────────────────────────────────────────────────┤
│  Inputs:                                                        │
│    ├─ upper_obs: float32[1, 480]    上半身观测                   │
│    └─ lower_obs: float32[1, 121]    下半身观测 (tracking)        │
│                  float32[1, 99]     下半身观测 (velocity)        │
│                                                                 │
│  Internal:                                                      │
│    ├─ upper_action = upper_actor(upper_obs)  → [1, 29]         │
│    ├─ lower_action = lower_actor(lower_obs)  → [1, 15]         │
│    └─ Merge:                                                    │
│        combined[0:15]  = lower_action[0:15]                     │
│        combined[15:29] = upper_action[15:29]                    │
│                                                                 │
│  Output:                                                        │
│    └─ actions: float32[1, 29]       合并后的关节动作             │
└─────────────────────────────────────────────────────────────────┘
```

### 2.2 Motion Mode (`dual_agent_motion.onnx`)

```
┌─────────────────────────────────────────────────────────────────┐
│                    dual_agent_motion.onnx                       │
├─────────────────────────────────────────────────────────────────┤
│  Inputs:                                                        │
│    ├─ upper_obs: float32[1, 480]    上半身观测                   │
│    ├─ lower_obs: float32[1, 121]    下半身观测                   │
│    └─ time_step: int64[1, 1]        当前时间步索引               │
│                                                                 │
│  Embedded Buffers (轨迹数据):                                    │
│    ├─ joint_pos:      [T, 29]       目标关节位置                 │
│    ├─ joint_vel:      [T, 29]       目标关节速度                 │
│    ├─ body_pos_w:     [T, N, 3]     目标 body 位置（世界坐标）   │
│    ├─ body_quat_w:    [T, N, 4]     目标 body 姿态（四元数）     │
│    ├─ body_lin_vel_w: [T, N, 3]     目标 body 线速度            │
│    └─ body_ang_vel_w: [T, N, 3]     目标 body 角速度            │
│                                                                 │
│  Outputs (7个):                                                  │
│    ├─ actions:        [1, 29]       合并后的关节动作             │
│    ├─ joint_pos:      [1, 29]       当前时间步的目标关节位置     │
│    ├─ joint_vel:      [1, 29]       当前时间步的目标关节速度     │
│    ├─ body_pos_w:     [1, N, 3]     当前时间步的目标 body 位置   │
│    ├─ body_quat_w:    [1, N, 4]     当前时间步的目标 body 姿态   │
│    ├─ body_lin_vel_w: [1, N, 3]     当前时间步的目标 body 线速度 │
│    └─ body_ang_vel_w: [1, N, 3]     当前时间步的目标 body 角速度 │
└─────────────────────────────────────────────────────────────────┘
```

---

## 3. 关节映射

### 3.1 G1 机器人 29 自由度定义

```python
# 下半身关节 (索引 0-14, 共15个)
LOWER_BODY_JOINTS = [
    "left_hip_pitch_joint",      # 0
    "left_hip_roll_joint",       # 1
    "left_hip_yaw_joint",        # 2
    "left_knee_joint",           # 3
    "left_ankle_pitch_joint",    # 4
    "left_ankle_roll_joint",     # 5
    "right_hip_pitch_joint",     # 6
    "right_hip_roll_joint",      # 7
    "right_hip_yaw_joint",       # 8
    "right_knee_joint",          # 9
    "right_ankle_pitch_joint",   # 10
    "right_ankle_roll_joint",    # 11
    "waist_yaw_joint",           # 12
    "waist_roll_joint",          # 13
    "waist_pitch_joint",         # 14
]

# 上半身关节 (索引 15-28, 共14个)
UPPER_BODY_JOINTS = [
    "left_shoulder_pitch_joint",  # 15
    "left_shoulder_roll_joint",   # 16
    "left_shoulder_yaw_joint",    # 17
    "left_elbow_joint",           # 18
    "left_wrist_roll_joint",      # 19
    "left_wrist_pitch_joint",     # 20
    "left_wrist_yaw_joint",       # 21
    "right_shoulder_pitch_joint", # 22
    "right_shoulder_roll_joint",  # 23
    "right_shoulder_yaw_joint",   # 24
    "right_elbow_joint",          # 25
    "right_wrist_roll_joint",     # 26
    "right_wrist_pitch_joint",    # 27
    "right_wrist_yaw_joint",      # 28
]
```

### 3.2 动作合并逻辑

```python
# 策略输出
upper_action = upper_policy(upper_obs)  # shape: [1, 29]，但只用 [15:29]
lower_action = lower_policy(lower_obs)  # shape: [1, 15]

# 合并
combined_action = np.zeros(29)
combined_action[0:15] = lower_action[0:15]    # 下半身动作
combined_action[15:29] = upper_action[15:29]  # 上半身动作
```

---

## 4. 观测构建

### 4.1 上半身观测 (Upper Body Observation)

**维度**: 480 = 96 × 5 (5步历史)

每步观测 (96维):
| 分量 | 维度 | 说明 | 预处理 |
|------|------|------|--------|
| `base_ang_vel` | 3 | 机器人基座角速度 (body frame) | × 0.2, clip(-100, 100) |
| `projected_gravity` | 3 | 重力在 body frame 的投影 | - |
| `velocity_commands` | 3 | 速度命令 [vx, vy, ω] | 对于 tracking 任务设为 [0,0,0] |
| `joint_pos_rel` | 29 | 相对默认位置的关节角度 | clip(-100, 100) |
| `joint_vel_rel` | 29 | 相对默认速度的关节角速度 | × 0.05, clip(-100, 100) |
| `last_action` | 29 | 上一步的动作 | clip(-12, 12) |

**历史拼接顺序**: `[t-4, t-3, t-2, t-1, t]`，展平为 480 维

```python
def build_upper_obs(robot_state, history_buffer):
    """
    robot_state: 当前机器人状态
    history_buffer: 存储过去 5 步观测的队列
    """
    current_obs = np.concatenate([
        robot_state.base_ang_vel * 0.2,           # 3
        robot_state.projected_gravity,             # 3
        np.array([0.0, 0.0, 0.0]),                 # 3 (velocity commands, tracking 任务为 0)
        np.clip(robot_state.joint_pos - DEFAULT_JOINT_POS, -100, 100),  # 29
        np.clip(robot_state.joint_vel * 0.05, -100, 100),               # 29
        np.clip(last_action, -12, 12),             # 29
    ])  # 总计 96
    
    history_buffer.append(current_obs)
    if len(history_buffer) > 5:
        history_buffer.pop(0)
    
    # 如果历史不足 5 步，用当前 obs 填充
    while len(history_buffer) < 5:
        history_buffer.insert(0, current_obs.copy())
    
    return np.concatenate(history_buffer)  # 480
```

### 4.2 下半身观测 - Tracking 任务 (Lower Body Observation)

**维度**: 121

| 分量 | 维度 | 说明 |
|------|------|------|
| `lower_body_command` | 30 | 目标关节位置(15) + 目标关节速度(15) |
| `motion_anchor_pos_b` | 3 | 目标 anchor (torso) 相对于当前 anchor 的位置 (body frame) |
| `motion_anchor_ori_b` | 6 | 目标 anchor 相对于当前 anchor 的姿态 (rotation matrix 前两列) |
| `projected_gravity` | 3 | 重力在 body frame 的投影 |
| `base_lin_vel` | 3 | 基座线速度 (body frame) |
| `base_ang_vel` | 3 | 基座角速度 (body frame) |
| `joint_pos` | 29 | 相对默认位置的关节角度 |
| `joint_vel` | 29 | 相对默认速度的关节角速度 |
| `actions` | 15 | 上一步的下半身动作 |

```python
def build_lower_obs_tracking(robot_state, trajectory, time_step, last_lower_action):
    """
    robot_state: 当前机器人状态
    trajectory: 参考轨迹数据
    time_step: 当前时间步索引
    last_lower_action: 上一步的下半身动作 [15]
    """
    # 从轨迹获取目标
    target_joint_pos = trajectory["joint_pos"][time_step][:15]  # 只取下半身
    target_joint_vel = trajectory["joint_vel"][time_step][:15]
    target_anchor_pos = trajectory["anchor_pos"][time_step]
    target_anchor_quat = trajectory["anchor_quat"][time_step]
    
    # 计算相对位姿
    anchor_pos_b = transform_to_body_frame(target_anchor_pos, robot_state)
    anchor_ori_b = compute_relative_rotation_6d(target_anchor_quat, robot_state.base_quat)
    
    lower_obs = np.concatenate([
        target_joint_pos,                          # 15
        target_joint_vel,                          # 15
        anchor_pos_b,                              # 3
        anchor_ori_b,                              # 6
        robot_state.projected_gravity,             # 3
        robot_state.base_lin_vel,                  # 3
        robot_state.base_ang_vel,                  # 3
        robot_state.joint_pos - DEFAULT_JOINT_POS, # 29
        robot_state.joint_vel,                     # 29
        last_lower_action,                         # 15
    ])  # 总计 121
    
    return lower_obs
```

### 4.3 下半身观测 - Velocity 任务 (Lower Body Observation)

**维度**: 99

| 分量 | 维度 | 说明 |
|------|------|------|
| `base_lin_vel` | 3 | 基座线速度 (body frame) |
| `base_ang_vel` | 3 | 基座角速度 (body frame), × 0.2 |
| `projected_gravity` | 3 | 重力在 body frame 的投影 |
| `velocity_commands` | 3 | 速度命令 [vx, vy, ω] |
| `joint_pos_rel` | 29 | 相对默认位置的关节角度 |
| `joint_vel_rel` | 29 | 相对默认速度的关节角速度, × 0.05 |
| `last_action` | 29 | 上一步的完整动作 |

---

## 5. 部署推理

### 5.1 Basic Mode 推理示例

```python
import onnxruntime as ort
import numpy as np

# 加载模型
session = ort.InferenceSession("dual_agent_combined.onnx")

# 初始化
history_buffer = []
last_action = np.zeros(29, dtype=np.float32)
last_lower_action = np.zeros(15, dtype=np.float32)

# 控制频率: 50Hz (dt = 0.02s)
# 仿真 decimation = 4, 仿真 dt = 0.005s
control_dt = 0.02

while running:
    # 1. 获取机器人状态
    robot_state = get_robot_state()
    
    # 2. 构建观测
    upper_obs = build_upper_obs(robot_state, history_buffer, last_action)
    lower_obs = build_lower_obs_tracking(robot_state, trajectory, time_step, last_lower_action)
    
    # 3. 推理
    outputs = session.run(
        ["actions"],
        {
            "upper_obs": upper_obs.reshape(1, -1).astype(np.float32),
            "lower_obs": lower_obs.reshape(1, -1).astype(np.float32),
        }
    )
    actions = outputs[0][0]  # [29]
    
    # 4. 动作后处理: action → target_joint_pos
    # actions 是相对于默认关节位置的偏移量，需要转换
    action_scale = 0.25  # G1 的 action scale
    target_joint_pos = DEFAULT_JOINT_POS + actions * action_scale
    
    # 5. 发送到机器人
    robot.set_joint_position_targets(target_joint_pos)
    
    # 6. 更新状态
    last_action = actions.copy()
    last_lower_action = actions[:15].copy()
    time_step += 1
    
    time.sleep(control_dt)
```

### 5.2 Motion Mode 推理示例

```python
import onnxruntime as ort
import numpy as np

# 加载模型 (轨迹已嵌入)
session = ort.InferenceSession("dual_agent_motion.onnx")

# 初始化
history_buffer = []
last_action = np.zeros(29, dtype=np.float32)
last_lower_action = np.zeros(15, dtype=np.float32)
time_step = 0

while running:
    # 1. 获取机器人状态
    robot_state = get_robot_state()
    
    # 2. 构建观测
    upper_obs = build_upper_obs(robot_state, history_buffer, last_action)
    lower_obs = build_lower_obs_tracking(robot_state, trajectory=None, time_step=time_step, last_lower_action=last_lower_action)
    # 注意: Motion Mode 下，lower_body_command 需要从模型输出的 joint_pos/joint_vel 构建
    # 或者使用上一步的输出作为当前步的 command 输入
    
    # 3. 推理
    outputs = session.run(
        ["actions", "joint_pos", "joint_vel", "body_pos_w", "body_quat_w", "body_lin_vel_w", "body_ang_vel_w"],
        {
            "upper_obs": upper_obs.reshape(1, -1).astype(np.float32),
            "lower_obs": lower_obs.reshape(1, -1).astype(np.float32),
            "time_step": np.array([[time_step]], dtype=np.int64),
        }
    )
    
    actions = outputs[0][0]           # [29] 合并动作
    target_joint_pos = outputs[1][0]  # [29] 当前时间步的参考关节位置 (来自轨迹)
    target_joint_vel = outputs[2][0]  # [29] 当前时间步的参考关节速度
    target_body_pos = outputs[3][0]   # [N, 3] 当前时间步的参考 body 位置
    target_body_quat = outputs[4][0]  # [N, 4] 当前时间步的参考 body 姿态
    
    # 4. 动作后处理
    action_scale = 0.25
    final_joint_pos = DEFAULT_JOINT_POS + actions * action_scale
    
    # 5. 发送到机器人
    robot.set_joint_position_targets(final_joint_pos)
    
    # 6. 可选: 使用 target_body_pos/quat 进行可视化或误差计算
    visualize_target(target_body_pos, target_body_quat)
    
    # 7. 更新状态
    last_action = actions.copy()
    last_lower_action = actions[:15].copy()
    time_step += 1
    
    # 时间步会在模型内部自动 clamp 到有效范围
    # 如需循环播放，可在外部处理: time_step = time_step % total_frames
```

---

## 6. 关键参数

### 6.1 控制参数

| 参数 | 值 | 说明 |
|------|-----|------|
| `control_dt` | 0.02s | 控制周期 (50Hz) |
| `sim_dt` | 0.005s | 仿真步长 |
| `decimation` | 4 | 每 4 个仿真步执行一次策略 |
| `action_scale` | 0.25 | 动作缩放因子 |

### 6.2 默认关节位置

```python
# G1 机器人默认关节位置 (抱箱姿态)
DEFAULT_JOINT_POS = np.array([
    # 下半身 (0-14)
    0.0, 0.0, 0.0,    # 左腿 hip (pitch, roll, yaw)
    0.0, 0.0, 0.0,    # 左腿 knee, ankle (pitch, roll)
    0.0, 0.0, 0.0,    # 右腿 hip
    0.0, 0.0, 0.0,    # 右腿 knee, ankle
    0.0, 0.0, 0.0,    # 腰部 (yaw, roll, pitch)
    # 上半身 (15-28) - 抱箱姿态
    -0.6, 1.0, -0.7,  # 左肩 (pitch, roll, yaw)
    -0.18,            # 左肘
    -1.2, 0.0, 0.6,   # 左腕 (roll, pitch, yaw)
    -0.6, -1.0, 0.7,  # 右肩
    -0.18,            # 右肘
    1.2, 0.0, -0.6,   # 右腕
])
```

---

## 7. ONNX 元数据

Motion Mode 导出的 ONNX 文件包含以下元数据（可通过 `onnx.load()` 读取）：

```python
import onnx

model = onnx.load("dual_agent_motion.onnx")
metadata = {prop.key: prop.value for prop in model.metadata_props}

# 可用字段:
# - model_type: "dual_agent_motion"
# - upper_obs_dim: "480"
# - lower_obs_dim: "121"
# - action_dim: "29"
# - lower_body_action_dim: "15"
# - upper_body_action_start: "15"
# - lower_body_joint_names: "left_hip_pitch_joint,..."
# - upper_body_joint_names: "left_shoulder_pitch_joint,..."
# - anchor_body_name: "torso_link"
# - trajectory_source: "overhurdle4.npz"
# - trajectory_total_frames: "..."
# - trajectory_dt: "0.02"
# - trajectory_duration_s: "..."
```

---

## 8. 导出命令

```bash
# Basic Mode (不嵌入轨迹)
python scripts/factoryIsaac/dual_agent_export_onnx.py \
    --upper_policy logs/rsl_rl/g1_box_transport/xxx/model_5000.pt \
    --lower_policy logs/rsl_rl/g1_track/xxx/model_5000.pt \
    --task tracking \
    --output_dir logs/dual_agent/exported

# Motion Mode (嵌入轨迹)
python scripts/factoryIsaac/dual_agent_export_onnx.py \
    --upper_policy logs/rsl_rl/g1_box_transport/xxx/model_5000.pt \
    --lower_policy logs/rsl_rl/g1_track/xxx/model_5000.pt \
    --task tracking \
    --embed_trajectory \
    --motion_file data/demo/holdthebox/overhurdle4.npz \
    --output_dir logs/dual_agent/exported
```

---

## 9. 常见问题

### Q1: 观测维度不匹配怎么办？

导出脚本会检查策略的输入维度，如果不匹配会报错。确保：
- 上半身策略输入维度 = 480 (96 × 5 历史)
- 下半身策略输入维度 = 121 (tracking) 或 99 (velocity)

### Q2: 如何处理历史观测？

上半身观测需要 5 步历史。在初始化时：
1. 用第一帧观测填充历史缓冲区
2. 每步更新缓冲区（先进先出）

### Q3: time_step 超出轨迹长度怎么办？

模型内部会 `clamp(time_step, max=total_frames-1)`，不会崩溃。如需循环播放，在外部处理取模。

### Q4: 如何获取轨迹总帧数？

- Motion Mode: 从 ONNX 元数据读取 `trajectory_total_frames`
- Basic Mode: 单独加载轨迹文件获取

---

## 10. 坐标系说明

| 坐标系 | 说明 |
|--------|------|
| World Frame | 世界坐标系，Z 轴向上 |
| Body Frame | 机器人基座坐标系，原点在 `torso_link` |
| `_w` 后缀 | 表示世界坐标系下的量 |
| `_b` 后缀 | 表示 body frame 下的量 |

`projected_gravity` 的计算：
```python
# 重力在世界坐标系下为 [0, 0, -9.81]
# 投影到 body frame
gravity_w = np.array([0, 0, -1])  # 归一化
gravity_b = quat_rotate_inverse(robot_quat, gravity_w)
```
