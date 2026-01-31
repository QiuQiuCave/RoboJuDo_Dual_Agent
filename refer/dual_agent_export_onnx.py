"""
Export a combined DualAgent policy to a single ONNX.

Upper body policy and lower body policy are merged into one network that takes
two inputs (upper_obs, lower_obs) and outputs a 29D action.

Two export modes are supported:
1. Basic mode (--task tracking/velocity): Exports policy only, no trajectory embedded
2. Motion mode (--embed_trajectory): Exports policy with motion trajectory embedded (tracking task only)
"""

import argparse
import importlib.util
import os
import torch
import numpy as np

try:
    import onnx
    HAS_ONNX = True
except ImportError:
    HAS_ONNX = False


def _load_rsl_exporter_module():
    """Load exporter.py directly to avoid importing IsaacLab runtime deps."""
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    exporter_path = os.path.join(
        repo_root, "source", "beyondAMP", "beyondAMP", "isaaclab", "rsl_rl", "exporter.py"
    )
    if not os.path.exists(exporter_path):
        raise FileNotFoundError(f"exporter.py not found at: {exporter_path}")
    spec = importlib.util.spec_from_file_location("dual_agent_rsl_exporter", exporter_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Failed to load exporter module from: {exporter_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


rsl_exporter = _load_rsl_exporter_module()


LOWER_BODY_ACTION_DOF = 15
UPPER_BODY_ACTION_START = 15
TOTAL_ACTION_DOF = 29

# G1 robot configuration for metadata
G1_LOWER_BODY_JOINT_NAMES = [
    "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint",
    "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
    "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint",
    "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
    "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint",
]

G1_UPPER_BODY_JOINT_NAMES = [
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint",
    "left_elbow_joint", "left_wrist_roll_joint", "left_wrist_pitch_joint", "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint",
    "right_elbow_joint", "right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint",
]

G1_LOWER_BODY_NAMES = [
    "pelvis",
    "left_hip_pitch_link", "left_hip_roll_link", "left_hip_yaw_link",
    "left_knee_link", "left_ankle_pitch_link", "left_ankle_roll_link",
    "right_hip_pitch_link", "right_hip_roll_link", "right_hip_yaw_link",
    "right_knee_link", "right_ankle_pitch_link", "right_ankle_roll_link",
]


def load_policy_from_checkpoint(checkpoint_path: str, device: str):
    """Load a policy from RSL-RL checkpoint or exported policy."""
    run_path = os.path.dirname(checkpoint_path)

    # Check if there's an exported policy.pth file (which is more reliable)
    exported_pth = os.path.join(run_path, "exported", "policy.pth")

    if os.path.exists(exported_pth):
        print(f"[INFO] Loading from exported policy: {exported_pth}")
        # weights_only=False is needed for PyTorch 2.6+ to load full model objects
        actor_critic = torch.load(exported_pth, map_location=device, weights_only=False)
        actor_critic.eval()
    else:
        print(f"[INFO] Loading from checkpoint: {checkpoint_path}")
        loaded_dict = torch.load(checkpoint_path, map_location=device, weights_only=False)

        from rsl_rl_amp.modules import ActorCritic

        state_dict = loaded_dict["model_state_dict"]

        # Actor input size from first layer
        actor_input_size = state_dict["actor.0.weight"].shape[1]
        # Critic input size from first layer
        critic_input_size = state_dict["critic.0.weight"].shape[1]

        actor_layer_indices = sorted(
            [int(k.split(".")[1]) for k in state_dict.keys() if k.startswith("actor.") and k.endswith(".weight")]
        )
        last_actor_layer_idx = actor_layer_indices[-1]
        action_size = state_dict[f"actor.{last_actor_layer_idx}.weight"].shape[0]

        actor_hidden_dims = [
            state_dict[f"actor.{idx}.weight"].shape[0] for idx in actor_layer_indices[:-1]
        ]

        critic_layer_indices = sorted(
            [int(k.split(".")[1]) for k in state_dict.keys() if k.startswith("critic.") and k.endswith(".weight")]
        )
        critic_hidden_dims = [
            state_dict[f"critic.{idx}.weight"].shape[0] for idx in critic_layer_indices[:-1]
        ]

        actor_critic = ActorCritic(
            num_actor_obs=actor_input_size,
            num_critic_obs=critic_input_size,
            num_actions=action_size,
            actor_hidden_dims=actor_hidden_dims if actor_hidden_dims else [512, 256, 128],
            critic_hidden_dims=critic_hidden_dims if critic_hidden_dims else [512, 256, 128],
            activation="elu",
            init_noise_std=1.0,
        ).to(device)

        actor_critic.load_state_dict(loaded_dict["model_state_dict"])
        actor_critic.eval()

    return actor_critic


def load_motion_trajectory(motion_file: str) -> dict:
    """
    Load motion trajectory from npz file.
    
    Returns dict with trajectory tensors ready for embedding.
    """
    if not os.path.exists(motion_file):
        raise FileNotFoundError(f"Motion file not found: {motion_file}")
    
    data = np.load(motion_file)
    
    trajectory = {
        "joint_pos": torch.from_numpy(data["joint_pos"].astype(np.float32)),
        "joint_vel": torch.from_numpy(data["joint_vel"].astype(np.float32)),
    }
    
    # Body tracking data (if available)
    if "body_pos" in data:
        trajectory["body_pos_w"] = torch.from_numpy(data["body_pos"].astype(np.float32))
    if "body_quat" in data:
        trajectory["body_quat_w"] = torch.from_numpy(data["body_quat"].astype(np.float32))
    if "body_lin_vel" in data:
        trajectory["body_lin_vel_w"] = torch.from_numpy(data["body_lin_vel"].astype(np.float32))
    if "body_ang_vel" in data:
        trajectory["body_ang_vel_w"] = torch.from_numpy(data["body_ang_vel"].astype(np.float32))
    
    # Metadata
    trajectory["total_frames"] = trajectory["joint_pos"].shape[0]
    trajectory["dt"] = float(data.get("dt", 0.02))
    trajectory["source_file"] = os.path.basename(motion_file)
    
    return trajectory


# ==============================================================================
# Basic Export Mode (Original Implementation)
# ==============================================================================


class CombinedActor(torch.nn.Module):
    """Wrap upper + lower actors and output a combined 29D action."""

    def __init__(self, upper_actor: torch.nn.Module, lower_actor: torch.nn.Module):
        super().__init__()
        self.upper_actor = upper_actor
        self.lower_actor = lower_actor

    def forward(self, upper_obs: torch.Tensor, lower_obs: torch.Tensor) -> torch.Tensor:
        upper_action = self.upper_actor(upper_obs)
        lower_action = self.lower_actor(lower_obs)

        combined_action = torch.zeros(
            upper_action.shape[0],
            TOTAL_ACTION_DOF,
            device=upper_action.device,
            dtype=upper_action.dtype,
        )
        combined_action[:, :LOWER_BODY_ACTION_DOF] = lower_action
        combined_action[:, UPPER_BODY_ACTION_START:TOTAL_ACTION_DOF] = upper_action[
            :, UPPER_BODY_ACTION_START:TOTAL_ACTION_DOF
        ]
        return combined_action


class CombinedPolicy:
    """Minimal policy container for the ONNX exporter."""

    is_recurrent = False

    def __init__(self, combined_actor: torch.nn.Module, upper_obs_dim: int, lower_obs_dim: int):
        self.actor = combined_actor
        self.upper_obs_dim = upper_obs_dim
        self.lower_obs_dim = lower_obs_dim


class _DualAgentOnnxExporter(rsl_exporter._OnnxPolicyExporter):
    """Custom ONNX exporter to support two inputs (upper_obs, lower_obs)."""

    def __init__(self, policy, normalizer=None, verbose=False):
        super().__init__(policy, normalizer, verbose)
        self.upper_obs_dim = policy.upper_obs_dim
        self.lower_obs_dim = policy.lower_obs_dim

    def forward(self, upper_obs, lower_obs):
        return self.actor(upper_obs, lower_obs)

    def export(self, path, filename):
        self.to("cpu")
        self.eval()
        upper_obs = torch.zeros(1, self.upper_obs_dim)
        lower_obs = torch.zeros(1, self.lower_obs_dim)
        torch.onnx.export(
            self,
            (upper_obs, lower_obs),
            os.path.join(path, filename),
            export_params=True,
            opset_version=11,
            verbose=self.verbose,
            input_names=["upper_obs", "lower_obs"],
            output_names=["actions"],
            dynamic_axes={},
        )


def export_combined_policy_as_onnx(policy, output_dir: str, filename: str, verbose: bool = False):
    """Export the combined policy using the shared export_policy_as_onnx entrypoint."""
    original_exporter = rsl_exporter._OnnxPolicyExporter
    rsl_exporter._OnnxPolicyExporter = _DualAgentOnnxExporter
    try:
        rsl_exporter.export_policy_as_onnx(policy, output_dir, filename=filename, verbose=verbose)
    finally:
        rsl_exporter._OnnxPolicyExporter = original_exporter


# ==============================================================================
# Motion Export Mode (New Implementation with Trajectory Embedded)
# ==============================================================================


class _DualAgentMotionOnnxExporter(torch.nn.Module):
    """
    Dual agent ONNX exporter with embedded motion trajectory.
    
    Similar to beyondMimic's _OnnxMotionPolicyExporter but for dual agent.
    
    Inputs:
        - upper_obs: (1, upper_obs_dim) upper body observations
        - lower_obs: (1, lower_obs_dim) lower body observations  
        - time_step: (1, 1) current time step index
    
    Outputs:
        - actions: (1, 29) combined actions
        - joint_pos: (1, num_joints) target joint positions at time_step
        - joint_vel: (1, num_joints) target joint velocities at time_step
        - body_pos_w: (1, num_bodies, 3) target body positions at time_step
        - body_quat_w: (1, num_bodies, 4) target body quaternions at time_step
        - body_lin_vel_w: (1, num_bodies, 3) target body linear velocities at time_step
        - body_ang_vel_w: (1, num_bodies, 3) target body angular velocities at time_step
    """

    def __init__(
        self,
        combined_actor: torch.nn.Module,
        upper_obs_dim: int,
        lower_obs_dim: int,
        trajectory: dict,
        verbose: bool = False,
    ):
        super().__init__()
        self.actor = combined_actor
        self.upper_obs_dim = upper_obs_dim
        self.lower_obs_dim = lower_obs_dim
        self.verbose = verbose
        
        # Register trajectory data as buffers (will be embedded in ONNX)
        self.register_buffer("joint_pos", trajectory["joint_pos"])
        self.register_buffer("joint_vel", trajectory["joint_vel"])
        
        # Optional body tracking data
        if "body_pos_w" in trajectory:
            self.register_buffer("body_pos_w", trajectory["body_pos_w"])
        else:
            # Create dummy buffer if not available
            self.register_buffer("body_pos_w", torch.zeros(trajectory["total_frames"], 1, 3))
            
        if "body_quat_w" in trajectory:
            self.register_buffer("body_quat_w", trajectory["body_quat_w"])
        else:
            self.register_buffer("body_quat_w", torch.zeros(trajectory["total_frames"], 1, 4))
            
        if "body_lin_vel_w" in trajectory:
            self.register_buffer("body_lin_vel_w", trajectory["body_lin_vel_w"])
        else:
            self.register_buffer("body_lin_vel_w", torch.zeros(trajectory["total_frames"], 1, 3))
            
        if "body_ang_vel_w" in trajectory:
            self.register_buffer("body_ang_vel_w", trajectory["body_ang_vel_w"])
        else:
            self.register_buffer("body_ang_vel_w", torch.zeros(trajectory["total_frames"], 1, 3))
        
        self.time_step_total = trajectory["total_frames"]
        self.trajectory_dt = trajectory["dt"]
        self.trajectory_source = trajectory["source_file"]

    def forward(self, upper_obs, lower_obs, time_step):
        # Clamp time_step to valid range
        time_step_clamped = torch.clamp(time_step.long().squeeze(-1), max=self.time_step_total - 1)
        
        # Get actions from combined actor
        actions = self.actor(upper_obs, lower_obs)
        
        # Return actions and trajectory data at current time_step
        return (
            actions,
            self.joint_pos[time_step_clamped],
            self.joint_vel[time_step_clamped],
            self.body_pos_w[time_step_clamped],
            self.body_quat_w[time_step_clamped],
            self.body_lin_vel_w[time_step_clamped],
            self.body_ang_vel_w[time_step_clamped],
        )

    def export(self, path, filename):
        self.to("cpu")
        self.eval()
        
        # Create example inputs
        upper_obs = torch.zeros(1, self.upper_obs_dim)
        lower_obs = torch.zeros(1, self.lower_obs_dim)
        time_step = torch.zeros(1, 1)
        
        output_path = os.path.join(path, filename)
        
        torch.onnx.export(
            self,
            (upper_obs, lower_obs, time_step),
            output_path,
            export_params=True,
            opset_version=11,
            verbose=self.verbose,
            input_names=["upper_obs", "lower_obs", "time_step"],
            output_names=[
                "actions",
                "joint_pos",
                "joint_vel",
                "body_pos_w",
                "body_quat_w",
                "body_lin_vel_w",
                "body_ang_vel_w",
            ],
            dynamic_axes={},
        )
        
        print(f"[INFO] Exported ONNX with embedded trajectory: {output_path}")
        print(f"       - Trajectory: {self.trajectory_source}")
        print(f"       - Total frames: {self.time_step_total}")
        print(f"       - Duration: {self.time_step_total * self.trajectory_dt:.2f}s @ {1.0/self.trajectory_dt:.1f}Hz")
        print(f"       - Joint pos shape: {tuple(self.joint_pos.shape)}")
        print(f"       - Body pos shape: {tuple(self.body_pos_w.shape)}")


def list_to_csv_str(arr, *, decimals: int = 3, delimiter: str = ",") -> str:
    """Convert list to CSV string for ONNX metadata."""
    fmt = f"{{:.{decimals}f}}"
    return delimiter.join(
        fmt.format(x) if isinstance(x, (int, float)) else str(x) for x in arr
    )


def attach_dual_agent_onnx_metadata(
    path: str,
    filename: str,
    upper_obs_dim: int,
    lower_obs_dim: int,
    trajectory_info: dict,
) -> None:
    """Attach metadata to exported ONNX file."""
    if not HAS_ONNX:
        print("[WARN] onnx package not installed, skipping metadata attachment")
        return
    
    onnx_path = os.path.join(path, filename)
    
    metadata = {
        "model_type": "dual_agent_motion",
        "upper_obs_dim": str(upper_obs_dim),
        "lower_obs_dim": str(lower_obs_dim),
        "action_dim": str(TOTAL_ACTION_DOF),
        "lower_body_action_dim": str(LOWER_BODY_ACTION_DOF),
        "upper_body_action_start": str(UPPER_BODY_ACTION_START),
        "lower_body_joint_names": list_to_csv_str(G1_LOWER_BODY_JOINT_NAMES),
        "upper_body_joint_names": list_to_csv_str(G1_UPPER_BODY_JOINT_NAMES),
        "lower_body_names": list_to_csv_str(G1_LOWER_BODY_NAMES),
        "anchor_body_name": "torso_link",
        "trajectory_source": trajectory_info["source_file"],
        "trajectory_total_frames": str(trajectory_info["total_frames"]),
        "trajectory_dt": str(trajectory_info["dt"]),
        "trajectory_duration_s": str(trajectory_info["total_frames"] * trajectory_info["dt"]),
    }
    
    model = onnx.load(onnx_path)
    
    for k, v in metadata.items():
        entry = onnx.StringStringEntryProto()
        entry.key = k
        entry.value = v
        model.metadata_props.append(entry)
    
    onnx.save(model, onnx_path)
    print(f"[INFO] Attached metadata to: {onnx_path}")


def export_dual_agent_motion_policy_as_onnx(
    upper_policy,
    lower_policy,
    motion_file: str,
    output_dir: str,
    filename: str = "dual_agent_motion.onnx",
    verbose: bool = False,
):
    """
    Export dual agent policy with embedded motion trajectory.
    
    This is the main entry point for motion export mode, similar to
    beyondMimic's export_motion_policy_as_onnx.
    """
    # Load trajectory
    print(f"[INFO] Loading trajectory: {motion_file}")
    trajectory = load_motion_trajectory(motion_file)
    
    # Get dimensions
    upper_obs_dim = upper_policy.actor[0].in_features
    lower_obs_dim = lower_policy.actor[0].in_features
    
    # Create combined actor
    combined_actor = CombinedActor(upper_policy.actor, lower_policy.actor)
    combined_actor.eval()
    
    # Create exporter
    exporter = _DualAgentMotionOnnxExporter(
        combined_actor,
        upper_obs_dim,
        lower_obs_dim,
        trajectory,
        verbose,
    )
    
    # Export
    os.makedirs(output_dir, exist_ok=True)
    exporter.export(output_dir, filename)
    
    # Attach metadata
    attach_dual_agent_onnx_metadata(
        output_dir,
        filename,
        upper_obs_dim,
        lower_obs_dim,
        trajectory,
    )
    
    return exporter


# ==============================================================================
# Main Entry Point
# ==============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="Export dual-agent policies into a single ONNX.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic export (no trajectory)
  python dual_agent_export_onnx.py --upper_policy path/to/upper.pt --lower_policy path/to/lower.pt

  # Export with embedded trajectory
  python dual_agent_export_onnx.py --upper_policy path/to/upper.pt --lower_policy path/to/lower.pt \\
      --embed_trajectory --motion_file data/demo/holdthebox/overhurdle4.npz
        """,
    )
    parser.add_argument(
        "--upper_policy",
        type=str,
        required=True,
        help="Path to upper body policy checkpoint",
    )
    parser.add_argument(
        "--lower_policy",
        type=str,
        required=True,
        help="Path to lower body policy checkpoint",
    )
    parser.add_argument(
        "--task",
        type=str,
        default="tracking",
        choices=["tracking", "velocity"],
        help="Task type: 'tracking' for motion tracking, 'velocity' for velocity tracking",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="logs/dual_agent/exported",
        help="Directory to save the exported ONNX file",
    )
    parser.add_argument(
        "--rldevice",
        type=str,
        default="cuda:0",
        help="Device for loading policies",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        default=False,
        help="Enable verbose ONNX export logging",
    )
    # New arguments for motion export mode
    parser.add_argument(
        "--embed_trajectory",
        action="store_true",
        help="Embed motion trajectory into ONNX (only for tracking task)",
    )
    parser.add_argument(
        "--motion_file",
        type=str,
        default=None,
        help="Path to motion file (.npz) for embedding. Default: data/demo/holdthebox/overhurdle4.npz",
    )
    args_cli = parser.parse_args()

    upper_policy_path = os.path.abspath(args_cli.upper_policy)
    lower_policy_path = os.path.abspath(args_cli.lower_policy)

    if not os.path.exists(upper_policy_path):
        raise FileNotFoundError(f"Upper policy not found: {upper_policy_path}")
    if not os.path.exists(lower_policy_path):
        raise FileNotFoundError(f"Lower policy not found: {lower_policy_path}")

    device = args_cli.rldevice

    print(f"[INFO] Loading upper body policy from: {upper_policy_path}")
    upper_policy = load_policy_from_checkpoint(upper_policy_path, device)

    print(f"[INFO] Loading lower body policy from: {lower_policy_path}")
    lower_policy = load_policy_from_checkpoint(lower_policy_path, device)

    upper_obs_dim = upper_policy.actor[0].in_features
    lower_obs_dim = lower_policy.actor[0].in_features

    expected_upper_dim = 480
    expected_lower_dim = 121 if args_cli.task == "tracking" else 99

    print(f"[INFO] Upper obs dim: {upper_obs_dim} (expected {expected_upper_dim})")
    print(f"[INFO] Lower obs dim: {lower_obs_dim} (expected {expected_lower_dim})")

    if upper_obs_dim != expected_upper_dim or lower_obs_dim != expected_lower_dim:
        print("\n" + "!" * 80)
        print("[ERROR] Observation dimension mismatch!")
        if upper_obs_dim != expected_upper_dim:
            print(f"Upper body: Policy expects {upper_obs_dim}, but expected {expected_upper_dim}.")
        if lower_obs_dim != expected_lower_dim:
            print(
                f"Lower body: Policy expects {lower_obs_dim}, but task '{args_cli.task}' expects {expected_lower_dim}."
            )
        print("!" * 80 + "\n")
        raise SystemExit(1)

    output_dir = os.path.abspath(args_cli.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    # Choose export mode
    if args_cli.embed_trajectory:
        if args_cli.task != "tracking":
            print("[WARN] --embed_trajectory is only supported for tracking task, ignoring")
            args_cli.embed_trajectory = False
    
    if args_cli.embed_trajectory:
        # Motion export mode: embed trajectory
        motion_file = args_cli.motion_file
        if motion_file is None:
            # Use default motion file
            repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
            motion_file = os.path.join(repo_root, "data/demo/holdthebox/overhurdle4.npz")
            print(f"[INFO] Using default motion file: {motion_file}")
        
        if not os.path.exists(motion_file):
            raise FileNotFoundError(f"Motion file not found: {motion_file}")
        
        export_dual_agent_motion_policy_as_onnx(
            upper_policy,
            lower_policy,
            motion_file,
            output_dir,
            filename="dual_agent_motion.onnx",
            verbose=args_cli.verbose,
        )
    else:
        # Basic export mode: policy only
        combined_actor = CombinedActor(upper_policy.actor, lower_policy.actor)
        combined_actor.eval()
        combined_policy = CombinedPolicy(combined_actor, upper_obs_dim, lower_obs_dim)

        onnx_filename = "dual_agent_combined.onnx"
        export_combined_policy_as_onnx(combined_policy, output_dir, onnx_filename, verbose=args_cli.verbose)

        onnx_path = os.path.join(output_dir, onnx_filename)
        print(f"[INFO] Exported ONNX: {onnx_path}")


if __name__ == "__main__":
    main()
