import logging
import os
from typing import Any

import numpy as np
import onnxruntime as ort

from robojudo.policy import Policy, policy_registry
from robojudo.policy.policy_cfgs import DualAgentPolicyCfg

logger = logging.getLogger(__name__)


def _select_providers(device: str) -> list[str]:
    device_lower = device.lower()
    if device_lower.startswith("cpu"):
        return ["CPUExecutionProvider"]
    if device_lower.startswith("cuda"):
        return ["CUDAExecutionProvider", "CPUExecutionProvider"]
    if device_lower.startswith("tensorrt"):
        return [
            "TensorrtExecutionProvider",
            "CUDAExecutionProvider",
            "CPUExecutionProvider",
        ]
    raise ValueError(f"Unknown device: {device}")


@policy_registry.register
class DualAgentPolicy(Policy):
    cfg_policy: DualAgentPolicyCfg

    def __init__(self, cfg_policy: DualAgentPolicyCfg, device: str):
        if not os.path.isfile(cfg_policy.policy_file):
            raise FileNotFoundError(f"Model file not found at {cfg_policy.policy_file}")

        sess_options = ort.SessionOptions()
        providers = _select_providers(device)
        self.session = ort.InferenceSession(cfg_policy.policy_file, sess_options, providers=providers)

        self.input_names = [i.name for i in self.session.get_inputs()]
        self.output_names = [o.name for o in self.session.get_outputs()]

        self.upper_input_name: str | None = None
        self.lower_input_name: str | None = None
        self.single_obs_name: str | None = None
        self.time_input_name: str | None = None
        self._resolve_input_names()

        super().__init__(cfg_policy=cfg_policy, device=device)

        self.mode = cfg_policy.mode
        self.upper_obs_dim = cfg_policy.upper_obs_dim
        self.lower_obs_dim = cfg_policy.lower_obs_dim
        self._infer_obs_dims_from_model()

        self._motion_extras: dict[str, np.ndarray] = {}
        self.reset()

    def _resolve_input_names(self) -> None:
        time_candidates = [name for name in self.input_names if "time" in name.lower()]
        self.time_input_name = time_candidates[0] if time_candidates else None

        obs_inputs = [name for name in self.input_names if name != self.time_input_name]
        upper_candidates = [name for name in obs_inputs if "upper" in name.lower()]
        lower_candidates = [name for name in obs_inputs if "lower" in name.lower()]

        self.upper_input_name = upper_candidates[0] if upper_candidates else None
        self.lower_input_name = lower_candidates[0] if lower_candidates else None

        if self.upper_input_name is None or self.lower_input_name is None:
            if len(obs_inputs) >= 2:
                if self.upper_input_name is None:
                    self.upper_input_name = obs_inputs[0]
                if self.lower_input_name is None:
                    self.lower_input_name = obs_inputs[1]
            elif len(obs_inputs) == 1:
                self.single_obs_name = obs_inputs[0]

        if self.single_obs_name is not None:
            logger.debug(
                "[DualAgentPolicy] Single input detected: %s; upper/lower obs will be concatenated.",
                self.single_obs_name,
            )

    def _infer_obs_dims_from_model(self) -> None:
        input_shapes = {inp.name: inp.shape for inp in self.session.get_inputs()}

        def infer_dim(name: str | None) -> int | None:
            if name is None:
                return None
            shape = input_shapes.get(name)
            if shape is None:
                return None
            for dim in reversed(shape):
                if isinstance(dim, int):
                    return dim
            return None

        upper_dim = infer_dim(self.upper_input_name)
        if upper_dim is not None:
            self.upper_obs_dim = upper_dim

        lower_dim = infer_dim(self.lower_input_name)
        if lower_dim is not None:
            self.lower_obs_dim = lower_dim

    def reset(self):
        self.timestep = 0
        self.last_action = np.zeros(self.num_actions)
        self._motion_extras = {}

    def post_step_callback(self, commands: list[str] | None = None):
        self.timestep += 1

    def _pad_or_trim(self, obs: np.ndarray, target_dim: int | None) -> np.ndarray:
        obs = np.asarray(obs, dtype=np.float32).reshape(-1)
        if target_dim is None or target_dim <= 0:
            return obs
        if obs.shape[0] < target_dim:
            pad = np.zeros((target_dim - obs.shape[0],), dtype=np.float32)
            obs = np.concatenate([obs, pad], axis=0)
        elif obs.shape[0] > target_dim:
            obs = obs[:target_dim]
        return obs

    def _build_lower_obs(self, env_data) -> np.ndarray:
        dof_pos = env_data.dof_pos
        dof_vel = env_data.dof_vel
        base_ang_vel = env_data.base_ang_vel
        base_lin_vel = env_data.base_lin_vel if env_data.base_lin_vel is not None else np.zeros(3)
        base_quat = env_data.base_quat
        base_pos = env_data.base_pos if env_data.base_pos is not None else np.zeros(3)

        dof_pos_rel = dof_pos - self.default_dof_pos
        obs_parts = [
            base_ang_vel,
            base_lin_vel,
            base_quat,
            base_pos,
            dof_pos_rel,
            dof_vel,
            self.last_action,
        ]
        obs = np.concatenate([np.asarray(part, dtype=np.float32).reshape(-1) for part in obs_parts], axis=0)
        return self._pad_or_trim(obs, self.lower_obs_dim)

    def _build_upper_obs(self, env_data) -> np.ndarray:
        fk_info = env_data.fk_info
        if fk_info is None:
            obs = np.zeros((self.upper_obs_dim,), dtype=np.float32)
            return obs

        body_names = sorted(fk_info.keys())
        obs_parts = []
        for name in body_names:
            body = fk_info[name]
            obs_parts.extend(
                [
                    body.get("pos", np.zeros(3)),
                    body.get("quat", np.array([0.0, 0.0, 0.0, 1.0])),
                    body.get("lin_vel", np.zeros(3)),
                    body.get("ang_vel", np.zeros(3)),
                ]
            )
        obs = np.concatenate([np.asarray(part, dtype=np.float32).reshape(-1) for part in obs_parts], axis=0)
        return self._pad_or_trim(obs, self.upper_obs_dim)

    def get_observation(self, env_data, ctrl_data) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        upper_obs = self._build_upper_obs(env_data)
        lower_obs = self._build_lower_obs(env_data)

        extras: dict[str, Any] = {}
        if self._motion_extras:
            extras.update(self._motion_extras)
            extras["dual_agent"] = self._motion_extras

        obs = {
            "upper_obs": upper_obs,
            "lower_obs": lower_obs,
        }
        return obs, extras

    def _build_ort_inputs(self, upper_obs: np.ndarray, lower_obs: np.ndarray) -> dict[str, np.ndarray]:
        ort_inputs: dict[str, np.ndarray] = {}

        if self.single_obs_name is not None:
            merged_obs = np.concatenate([upper_obs, lower_obs], axis=0)
            ort_inputs[self.single_obs_name] = np.expand_dims(merged_obs, axis=0).astype(np.float32)
        else:
            if self.upper_input_name is not None:
                ort_inputs[self.upper_input_name] = np.expand_dims(upper_obs, axis=0).astype(np.float32)
            if self.lower_input_name is not None:
                ort_inputs[self.lower_input_name] = np.expand_dims(lower_obs, axis=0).astype(np.float32)

        if self.time_input_name is not None:
            ort_inputs[self.time_input_name] = np.expand_dims(np.array([float(self.timestep)], dtype=np.float32), axis=0)

        return ort_inputs

    def _extract_action(self, output_map: dict[str, np.ndarray], outputs: list[np.ndarray]) -> np.ndarray:
        if "actions" in output_map:
            return output_map["actions"]
        if "action" in output_map:
            return output_map["action"]
        return outputs[0]

    def get_action(self, obs: np.ndarray | dict[str, np.ndarray]) -> np.ndarray:
        if isinstance(obs, dict):
            upper_obs = obs.get("upper_obs")
            lower_obs = obs.get("lower_obs")
            if upper_obs is None or lower_obs is None:
                raise ValueError("DualAgentPolicy expects 'upper_obs' and 'lower_obs' in observation dict")
        else:
            raise ValueError("DualAgentPolicy expects observation dict with 'upper_obs' and 'lower_obs'")

        ort_inputs = self._build_ort_inputs(upper_obs, lower_obs)
        outputs = self.session.run(None, ort_inputs)
        output_map = {name: np.asarray(output) for name, output in zip(self.output_names, outputs, strict=True)}

        actions = np.asarray(self._extract_action(output_map, outputs)).squeeze()
        if actions.shape[0] != self.num_actions:
            logger.warning(
                "[DualAgentPolicy] Action dim mismatch: got %s, expected %s. Truncating/padding.",
                actions.shape[0],
                self.num_actions,
            )
            actions = self._pad_or_trim(actions, self.num_actions)

        actions = (1 - self.action_beta) * self.last_action + self.action_beta * actions
        self.last_action = actions.copy()

        processed_actions = actions
        if self.action_clip is not None:
            processed_actions = np.clip(processed_actions, -self.action_clip, self.action_clip)
        processed_actions = processed_actions * self.action_scale

        if self.mode == "motion":
            motion_extras: dict[str, np.ndarray] = {}
            for key in [
                "joint_pos",
                "joint_vel",
                "body_pos_w",
                "body_quat_w",
                "body_lin_vel_w",
                "body_ang_vel_w",
            ]:
                if key in output_map:
                    motion_extras[key] = np.asarray(output_map[key]).squeeze()
            if motion_extras:
                motion_extras["time_step"] = np.array(self.timestep, dtype=np.float32)
                self._motion_extras = motion_extras

        return processed_actions

    def get_init_dof_pos(self) -> np.ndarray:
        return self.default_pos.copy()
