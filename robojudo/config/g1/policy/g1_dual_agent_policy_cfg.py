from robojudo.config.g1.env.g1_env_cfg import G1_29DoF
from robojudo.policy.policy_cfgs import DualAgentPolicyCfg
from robojudo.tools.tool_cfgs import DoFConfig


class G1DualAgentDoF(G1_29DoF):
    pass


class G1DualAgentPolicyCfg(DualAgentPolicyCfg):
    robot: str = "g1"
    policy_name: str = "agent_basic"

    obs_dof: DoFConfig = G1DualAgentDoF()
    action_dof: DoFConfig = obs_dof
