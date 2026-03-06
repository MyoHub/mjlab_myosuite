"""Compatibility layer for rsl_rl and public mjlab.

Provides step adapter (5-value step), agent_cfg adaptation (policy, obs_groups,
algorithm filter), and MyosuiteVelocityRunner for safe save/ONNX with public mjlab.
"""

from myosuite_mjlab.rl.agent_cfg import adapt_agent_cfg_for_rsl_rl
from myosuite_mjlab.rl.runner import MyosuiteVelocityRunner
from myosuite_mjlab.rl.step_adapter import RslRlStepAdapter

__all__ = [
    "RslRlStepAdapter",
    "adapt_agent_cfg_for_rsl_rl",
    "MyosuiteVelocityRunner",
]
