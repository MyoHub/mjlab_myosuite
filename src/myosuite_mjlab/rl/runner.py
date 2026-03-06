"""Runner compatible with public mjlab: safe save() and ONNX export."""

from __future__ import annotations

import os

from mjlab.rl.runner import MjlabOnPolicyRunner
from mjlab.tasks.velocity.rl import VelocityOnPolicyRunner

# Optional: only used when ONNX export succeeds and wandb is used
try:
    import wandb
except ImportError:
    wandb = None


class MyosuiteVelocityRunner(VelocityOnPolicyRunner):
    """Velocity runner that works with public mjlab (no self.logger, safe ONNX)."""

    def export_policy_to_onnx(
        self, path: str, filename: str = "policy.onnx", verbose: bool = False
    ) -> None:
        """Export policy to ONNX when supported; no-op otherwise."""
        policy = getattr(self.alg, "get_policy", None)
        if callable(policy):
            policy = policy()
        else:
            policy = getattr(self.alg, "policy", None)
        if policy is None or not hasattr(policy, "as_onnx"):
            return
        super().export_policy_to_onnx(path, filename, verbose)

    def save(self, path: str, infos=None) -> None:
        """Save checkpoint; use logger_type and only attach ONNX/wandb when file exists."""
        MjlabOnPolicyRunner.save(self, path, infos)
        policy_path = path.split("model")[0]
        filename = os.path.basename(os.path.dirname(policy_path)) + ".onnx"
        self.export_policy_to_onnx(policy_path, filename)
        logger_type = getattr(self, "logger_type", "tensorboard")
        onnx_path = os.path.join(policy_path, filename)
        if not os.path.isfile(onnx_path):
            return
        try:
            from mjlab.rl.exporter_utils import (
                attach_metadata_to_onnx,
                get_base_metadata,
            )
        except ImportError:
            return
        run_name = "local"
        if logger_type == "wandb" and wandb is not None and wandb.run:
            run_name = getattr(wandb.run, "name", None) or "local"
        metadata = get_base_metadata(self.env.unwrapped, run_name)
        attach_metadata_to_onnx(onnx_path, metadata)
        if logger_type == "wandb" and wandb is not None:
            wandb.save(policy_path + filename, base_path=os.path.dirname(policy_path))
