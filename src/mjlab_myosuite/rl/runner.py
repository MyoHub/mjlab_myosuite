"""RL runner for MyoSuite environments."""

import os

from rsl_rl.runners import OnPolicyRunner

from ..wrapper import MyoSuiteVecEnvWrapper
from .exporter import attach_myosuite_onnx_metadata, export_myosuite_policy_as_onnx


class MyoSuiteOnPolicyRunner(OnPolicyRunner):
  """On-policy runner for MyoSuite environments.

  This runner extends the base OnPolicyRunner and supports ONNX export
  for MyoSuite environments with ManagerBasedRlEnv-compatible structure.
  """

  env: MyoSuiteVecEnvWrapper  # type: ignore[assignment]

  def save(self, path: str, infos=None):
    """Save the model and training information, including ONNX export.

    This method saves the PyTorch model checkpoint and exports the policy
    to ONNX format with MyoSuite-specific metadata.
    """
    # Save the base model checkpoint
    super().save(path, infos)

    # Export ONNX model if using wandb logger
    if self.logger_type in ["wandb"]:
      try:
        import wandb

        policy_path = path.split("model")[0]
        filename = os.path.basename(os.path.dirname(policy_path)) + ".onnx"

        # Get normalizer if available
        if (
          hasattr(self.alg.policy, "actor_obs_normalization")
          and self.alg.policy.actor_obs_normalization
        ):
          normalizer = self.alg.policy.actor_obs_normalizer
        else:
          normalizer = None

        # Export policy to ONNX
        export_myosuite_policy_as_onnx(
          actor_critic=self.alg.policy,
          normalizer=normalizer,
          path=policy_path,
          filename=filename,
          verbose=False,
        )

        # Attach metadata to ONNX model
        try:
          run_name: str = wandb.run.name if wandb.run else "unknown"  # type: ignore
        except Exception:
          run_name = "unknown"

        attach_myosuite_onnx_metadata(
          env=self.env.unwrapped,
          run_path=run_name,
          path=policy_path,
          filename=filename,
        )

        # Save to wandb
        if not self.disable_logs:
          wandb.save(
            os.path.join(policy_path, filename),
            base_path=os.path.dirname(policy_path),
          )
      except ImportError:
        # wandb or ONNX utilities not available, skip ONNX export
        pass
      except Exception as e:
        # Log error but don't fail the save operation
        print(f"[WARNING] Failed to export ONNX model: {e}")
