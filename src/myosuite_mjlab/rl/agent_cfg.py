"""Adapt mjlab agent config to the shape expected by rsl_rl>=4.

- Adds ``policy`` from actor/critic when missing.
- Maps obs_groups actor -> policy when policy missing.
- Filters algorithm dict to PPO-accepted keys (drops optimizer, share_cnn_encoders, etc.).
"""

from __future__ import annotations

# PPO kwargs accepted by rsl_rl; extras like optimizer/share_cnn_encoders are dropped.
_PPO_ALGORITHM_KEYS = {
    "num_learning_epochs",
    "num_mini_batches",
    "clip_param",
    "gamma",
    "lam",
    "value_loss_coef",
    "entropy_coef",
    "learning_rate",
    "max_grad_norm",
    "use_clipped_value_loss",
    "schedule",
    "desired_kl",
    "device",
    "normalize_advantage_per_mini_batch",
    "rnd_cfg",
    "symmetry_cfg",
    "class_name",
}


def adapt_agent_cfg_for_rsl_rl(agent_cfg: dict) -> None:
    """Mutate agent_cfg in place for rsl_rl>=4 compatibility."""
    # policy: build from actor/critic when missing
    if "policy" not in agent_cfg and "actor" in agent_cfg:
        actor_cfg = agent_cfg.get("actor") or {}
        critic_cfg = agent_cfg.get("critic") or actor_cfg
        agent_cfg["policy"] = {
            "class_name": "ActorCritic",
            "actor_hidden_dims": list(actor_cfg.get("hidden_dims", (128, 128, 128))),
            "critic_hidden_dims": list(critic_cfg.get("hidden_dims", (128, 128, 128))),
            "activation": actor_cfg.get("activation", "elu"),
            "init_noise_std": actor_cfg.get("init_noise_std", 1.0),
            "noise_std_type": actor_cfg.get("noise_std_type", "scalar"),
            "actor_obs_normalization": actor_cfg.get("obs_normalization", False),
            "critic_obs_normalization": critic_cfg.get("obs_normalization", False),
        }
    # obs_groups: policy key required; map actor -> policy
    obs_groups = agent_cfg.get("obs_groups") or {}
    if "policy" not in obs_groups and "actor" in obs_groups:
        agent_cfg["obs_groups"] = {**obs_groups, "policy": obs_groups["actor"]}
    # algorithm: keep only PPO-accepted keys
    alg_cfg = agent_cfg.get("algorithm")
    if isinstance(alg_cfg, dict):
        agent_cfg["algorithm"] = {
            k: v for k, v in alg_cfg.items() if k in _PPO_ALGORITHM_KEYS
        }
