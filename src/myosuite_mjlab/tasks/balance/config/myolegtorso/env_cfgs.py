"""MyoLegsTorso standing-balance env config using synergy actions.

Mirrors the fork's balance setup: synergy tendon effort, standing rewards,
push curriculum. Uses "actor" observation group for mjlab compatibility.
"""

from __future__ import annotations

from dataclasses import replace

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as envs_mdp
from mjlab.managers.curriculum_manager import CurriculumTermCfg
from myosuite_mjlab.mdp import (
    SynergyTendonEffortActionCfg,
    action_scale_curriculum,
    push_velocity_curriculum,
)
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg, ObjRef
from mjlab.tasks.velocity import mdp
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg
from mjlab.tasks.velocity.velocity_env_cfg import make_velocity_env_cfg

from myosuite_mjlab.tasks.velocity.myolegtorso.robot_cfg import (
    get_myolegtorso_robot_cfg,
)


def myolegtorso_balance_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    """Standing-balance configuration for MyoLegsTorso using synergy actions."""
    cfg = make_velocity_env_cfg()

    cfg.scene.entities = {"robot": get_myolegtorso_robot_cfg()}

    assert cfg.scene.terrain is not None
    cfg.scene.terrain.terrain_type = "plane"
    cfg.scene.terrain.terrain_generator = None

    terrain_scan_sensors = [
        s for s in cfg.scene.sensors if getattr(s, "name", None) == "terrain_scan"
    ]
    feet_ground_cfg = ContactSensorCfg(
        name="feet_ground_contact",
        primary=ContactMatch(
            mode="subtree",
            pattern=r"^(calcn_l|calcn_r)$",
            entity="robot",
        ),
        secondary=ContactMatch(mode="body", pattern="terrain"),
        fields=("found", "force"),
        reduce="netforce",
        num_slots=1,
        track_air_time=True,
    )
    if terrain_scan_sensors:
        terrain_scan = replace(
            terrain_scan_sensors[0],
            frame=ObjRef(type="body", name="pelvis", entity="robot"),
        )
        cfg.scene.sensors = (terrain_scan, feet_ground_cfg)
    else:
        cfg.scene.sensors = (feet_ground_cfg,)

    cfg.actions = {
        "tendon_synergy": SynergyTendonEffortActionCfg(
            entity_name="robot",
            scale=0.05,
        ),
    }

    if "command_vel" in cfg.curriculum:
        del cfg.curriculum["command_vel"]
    if "terrain_levels" in cfg.curriculum:
        del cfg.curriculum["terrain_levels"]

    cfg.curriculum["action_scale"] = CurriculumTermCfg(
        func=action_scale_curriculum,
        params={
            "action_name": "tendon_synergy",
            "stages": [
                {"step": 0, "scale": 0.05},
                {"step": 50_000, "scale": 0.15},
                {"step": 100_000, "scale": 0.4},
                {"step": 200_000, "scale": 0.7},
                {"step": 300_000, "scale": 1.0},
            ],
        },
    )

    cfg.observations["actor"].terms["base_lin_vel"] = ObservationTermCfg(
        func=mdp.base_lin_vel,
        params={"asset_cfg": SceneEntityCfg("robot")},
        noise=cfg.observations["actor"].terms["base_lin_vel"].noise,
    )
    cfg.observations["actor"].terms["base_ang_vel"] = ObservationTermCfg(
        func=mdp.base_ang_vel,
        params={"asset_cfg": SceneEntityCfg("robot")},
        noise=cfg.observations["actor"].terms["base_ang_vel"].noise,
    )
    cfg.observations["critic"].terms["base_lin_vel"] = cfg.observations["actor"].terms[
        "base_lin_vel"
    ]
    cfg.observations["critic"].terms["base_ang_vel"] = cfg.observations["actor"].terms[
        "base_ang_vel"
    ]

    twist_cmd = cfg.commands["twist"]
    assert isinstance(twist_cmd, UniformVelocityCommandCfg)
    twist_cmd.ranges.lin_vel_x = (0.0, 0.0)
    twist_cmd.ranges.lin_vel_y = (0.0, 0.0)
    twist_cmd.ranges.ang_vel_z = (0.0, 0.0)

    for group_name in ["actor", "critic"]:
        obs_terms = cfg.observations[group_name].terms
        if "height_scan" in obs_terms:
            del obs_terms["height_scan"]

    cfg.observations["critic"].terms["foot_height"].params["asset_cfg"].site_names = (
        "l_foot_touch",
        "r_foot_touch",
    )
    site_names = ("l_foot_touch", "r_foot_touch")
    for reward_name in ["foot_swing_height", "foot_slip"]:
        if reward_name in cfg.rewards:
            cfg.rewards[reward_name].params["asset_cfg"].site_names = site_names

    keep_rewards = {
        "upright",
        "upright_torso",
        "pose",
        "body_ang_vel",
        "dof_pos_limits",
        "action_rate_l2",
        "alive",
        "air_time",
        "foot_swing_height",
        "foot_slip",
        "soft_landing",
    }
    for name in list(cfg.rewards.keys()):
        if name not in keep_rewards:
            cfg.rewards.pop(name)

    cfg.rewards["upright"].params["asset_cfg"].body_names = ("pelvis",)
    cfg.rewards["upright"].weight = 10.0
    cfg.rewards["upright_torso"] = RewardTermCfg(
        func=mdp.flat_orientation,
        weight=50.0,
        params={
            "std": 0.25,
            "asset_cfg": SceneEntityCfg("robot", body_names=("torso",)),
        },
    )

    clean_joints_regex = r"^(?!.*(translation|rotation[2-3]|beta|Abs_t|Abs_r)).*"
    cfg.rewards["pose"].params["asset_cfg"].joint_names = (clean_joints_regex,)
    cfg.rewards["pose"].params["std_standing"] = {".*": 0.05}
    cfg.rewards["pose"].params["std_walking"] = {".*": 0.2}
    cfg.rewards["pose"].params["std_running"] = {".*": 0.4}
    cfg.rewards["pose"].weight = 1

    cfg.rewards["alive"] = RewardTermCfg(func=envs_mdp.is_alive, weight=10.0)
    cfg.rewards["dof_pos_limits"] = RewardTermCfg(
        func=mdp.joint_pos_limits, weight=-10.0
    )
    cfg.rewards["body_ang_vel"].params["asset_cfg"].body_names = ("pelvis",)
    cfg.rewards["body_ang_vel"].weight = -0.05
    cfg.rewards["action_rate_l2"].weight = -0.01

    cfg.terminations["fell_over"].params["limit_angle"] = 1.2
    cfg.terminations["base_height"] = TerminationTermCfg(
        func=envs_mdp.root_height_below_minimum, params={"minimum_height": 0.75}
    )

    cfg.events["reset_robot_joints"].params["position_range"] = (-0.05, 0.05)

    base_push_range = {
        "x": (-0.35, 0.35),
        "y": (-0.35, 0.35),
        "z": (-0.25, 0.25),
        "roll": (-0.35, 0.35),
        "pitch": (-0.35, 0.35),
        "yaw": (-0.5, 0.5),
    }
    cfg.events["push_robot"] = EventTermCfg(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(1.5, 3.5),
        params={
            "velocity_range": {k: (0.0, 0.0) for k in base_push_range},
        },
    )
    cfg.curriculum["push_vel"] = CurriculumTermCfg(
        func=push_velocity_curriculum,
        params={
            "event_name": "push_robot",
            "base_velocity_range": base_push_range,
            "stages": [
                {"step": 0, "scale": 0.0},
            ],
        },
    )

    cfg.episode_length_s = 20.0
    cfg.viewer.body_name = "pelvis"

    if play:
        cfg.episode_length_s = int(1e9)
        cfg.observations["actor"].enable_corruption = False
        cfg.events.pop("push_robot", None)
        if "push_vel" in cfg.curriculum:
            cfg.curriculum.pop("push_vel")
        if "action_scale" in cfg.curriculum:
            cfg.curriculum.pop("action_scale")
        cfg.actions["tendon_synergy"].scale = 1.0

    return cfg
