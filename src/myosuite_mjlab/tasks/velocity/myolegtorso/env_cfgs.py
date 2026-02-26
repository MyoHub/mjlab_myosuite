"""MyoLegTorso velocity env config: flat terrain, tendon effort."""

from dataclasses import replace

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as envs_mdp
from mjlab.envs.mdp.actions import TendonEffortActionCfg
from mjlab.managers.observation_manager import ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg, ObjRef
from mjlab.tasks.velocity import mdp
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg
from mjlab.tasks.velocity.velocity_env_cfg import make_velocity_env_cfg
from myosuite_mjlab.mdp import (
    actuator_force,
    com_lin_vel_w,
    com_pos_w,
    cyclic_hip_flexion_penalty,
    tendon_length,
    tendon_velocity,
)

from .robot_cfg import get_myolegtorso_robot_cfg


def myolegtorso_flat_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    """MyoLegTorso flat velocity: keyframe posture with tendon effort action."""
    cfg = make_velocity_env_cfg()
    cfg.scene.entities = {"robot": get_myolegtorso_robot_cfg()}

    terrain_scan_sensors = [
        s for s in cfg.scene.sensors if getattr(s, "name", None) == "terrain_scan"
    ]
    feet_ground_cfg = ContactSensorCfg(
        name="feet_ground_contact",
        primary=ContactMatch(
            mode="subtree", pattern=r"^(calcn_l|calcn_r)$", entity="robot"
        ),
        secondary=ContactMatch(mode="body", pattern="terrain"),
        fields=("found", "force"),
        reduce="netforce",
        num_slots=1,
        track_air_time=True,
    )
    if terrain_scan_sensors:
        terrain_scan_pelvis = replace(
            terrain_scan_sensors[0],
            frame=ObjRef(type="body", name="pelvis", entity="robot"),
        )
        cfg.scene.sensors = (terrain_scan_pelvis, feet_ground_cfg)
    else:
        cfg.scene.sensors = (feet_ground_cfg,)

    cfg.actions = {
        "tendon_effort": TendonEffortActionCfg(
            entity_name="robot",
            actuator_names=(r".*_tendon",),
            scale=1.0,
        )
    }

    # Observations
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

    cfg.observations["actor"].terms["tendon_length"] = ObservationTermCfg(
        func=tendon_length,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )
    cfg.observations["actor"].terms["tendon_velocity"] = ObservationTermCfg(
        func=tendon_velocity,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )
    cfg.observations["actor"].terms["actuator_force"] = ObservationTermCfg(
        func=actuator_force,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )
    cfg.observations["actor"].terms["com_pos"] = ObservationTermCfg(
        func=com_pos_w,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )
    cfg.observations["actor"].terms["com_lin_vel"] = ObservationTermCfg(
        func=com_lin_vel_w,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )
    for key in (
        "tendon_length",
        "tendon_velocity",
        "actuator_force",
        "com_pos",
        "com_lin_vel",
    ):
        cfg.observations["critic"].terms[key] = cfg.observations["actor"].terms[key]

    cfg.observations["actor"].terms.pop("height_scan", None)
    cfg.observations["critic"].terms.pop("height_scan", None)
    cfg.observations["critic"].terms["foot_height"].params["asset_cfg"].site_names = (
        "l_foot_touch",
        "r_foot_touch",
    )

    # Terrain
    assert cfg.scene.terrain is not None
    cfg.scene.terrain.terrain_type = "plane"
    cfg.scene.terrain.terrain_generator = None
    cfg.curriculum.pop("terrain_levels", None)

    # Rewards
    cfg.events["base_com"].params["asset_cfg"].body_names = ("pelvis",)
    cfg.events["foot_friction"].params["asset_cfg"].geom_names = ".*_col"
    cfg.rewards["upright"].params["asset_cfg"].body_names = ("pelvis",)
    cfg.rewards["body_ang_vel"].params["asset_cfg"].body_names = ("pelvis",)

    cfg.rewards["track_linear_velocity"].params["std"] = 0.5
    cfg.rewards["track_linear_velocity"].weight = 3.0
    cfg.rewards["track_angular_velocity"].params["std"] = 0.5
    cfg.rewards["track_angular_velocity"].weight = 2.0

    site_names = ("l_foot_touch", "r_foot_touch")
    for reward_name in ("foot_clearance", "foot_swing_height", "foot_slip"):
        if reward_name in cfg.rewards:
            cfg.rewards[reward_name].params["asset_cfg"].site_names = site_names

    clean_joints_regex = r"^(?!.*(translation|rotation[2-3]|beta|Abs_t|Abs_r)).*"
    cfg.rewards["pose"].params["asset_cfg"].joint_names = (clean_joints_regex,)
    cfg.rewards["pose"].params["std_standing"] = {".*": 0.05}
    cfg.rewards["pose"].params["std_walking"] = {".*": 0.2}
    cfg.rewards["pose"].params["std_running"] = {".*": 0.4}
    cfg.rewards["pose"].weight = 2.0

    cfg.rewards["alive"] = RewardTermCfg(func=envs_mdp.is_alive, weight=10.0)
    cfg.rewards["upright"].weight = 5.0
    cfg.rewards["body_ang_vel"].weight = -0.05
    cfg.rewards["action_rate_l2"].weight = -0.01

    cfg.rewards["cyclic_hip"] = RewardTermCfg(
        func=cyclic_hip_flexion_penalty,
        weight=-0.5,
        params={
            "hip_period": 100,
            "amplitude": 0.8,
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=("hip_flexion_l", "hip_flexion_r"),
                preserve_order=True,
            ),
        },
    )

    for reward_name in ("foot_clearance", "foot_swing_height", "foot_slip"):
        if reward_name in cfg.rewards:
            cfg.rewards[reward_name].weight = 2.0
    cfg.rewards["soft_landing"].weight = 0.002

    # Terminations
    cfg.terminations["fell_over"].params["limit_angle"] = 1.0
    cfg.terminations["base_height"] = TerminationTermCfg(
        func=envs_mdp.root_height_below_minimum,
        params={"minimum_height": 0.8},
    )

    cfg.episode_length_s = 10.0
    cfg.viewer.body_name = "pelvis"

    twist_cmd = cfg.commands["twist"]
    assert isinstance(twist_cmd, UniformVelocityCommandCfg)
    twist_cmd.ranges.lin_vel_x = (-1.0, 1.0)
    twist_cmd.ranges.lin_vel_y = (-0.5, 0.5)

    if play:
        cfg.episode_length_s = int(1e9)
        cfg.observations["actor"].enable_corruption = False
        cfg.events.pop("push_robot", None)
        cfg.curriculum.pop("command_vel", None)
        twist_cmd.ranges.lin_vel_x = (-0.5, 1.0)
        twist_cmd.ranges.ang_vel_z = (-0.5, 0.5)

    return cfg
