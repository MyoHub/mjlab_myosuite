"""MyoSkeleton velocity env config: flat terrain and standing-only."""

import math

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as envs_mdp
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.tasks.velocity import mdp
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg
from mjlab.tasks.velocity.velocity_env_cfg import make_velocity_env_cfg

from .robot_cfg import MYOSKELETON_ACTION_SCALE, get_myoskeleton_robot_cfg


def myoskeleton_flat_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    """Create MyoSkeleton flat terrain velocity configuration."""
    cfg = make_velocity_env_cfg()

    # Simulation settings
    cfg.sim.njmax = 1000
    cfg.sim.nconmax = 100

    cfg.scene.entities = {"robot": get_myoskeleton_robot_cfg()}

    # Contact sensors for feet
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
    self_collision_cfg = ContactSensorCfg(
        name="self_collision",
        primary=ContactMatch(mode="subtree", pattern="pelvis", entity="robot"),
        secondary=ContactMatch(mode="subtree", pattern="pelvis", entity="robot"),
        fields=("found",),
        reduce="none",
        num_slots=1,
    )
    # Replace terrain_scan with our sensors
    cfg.scene.sensors = (feet_ground_cfg, self_collision_cfg)

    # Action configuration
    joint_pos_action = cfg.actions["joint_pos"]
    assert isinstance(joint_pos_action, JointPositionActionCfg)
    # Scale is assigned per-actuator in mjlab reference; we do it globally or per-robot.
    # Actually MYOSKELETON_ACTION_SCALE is a dict.
    # JointPositionActionCfg.scale can be a float or a dict? Usually a float.
    # In mjlab/tasks/velocity/config/myoskeleton/env_cfgs.py:
    # joint_pos_action.scale = MYOSKELETON_ACTION_SCALE
    # Wait, if it's a dict, mjlab's action manager handles it.
    joint_pos_action.scale = MYOSKELETON_ACTION_SCALE

    # Viewer and Commands
    cfg.viewer.body_name = "thoracic_spine"
    twist_cmd = cfg.commands["twist"]
    assert isinstance(twist_cmd, UniformVelocityCommandCfg)
    twist_cmd.viz.z_offset = 1.2
    twist_cmd.ranges.lin_vel_x = (-1.0, 1.0)
    twist_cmd.ranges.lin_vel_y = (-0.5, 0.5)

    # MDP Term Overrides
    site_names = ("l_foot_touch", "r_foot_touch")
    cfg.observations["critic"].terms["foot_height"].params[
        "asset_cfg"
    ].site_names = site_names

    cfg.events["foot_friction"].params["asset_cfg"].geom_names = ".*_coll"
    cfg.events["base_com"].params["asset_cfg"].body_names = ("pelvis",)

    # Rewards
    cfg.rewards["upright"].params["asset_cfg"].body_names = ("pelvis",)
    cfg.rewards["body_ang_vel"].params["asset_cfg"].body_names = ("pelvis",)

    cfg.rewards["track_linear_velocity"].params["std"] = 0.5
    cfg.rewards["track_linear_velocity"].weight = 3.0
    cfg.rewards["track_angular_velocity"].params["std"] = 0.5
    cfg.rewards["track_angular_velocity"].weight = 2.0

    for reward_name in ["foot_clearance", "foot_swing_height", "foot_slip"]:
        cfg.rewards[reward_name].params["asset_cfg"].site_names = site_names

    clean_joints_regex = r"^(?!.*(translation|rotation[2-3]|beta|Abs_t|Abs_r)).*"
    cfg.rewards["pose"].params["asset_cfg"].joint_names = (clean_joints_regex,)

    cfg.rewards["pose"].params["std_standing"] = {".*": 0.05}
    cfg.rewards["pose"].params["std_walking"] = {".*": 0.2}
    cfg.rewards["pose"].params["std_running"] = {".*": 0.4}
    cfg.rewards["pose"].weight = 5.0

    cfg.rewards["alive"] = RewardTermCfg(func=envs_mdp.is_alive, weight=10.0)
    cfg.rewards["upright"].weight = 5.0

    cfg.rewards["body_ang_vel"].weight = -0.05
    cfg.rewards["angular_momentum"].weight = -0.01

    cfg.rewards["self_collisions"] = RewardTermCfg(
        func=mdp.self_collision_cost,
        weight=-1.0,
        params={"sensor_name": self_collision_cfg.name},
    )

    cfg.rewards["action_rate_l2"].weight = -0.01

    # Terminations
    cfg.terminations["fell_over"].params["limit_angle"] = 1.0
    cfg.terminations["base_height"] = TerminationTermCfg(
        func=envs_mdp.root_height_below_minimum, params={"minimum_height": 0.6}
    )

    # Flatten terrain
    cfg.scene.terrain.terrain_type = "plane"
    cfg.scene.terrain.terrain_generator = None
    if "terrain_levels" in cfg.curriculum:
        del cfg.curriculum["terrain_levels"]

    if "height_scan" in cfg.observations["actor"].terms:
        del cfg.observations["actor"].terms["height_scan"]
    if "height_scan" in cfg.observations["critic"].terms:
        del cfg.observations["critic"].terms["height_scan"]

    cfg.episode_length_s = 10.0

    if play:
        cfg.episode_length_s = int(1e9)
        cfg.observations["actor"].enable_corruption = False
        cfg.events.pop("push_robot", None)
        if "command_vel" in cfg.curriculum:
            del cfg.curriculum["command_vel"]
        twist_cmd.ranges.lin_vel_x = (-0.5, 1.0)
        twist_cmd.ranges.ang_vel_z = (-0.5, 0.5)

    return cfg


def myoskeleton_standing_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    """Create MyoSkeleton standing-only configuration."""
    cfg = myoskeleton_flat_env_cfg(play=play)

    twist_cmd = cfg.commands["twist"]
    assert isinstance(twist_cmd, UniformVelocityCommandCfg)
    twist_cmd.rel_standing_envs = 1.0
    twist_cmd.ranges.lin_vel_x = (0.0, 0.0)
    twist_cmd.ranges.lin_vel_y = (0.0, 0.0)
    twist_cmd.ranges.ang_vel_z = (0.0, 0.0)

    cfg.rewards["alive"].weight = 25.0
    cfg.rewards["upright"].weight = 12.0
    cfg.rewards["pose"].weight = 3.0

    cfg.rewards["track_linear_velocity"].weight = 2.0
    cfg.rewards["track_linear_velocity"].params["std"] = 0.5
    cfg.rewards["track_angular_velocity"].weight = 1.5
    cfg.rewards["track_angular_velocity"].params["std"] = 0.5

    cfg.rewards["foot_clearance"].weight = 0.0
    cfg.rewards["foot_swing_height"].weight = 0.0
    cfg.rewards["foot_slip"].weight = 0.0
    cfg.rewards["air_time"].weight = 0.0
    cfg.rewards["soft_landing"].weight = 0.0

    cfg.rewards["action_rate_l2"].weight = -0.01
    cfg.rewards["body_ang_vel"].weight = -0.03
    cfg.rewards["joint_vel"] = RewardTermCfg(func=envs_mdp.joint_vel_l2, weight=-0.001)

    if play:
        cfg.events.pop("push_robot", None)
    else:
        cfg.events["push_robot"] = EventTermCfg(
            func=mdp.push_by_setting_velocity,
            mode="interval",
            interval_range_s=(1.5, 3.5),
            params={
                "velocity_range": {
                    "x": (-0.35, 0.35),
                    "y": (-0.35, 0.35),
                    "z": (-0.25, 0.25),
                    "roll": (-0.35, 0.35),
                    "pitch": (-0.35, 0.35),
                    "yaw": (-0.5, 0.5),
                },
            },
        )

    if "command_vel" in cfg.curriculum:
        del cfg.curriculum["command_vel"]

    cfg.terminations["fell_over"].params["limit_angle"] = math.radians(45.0)

    return cfg
