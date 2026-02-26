"""MyoSkeleton robot config: adapted from MyoSuite's myobody via MjSpec."""

from __future__ import annotations

import os
from pathlib import Path

import mujoco

from mjlab.actuator import BuiltinPositionActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.utils.spec_config import CollisionCfg


def _resolve_myobody_xml_path() -> Path:
    """Resolve MyoFullBody (myobody.xml) path from myosuite assets."""
    env_override = os.environ.get("MYOSUITE_MJLAB_MYOBODY_XML")
    if env_override:
        xml_path = Path(env_override)
        if xml_path.exists():
            return xml_path

    try:
        import myosuite  # type: ignore
    except ImportError as exc:
        raise FileNotFoundError(
            "myosuite is required to resolve MyoFullBody XML. "
            "Install myosuite or set MYOSUITE_MJLAB_MYOBODY_XML."
        ) from exc

    myosuite_root = Path(myosuite.__file__).resolve().parent
    candidate = myosuite_root / "simhive" / "myo_sim" / "body" / "myobody.xml"
    if candidate.exists():
        return candidate

    raise FileNotFoundError(
        "Could not locate myobody.xml from myosuite package assets; "
        "set MYOSUITE_MJLAB_MYOBODY_XML explicitly."
    )


FINGER_JOINTS: tuple[str, ...] = (
    "cmc_flexion_r",
    "cmc_abduction_r",
    "mp_flexion_r",
    "ip_flexion_r",
    "mcp2_flexion_r",
    "mcp2_abduction_r",
    "pm2_flexion_r",
    "md2_flexion_r",
    "mcp3_flexion_r",
    "mcp3_abduction_r",
    "pm3_flexion_r",
    "md3_flexion_r",
    "mcp4_flexion_r",
    "mcp4_abduction_r",
    "pm4_flexion_r",
    "md4_flexion_r",
    "mcp5_flexion_r",
    "mcp5_abduction_r",
    "pm5_flexion_r",
    "md5_flexion_r",
    "cmc_flexion_l",
    "cmc_abduction_l",
    "mp_flexion_l",
    "ip_flexion_l",
    "mcp2_flexion_l",
    "mcp2_abduction_l",
    "pm2_flexion_l",
    "md2_flexion_l",
    "mcp3_flexion_l",
    "mcp3_abduction_l",
    "pm3_flexion_l",
    "md3_flexion_l",
    "mcp4_flexion_l",
    "mcp4_abduction_l",
    "pm4_flexion_l",
    "md4_flexion_l",
    "mcp5_flexion_l",
    "mcp5_abduction_l",
    "pm5_flexion_l",
    "md5_flexion_l",
)


def get_myoskeleton_spec() -> mujoco.MjSpec:
    """Load the MyoSkeleton MjSpec (myobody.xml) and disable fingers."""
    spec = mujoco.MjSpec.from_file(str(_resolve_myobody_xml_path()))

    # Remove finger joints as in mjlab reference
    finger_set = set(FINGER_JOINTS)
    for j in list(spec.joints):
        if j.name in finger_set:
            spec.delete(j)

    # Add root angular momentum sensor if not present (parity with mjlab)
    if not any(s.name == "root_angmom" for s in spec.sensors):
        # We need to find the worldbody root body.
        # myobody.xml has <body name="Full Body" ...> with <freejoint name="root"/>
        spec.lookaway_sensors = (
            True  # Avoid errors if root not found? No, let's find it.
        )
        root_body = None
        for b in spec.bodies:
            if b.name == "Full Body":
                root_body = b
                break

        if root_body:
            s_cfg = spec.add_sensor("subtreeangmom")
            s_cfg.name = "root_angmom"
            s_cfg.parentid = root_body.id

    return spec


# Actuator calibration logic from mjlab reference
NATURAL_FREQ = 10.0 * 2.0 * 3.1415926535  # 10 Hz
BASE_INERTIA = 0.5e-5


def _actuator_params(gear: float) -> tuple[float, float, float, float]:
    armature = BASE_INERTIA * gear * gear
    stiffness = armature * NATURAL_FREQ**2
    damping = 2.0 * 2.0 * armature * NATURAL_FREQ
    effort_limit = gear * 0.5
    return stiffness, damping, effort_limit, armature


# Actuator definitions (matches mjlab myoskeleton_constants.py)
_S_SPINE_FB, _D_SPINE_FB, _E_SPINE_FB, _A_SPINE_FB = _actuator_params(160)
_S_SPINE_AR, _D_SPINE_AR, _E_SPINE_AR, _A_SPINE_AR = _actuator_params(100)
_S_ARM, _D_ARM, _E_ARM, _A_ARM = _actuator_params(250)
_S_WRIST, _D_WRIST, _E_WRIST, _A_WRIST = _actuator_params(50)
_S_HIP_FLEX, _D_HIP_FLEX, _E_HIP_FLEX, _A_HIP_FLEX = _actuator_params(275)
_S_HIP_ADD, _D_HIP_ADD, _E_HIP_ADD, _A_HIP_ADD = _actuator_params(530)
_S_HIP_ROT, _D_HIP_ROT, _E_HIP_ROT, _A_HIP_ROT = _actuator_params(600)
_S_KNEE, _D_KNEE, _E_KNEE, _A_KNEE = _actuator_params(600)
_S_ANKLE, _D_ANKLE, _E_ANKLE, _A_ANKLE = _actuator_params(500)
_S_FOOT, _D_FOOT, _E_FOOT, _A_FOOT = _actuator_params(50)


MYOSKELETON_ARTICULATION = EntityArticulationInfoCfg(
    actuators=(
        BuiltinPositionActuatorCfg(
            target_names_expr=(
                "L5_S1_Flex_Ext",
                "L5_S1_Lat_Bending",
                "L4_L5_Flex_Ext",
                "L4_L5_Lat_Bending",
                "L3_L4_Flex_Ext",
                "L3_L4_Lat_Bending",
                "L2_L3_Flex_Ext",
                "L2_L3_Lat_Bending",
                "L1_L2_Flex_Ext",
                "L1_L2_Lat_Bending",
                "L1_T12_Flex_Ext",
                "L1_T12_Lat_Bending",
            ),
            stiffness=_S_SPINE_FB,
            damping=_D_SPINE_FB,
            effort_limit=_E_SPINE_FB,
            armature=_A_SPINE_FB,
        ),
        BuiltinPositionActuatorCfg(
            target_names_expr=(
                "L5_S1_axial_rotation",
                "L4_L5_axial_rotation",
                "L3_L4_axial_rotation",
                "L2_L3_axial_rotation",
                "L1_L2_axial_rotation",
                "L1_T12_axial_rotation",
            ),
            stiffness=_S_SPINE_AR,
            damping=_D_SPINE_AR,
            effort_limit=_E_SPINE_AR,
            armature=_A_SPINE_AR,
        ),
        BuiltinPositionActuatorCfg(
            target_names_expr=(
                "shoulder_elv_r",
                "shoulder1_r2_r",
                "shoulder_rot_r",
                "elbow_flex_r",
                "pro_sup",
                "shoulder_elv_l",
                "shoulder1_r2_l",
                "shoulder_rot_l",
                "elbow_flex_l",
                "pro_sup_l",
            ),
            stiffness=_S_ARM,
            damping=_D_ARM,
            effort_limit=_E_ARM,
            armature=_A_ARM,
        ),
        BuiltinPositionActuatorCfg(
            target_names_expr=("flexion_r", "deviation", "flexion_l", "deviation_l"),
            stiffness=_S_WRIST,
            damping=_D_WRIST,
            effort_limit=_E_WRIST,
            armature=_A_WRIST,
        ),
        BuiltinPositionActuatorCfg(
            target_names_expr=("hip_flexion_r", "hip_flexion_l"),
            stiffness=_S_HIP_FLEX,
            damping=_D_HIP_FLEX,
            effort_limit=_E_HIP_FLEX,
            armature=_A_HIP_FLEX,
        ),
        BuiltinPositionActuatorCfg(
            target_names_expr=("hip_adduction_r", "hip_adduction_l"),
            stiffness=_S_HIP_ADD,
            damping=_D_HIP_ADD,
            effort_limit=_E_HIP_ADD,
            armature=_A_HIP_ADD,
        ),
        BuiltinPositionActuatorCfg(
            target_names_expr=("hip_rotation_r", "hip_rotation_l"),
            stiffness=_S_HIP_ROT,
            damping=_D_HIP_ROT,
            effort_limit=_E_HIP_ROT,
            armature=_A_HIP_ROT,
        ),
        BuiltinPositionActuatorCfg(
            target_names_expr=("knee_angle_r", "knee_angle_l"),
            stiffness=_S_KNEE,
            damping=_D_KNEE,
            effort_limit=_E_KNEE,
            armature=_A_KNEE,
        ),
        BuiltinPositionActuatorCfg(
            target_names_expr=("ankle_angle_r", "ankle_angle_l"),
            stiffness=_S_ANKLE,
            damping=_D_ANKLE,
            effort_limit=_E_ANKLE,
            armature=_A_ANKLE,
        ),
        BuiltinPositionActuatorCfg(
            target_names_expr=(
                "subtalar_angle_r",
                "mtp_angle_r",
                "subtalar_angle_l",
                "mtp_angle_l",
            ),
            stiffness=_S_FOOT,
            damping=_D_FOOT,
            effort_limit=_E_FOOT,
            armature=_A_FOOT,
        ),
    ),
    soft_joint_pos_limit_factor=0.9,
)

MYOSKELETON_COLLISION = CollisionCfg(
    geom_names_expr=(".*",),
    contype=0,
    conaffinity=1,
    condim=3,
)

INIT_STATE = EntityCfg.InitialStateCfg(
    pos=(0.0, 0.0, 0.95),
    joint_pos={".*": 0.0},
    joint_vel={".*": 0.0},
)


def get_myoskeleton_robot_cfg() -> EntityCfg:
    """Get MyoSkeleton robot entity config (PD position controlled)."""
    return EntityCfg(
        spec_fn=get_myoskeleton_spec,
        articulation=MYOSKELETON_ARTICULATION,
        init_state=INIT_STATE,
        collisions=(MYOSKELETON_COLLISION,),
    )


# Action scale mapping (0.25 * effort_limit / stiffness)
MYOSKELETON_ACTION_SCALE: dict[str, float] = {}
for _a in MYOSKELETON_ARTICULATION.actuators:
    if isinstance(_a, BuiltinPositionActuatorCfg):
        _scale = 0.25 * _a.effort_limit / _a.stiffness
        for _n in _a.target_names_expr:
            MYOSKELETON_ACTION_SCALE[_n] = _scale
