"""MyoLegTorso robot config: composed from MyoSuite assets via MjSpec."""

from __future__ import annotations

import os
import shutil
import tempfile
from pathlib import Path

import mujoco

from mjlab.actuator import XmlMuscleActuatorCfg
from mjlab.actuator.actuator import TransmissionType
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.utils.spec_config import CollisionCfg

_SPEC_CACHE: mujoco.MjSpec | None = None


def _myosuite_myo_sim() -> Path:
    """Return myo_sim root.

    Canonical location for a full pip install of myosuite is
    myosuite/simhive/myo_sim. Use MYOSUITE_MJLAB_MYO_SIM to override.
    """
    env_path = os.environ.get("MYOSUITE_MJLAB_MYO_SIM")
    if env_path:
        path = Path(env_path).resolve()
        if path.exists() and (path / "leg" / "assets").exists():
            return path
        raise FileNotFoundError(
            f"MYOSUITE_MJLAB_MYO_SIM={env_path} missing leg/assets. "
            "Point to a myo_sim root (e.g. mjlab/submodules/myo_sim)."
        )
    try:
        import myosuite  # type: ignore
    except ImportError as exc:
        raise FileNotFoundError("myosuite is required.") from exc
    root = Path(myosuite.__file__).resolve().parent
    path = root / "simhive" / "myo_sim"
    if not path.exists():
        raise FileNotFoundError(
            f"myosuite simhive/myo_sim not found at {path}. "
            "Use a full myosuite install (models at myosuite/simhive/myo_sim) or set "
            "MYOSUITE_MJLAB_MYO_SIM to a myo_sim root."
        )
    return path


def _mjlab_submodule_leg_xml() -> Path | None:
    """Return path to myolegstorso_mjlab.xml in mjlab submodule if present."""
    try:
        import mjlab as mj

        mjlab_src = Path(mj.__file__).resolve().parent
        # submodules live next to src in mjlab repo
        candidate = (
            mjlab_src.parent
            / "submodules"
            / "myo_sim"
            / "leg"
            / "myolegstorso_mjlab.xml"
        )
        return candidate if candidate.exists() else None
    except Exception:
        return None


def get_myolegtorso_spec() -> mujoco.MjSpec:
    """Load the MyoLegTorso entity-only MjSpec (robot, no terrain)."""
    global _SPEC_CACHE
    if _SPEC_CACHE is not None:
        return _SPEC_CACHE

    myo_sim_root: Path | None = None  # set when we load from env or submodule
    # 1) mjlab repo submodule (when mjlab is run from source)
    xml_path: Path | None = _mjlab_submodule_leg_xml()
    if xml_path is not None:
        spec = mujoco.MjSpec.from_file(str(xml_path))
        myo_sim_root = xml_path.resolve().parent.parent  # leg -> myo_sim
    else:
        # 2) Load from MYOSUITE_MJLAB_MYO_SIM/leg/ if that XML exists
        env_myo_sim = os.environ.get("MYOSUITE_MJLAB_MYO_SIM")
        if env_myo_sim:
            root = Path(env_myo_sim).resolve()
            direct_xml = root / "leg" / "myolegstorso_mjlab.xml"
            if direct_xml.exists():
                spec = mujoco.MjSpec.from_file(str(direct_xml))
                myo_sim_root = root
            else:
                spec = None
        else:
            spec = None
        if spec is None:
            # 3) Standalone: build a temp tree so includes resolve (our XML + myosuite assets).
            myo_sim = _myosuite_myo_sim()
            myo_sim_root = myo_sim
            our_dir = Path(__file__).resolve().parent
            leg_assets = myo_sim / "leg" / "assets"
            if not leg_assets.exists():
                raise FileNotFoundError(
                    f"myosuite leg assets not found at {leg_assets}"
                )

            with tempfile.TemporaryDirectory(prefix="myolegtorso_mjlab_") as tmp:
                tmp = Path(tmp)
                # Recreate the myo_sim layout expected by the XML includes:
                # tmp/myo_sim/{leg,torso,head,meshes}/...
                tmp_myo = tmp / "myo_sim"
                tmp_myo.mkdir()

                # leg: our custom XML plus assets copied from myosuite.
                tmp_leg = tmp_myo / "leg"
                tmp_leg.mkdir()
                shutil.copy(
                    our_dir / "myolegstorso_mjlab.xml",
                    tmp_leg / "myolegstorso_mjlab.xml",
                )
                (tmp_leg / "assets").mkdir()
                for f in leg_assets.iterdir():
                    (tmp_leg / "assets" / f.name).symlink_to(f)
                shutil.copy(
                    our_dir / "assets" / "mjlab_bootstrap.xml",
                    tmp_leg / "assets" / "mjlab_bootstrap.xml",
                )

                # torso/head: only assets need to be reachable via "../torso/assets/..."
                tmp_torso = tmp_myo / "torso"
                tmp_torso.mkdir()
                (tmp_torso / "assets").symlink_to(myo_sim / "torso" / "assets")
                tmp_head = tmp_myo / "head"
                tmp_head.mkdir()
                (tmp_head / "assets").symlink_to(myo_sim / "head" / "assets")

                # Some torso assets reference meshes via paths like "../meshes/..."
                # or "myo_sim/meshes/..." relative to the torso directory.
                if (myo_sim / "meshes").exists():
                    (tmp_myo / "meshes").symlink_to(myo_sim / "meshes")
                    # Make "torso/myo_sim/meshes" resolve to the same meshes directory.
                    (tmp_torso / "myo_sim").symlink_to(tmp_myo)

                spec = mujoco.MjSpec.from_file(str(tmp_leg / "myolegstorso_mjlab.xml"))

    spec.option.jacobian = mujoco.mjtJacobian.mjJAC_SPARSE
    if myo_sim_root is None:
        myo_sim_root = _myosuite_myo_sim()
    spec.meshdir = str(myo_sim_root)
    spec.texturedir = str(myo_sim_root)
    _SPEC_CACHE = spec
    return spec


MYOLEGSTORSO_ARTICULATION = EntityArticulationInfoCfg(
    actuators=(
        XmlMuscleActuatorCfg(
            transmission_type=TransmissionType.TENDON,
            target_names_expr=(r".*_tendon",),
        ),
    ),
)

MYOLEGSTORSO_COLLISION = CollisionCfg(
    geom_names_expr=(".*",),
    contype=0,
    conaffinity=1,
    condim=3,
)

INIT_STATE = EntityCfg.InitialStateCfg(
    pos=(0.0, 0.0, 0.92),
    rot=(1.0, 0.0, 0.0, 0.0),
    joint_pos=None,  # use keyframe
    joint_vel={".*": 0.0},
)


def get_myolegtorso_robot_cfg() -> EntityCfg:
    """Get MyoLegTorso robot entity config."""
    return EntityCfg(
        spec_fn=get_myolegtorso_spec,
        articulation=MYOLEGSTORSO_ARTICULATION,
        init_state=INIT_STATE,
        collisions=(MYOLEGSTORSO_COLLISION,),
    )
