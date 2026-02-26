"""Wrapper script to play MyoSuite-mjlab tasks."""

# Set OpenGL backend before any import that loads mujoco.
import os

if not os.environ.get("MUJOCO_GL"):
    os.environ["MUJOCO_GL"] = "egl"
if not os.environ.get("MUJOCO_EGL_DEVICE_ID"):
    os.environ["MUJOCO_EGL_DEVICE_ID"] = "0"

# Register myosuite_mjlab tasks before mjlab CLI reads the registry.
import myosuite_mjlab.tasks  # noqa: F401

import mjlab.scripts.play


def main():
    mjlab.scripts.play.main()


if __name__ == "__main__":
    main()
