"""myosuite_mjlab package.

Importing this package eagerly registers all MyoSuite-based tasks with
``mjlab.tasks.registry`` under the ``MjlabMyoSuite-*`` task IDs.
"""

# Import task subpackages for side-effectful registration with mjlab.
# This keeps ``import myosuite_mjlab`` as the single entrypoint users need.
from . import tasks as _tasks  # noqa: F401
