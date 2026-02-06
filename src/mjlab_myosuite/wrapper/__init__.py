"""Public wrapper package for MyoSuite environments.

This package exposes `MyoSuiteVecEnvWrapper` as the main entry point while
keeping the public import path stable:

    from mjlab_myosuite.wrapper import MyoSuiteVecEnvWrapper
"""

from .vec_env_wrapper import MyoSuiteVecEnvWrapper

__all__ = ["MyoSuiteVecEnvWrapper"]
