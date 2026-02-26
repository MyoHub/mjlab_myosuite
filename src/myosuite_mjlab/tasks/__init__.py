"""Task registration entrypoint for myosuite_mjlab."""

from myosuite_mjlab.tasks.velocity import myoleg as _myoleg
from myosuite_mjlab.tasks.velocity import myoskeleton as _myoskeleton
from myosuite_mjlab.tasks.velocity import myolegtorso as _myolegtorso
from myosuite_mjlab.tasks.balance import _myolegtorso as _balance_myolegtorso  # noqa: F401

__all__ = ["_myoleg", "_myoskeleton", "_myolegtorso", "_balance_myolegtorso"]
