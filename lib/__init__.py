"""ScaleOp library (`lib`) — the importable API.

Only `utils` and `models` are implemented at M0; the remaining modules are
milestone-tagged placeholders filled in as Phase 1 progresses (M1–M4).
"""

__version__ = "0.1.0"

from . import models, utils  # noqa: F401  (eager: light deps only)

__all__ = ["utils", "models", "__version__"]
