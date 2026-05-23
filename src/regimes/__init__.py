"""regimes — an autonomous eval-improvement loop on the ActiveGraph runtime.

Public surface is intentionally small. The loop, its events, and its
behaviors are defined inside this package; the rest of the world reads
the event log to see what happened.
"""

from __future__ import annotations

__version__ = "0.1.0"

from regimes.split import Split, load_split

__all__ = ["Split", "load_split", "__version__"]
