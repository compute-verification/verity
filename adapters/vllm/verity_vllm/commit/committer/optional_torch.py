"""`torch`, or None where there is none to import.

The host committer's offline core needs no torch: the roots of a stream it is handed, the openings and their check, the identity
check. So the modules that hold it beside what stages on a device read `torch` from here, and import where torch is absent.
"""

from __future__ import annotations

try:
    import torch
except ImportError:                      # the offline core (tests, CPU cross-checks) needs no torch
    torch = None  # type: ignore[assignment]

__all__ = ["torch"]
