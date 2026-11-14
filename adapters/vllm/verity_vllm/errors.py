"""The integration's one refusal type, `Refused`: what was asked is outside what the code serves or proves, said by reason.
It is `verity.errors.Refused`, which PoUW's serving code raises too, so the adapter catches both as one.

Lint P14 (`tests/lint/test_p14_one_way.py`) holds every other refusal class to its shrinking allowlist.
"""
from __future__ import annotations

from verity.errors import Refused

__all__ = ["Refused"]
