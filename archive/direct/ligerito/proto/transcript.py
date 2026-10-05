"""The prototype draws its coins through ``archive/direct/ligerito/transcript.py`` (the Coins protocol)."""
from __future__ import annotations

from ..transcript import Coins, FiatShamirCoins, LiveCoins, LocalCoins, Transcript  # noqa: F401

__all__ = ["Coins", "FiatShamirCoins", "LiveCoins", "LocalCoins", "Transcript"]
