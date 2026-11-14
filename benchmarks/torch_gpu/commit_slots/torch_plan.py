"""The serve-time commit call's plan (`verity/commitments/cuda/commit_slots/slots_gpu.py`) on torch, with the benchmarks'
build directory and variant (`../sha512_rows/torch_engine.py`): ``slots_gpu.Plan`` whose engine's ``runtime()`` is torch,
which verity doesn't import.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "sha512_rows"))
import torch_engine as T

from verity.commitments.cuda.commit_slots import slots as S
from verity.commitments.cuda.commit_slots import slots_gpu as G


class Engine(T.Torch, G.Plan.ENGINE):
    """slots_gpu's engine on torch."""


class Plan(G.Plan):
    """``slots_gpu.Plan`` on torch, building under ``torch_engine.BUILD`` on ``torch_engine.VARIANT`` unless given."""

    ENGINE = Engine

    def __init__(self, context: bytes, layout: S.CommitShape, device: Any, *args: Any, build: Path = T.BUILD,
                 default: str = T.VARIANT, **kw: Any) -> None:
        super().__init__(context, layout, device, *args, build=build, default=default, **kw)
