"""verity.evaluation: computing values.

    reference   the plain-integer reference: the gate-by-gate oracle (`evaluate_region`, `evaluate_call`, still defined
                in `verity.ir.evaluate`), `evaluate` for one instance of a Definition, and `ScalarReference` for silicon
                functions no core Definition wraps yet
    batch       `evaluate_batch` over numpy arrays, the kernel registry (`register_kernel`), and `self_check`

The reference defines what is correct; kernels are speed only, registered next to what they compute and checked against
the reference by `self_check`.  Nothing here is recorded, so nothing here can move a digest or a gate index.
"""
from __future__ import annotations

from .batch import Kernel, SelfCheck, evaluate_batch, kernels, register_kernel, sample_uniform, self_check  # noqa: F401
from .reference import (  # noqa: F401
    ExecutionTranscript,
    MissingValue,
    ScalarReference,
    evaluate,
    evaluate_call,
    evaluate_region,
)
