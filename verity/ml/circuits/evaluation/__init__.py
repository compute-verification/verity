"""verity.ml.circuits.evaluation: computing values.

    reference   the plain-integer reference: the gate-by-gate oracle (`evaluate_region`, `evaluate_call`, still defined
                in `verity.ml.circuits.evaluate`), `evaluate` for a Definition on one input, `ScalarReference` for silicon
                functions no core Definition wraps yet, and `RegisteredTable`, a registered table `evaluate` reads lazily
    batch       `evaluate_batch` over numpy arrays, the kernel registry (`register_kernel`), and `self_check`
    bits        bit-sliced evaluation of Boolean Definitions (`evaluate_bits`, `evaluate_words`), 64 inputs per word

The reference defines what is correct; kernels are speed only, registered next to what they compute and checked against
the reference by `self_check`.  Nothing here is recorded, so nothing here can move a digest or a gate index.
"""
from __future__ import annotations

from .bits import evaluate_bits, evaluate_words  # noqa: F401
from .batch import Kernel, SelfCheck, evaluate_batch, kernels, register_kernel, sample_uniform, self_check  # noqa: F401
from .reference import (  # noqa: F401
    ExecutionTranscript,
    MissingValue,
    RegisteredTable,
    ScalarReference,
    evaluate,
    evaluate_call,
    evaluate_region,
)
