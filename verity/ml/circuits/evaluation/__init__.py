"""verity.ml.circuits.evaluation: computing values.

    reference   the plain-integer reference: the gate-by-gate oracle (`evaluate_region`, `evaluate_call`) over a `Trace`,
                `evaluate` for a Definition on one input, `ScalarReference` for silicon functions no core Definition wraps
                yet, and `RegisteredTable`, a registered table `evaluate` reads lazily
    batch       `evaluate_batch` over numpy arrays, the kernel registry (`register_kernel`), and `self_check`
    bits        bit-sliced evaluation of Boolean Definitions (`evaluate_bits`, `evaluate_words`), 64 inputs per word

The reference defines what is correct; kernels are speed only, registered next to what they compute and checked against
the reference by `self_check`.  A Boolean Definition takes its word Definition's kernel through the word view
(`word_kernel`), so a word Definition is the Boolean one's accelerator exactly when it has a kernel.  Nothing here is
recorded, so nothing here can move a digest or a gate index.
"""
from __future__ import annotations

from .bits import evaluate_bits, evaluate_words  # noqa: F401
from .batch import (  # noqa: F401
    Kernel,
    SelfCheck,
    evaluate_batch,
    kernels,
    register_kernel,
    sample_uniform,
    self_check,
    word_kernel,
)
from .reference import (  # noqa: F401
    MissingValue,
    RegisteredTable,
    ScalarReference,
    Trace,
    evaluate,
    evaluate_call,
    evaluate_region,
)
