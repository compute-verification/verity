"""verity_vllm.program.frontend — authoring frontends to `Program`.

    Gate -> exactly one Value<w>.
    Composite interfaces -> ordered collections of value references.
    Commitment positions -> physical authentication locations for those values,
                            not semantic components of gate outputs.

`torch_frontend` (needs torch) is the PyTorch authoring frontend. `correspond` (program correspondence across
reference encodings) and `provenance` (export/derivation provenance records) are pure `verity.ml.circuits` and import
without torch. Dead gates are `verity.ml.circuits.liveness`.
"""

from verity_vllm.program.frontend.correspond import CorrespondenceReport, correspond  # noqa: F401
from verity_vllm.program.frontend.provenance import DerivationRule, RuleApplication, registry_version  # noqa: F401

try:
    import torch as _torch  # noqa: F401
except ImportError:  # pure-IR use (no torch): the analyses above are still available
    _torch = None
else:
    from verity_vllm.program.frontend.torch_frontend import (Arr, Bound, CODES, Dep, ExportReport, DeadGateWarning, FrontendError, Function,  # noqa: F401
                                 Ops, Prim, Registered, Tup, batch, bind, check_identity, definition_digest, export,
                                 export_report, function, ops, scan, type_of)
