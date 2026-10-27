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
    import torch as _torch
except ImportError:  # pure-circuit use (no torch): the analyses above are still available
    _torch = None
else:
    from verity_vllm.program.frontend.errors import CODES, DeadGateWarning, FrontendError  # noqa: F401
    from verity_vllm.program.frontend.program_identity import check_identity  # noqa: F401
    from verity_vllm.program.frontend.schemas import Arr, Dep, Tup  # noqa: F401
    from verity_vllm.program.frontend.torch_frontend import (Bound, ExportReport, Function, Ops, Prim, Registered,  # noqa: F401
                                                             batch, bind, definition_digest, export, export_report, function,
                                                             ops, scan)
    from verity_vllm.program.frontend.value_types import type_of  # noqa: F401
