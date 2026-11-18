"""circuit-check's sidecar of the predictor: a model's served bindings, which only the predictor's package may build."""
from __future__ import annotations

from typing import Any


def served_bindings(config: str, cls: str) -> list[Any]:
    """The served bindings of the model a φ file registers (``config``: `Phi.to_json()`, as a served run states φ) at the window
    class ``cls`` (``SLOTS,PROMPT,OUTPUTS[,LANES,EVERY]``): the Definition of each proof block of its class circuit, the
    templates of `window_circuit.lower`'s units (unwired: the units alone), by id."""
    import json
    from pathlib import Path

    from verity_vllm.engine.served_seed import DEFAULT_SEED
    from verity_vllm.program.predict import window_circuit as WC
    from verity_vllm.program.predict.phi import Phi
    from verity_vllm.program.predict.steps import Registered

    phi = Phi.from_json(json.loads(Path(config).read_text()))
    low = WC.lower(Registered.of(phi, DEFAULT_SEED), WC.WindowClass(*(int(x) for x in cls.split(","))), wired=False)
    return sorted((fn for fn, _ in low.circuit.templates), key=lambda fn: fn.id)
