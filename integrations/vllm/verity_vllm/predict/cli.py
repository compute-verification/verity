"""`verity-vllm predict`: the ahead-of-time predictor's commands.

    verity-vllm predict build --phi phi.json --wrapper step|request [--LP N --T N | --M N --pos0 N] --out DIR
    verity-vllm predict inventory --rows ROWS_ROOT --out inventory.json.gz
    verity-vllm predict score (--inventory inventory.json.gz | --rows ROWS_ROOT) --hf-home HF --out score.json [--md score.md]

None of them writes into a Build row: `score` and `build` refuse an output path inside a rows root they read.
"""

from __future__ import annotations

import gzip
import json
import os
import sys
from dataclasses import dataclass, field

from verity_vllm.config import UsageError, option

from . import inventory as INV
from . import score as S


@dataclass(frozen=True, kw_only=True)
class BuildOptions:
    """Predict one Program from φ and write its descriptor and summary (never into a Build row)."""
    phi: str = option("--phi", help="phi JSON (`verity-predict/phi/v1`)")
    wrapper: str = option("--wrapper", default="request", choices=["step", "request"])
    M: int = option("--M", type=int, default=1)
    pos0: int = option("--pos0", type=int, default=8)
    LP: int | None = option("--LP", type=int, default=None)
    T: int | None = option("--T", type=int, default=None)
    out: str = option("--out")
    cmd: str = field(default="build", init=False)


@dataclass(frozen=True, kw_only=True)
class InventoryOptions:
    """The traced facts of every row under a rows root (read-only)."""
    rows: str = option("--rows", help="rows root (<root>/<job>/<row>/build_summary.json)")
    out: str = option("--out", help="inventory JSON(.gz)")
    cmd: str = field(default="inventory", init=False)


COMMANDS = {"build": BuildOptions, "inventory": InventoryOptions, "score": S.Options}


def main(a) -> int:
    if isinstance(a, S.Options):
        return S.main(a)
    if a.cmd == "inventory":
        S._refuse_inside(a.out, [a.rows])
        inv = INV.inventory(a.rows)
        data = json.dumps(inv).encode()
        with open(a.out, "wb") as f:
            f.write(gzip.compress(data) if a.out.endswith(".gz") else data)
        print(f"[predict] inventory: {len(inv['rows'])} rows -> {a.out}", file=sys.stderr)
        return 0
    from verity.ml.circuits.codec import canonical_json
    from .build import build_program
    from .phi import load_phi
    if os.path.exists(os.path.join(a.out, "build_summary.json")) or os.path.basename(os.path.normpath(a.out)).startswith("build_"):
        raise UsageError(f"--out {a.out} looks like a Build row or Build dir: a prediction is never written into one")
    phi = load_phi(a.phi)
    if a.wrapper == "request" and (a.LP is None or a.T is None):
        raise UsageError("--wrapper request needs --LP and --T")
    r = build_program(phi, a.wrapper, M=a.M, pos0=a.pos0) if a.wrapper == "step" else build_program(phi, "request", LP=a.LP, T=a.T)
    os.makedirs(a.out, exist_ok=True)
    with gzip.open(os.path.join(a.out, "predicted_descriptor.json.gz"), "wb") as f:
        f.write(canonical_json(r.desc))
    with open(os.path.join(a.out, "predicted.json"), "w") as f:
        json.dump({"schema": "verity-predict/build/v1", "context_digest": phi.digest(), **r.summary()}, f, indent=1, sort_keys=True)
    print(json.dumps(r.summary()), file=sys.stderr)
    return 0
