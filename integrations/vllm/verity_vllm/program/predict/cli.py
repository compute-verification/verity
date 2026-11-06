"""`verity-vllm predict`: the ahead-of-time predictor's commands.

    verity-vllm predict build --phi phi.json --wrapper step|request [--LP N --T N | --M N --pos0 N] --out DIR
    verity-vllm predict node --row ROW --config CONFIG_JSON --wrapper step|request [--LP N --T N | --M N --pos0 N] --out DIR
    verity-vllm predict node-registration --row ROW --config CONFIG_JSON --program NODE_DESCRIPTOR --commit-out OUT ...
    verity-vllm predict inventory --rows ROWS_ROOT --out inventory.json.gz
    verity-vllm predict score (--inventory inventory.json.gz | --rows ROWS_ROOT) --hf-home HF --out score.json [--md score.md]
    verity-vllm predict compare ROW_DIR --row ROW --config CONFIG_JSON --out comparison.json [--workers N] [--predicted DIR]
    verity-vllm predict emit --row ROW --rank R --config CONFIG_JSON --trace DIR --out DIR --wrapper ...   (`aot`)
    verity-vllm predict row --row ROW --out DIR [--config CONFIG_JSON] [--jobs N]                (`aot`: a row's Build of record)
    verity-vllm predict compiled --role ROLE --device DEVICE --workload WL --out build.json      (`compiled`: the served Build)

`node-registration` registers a node Program's θ from its TP Commit of record (`node_registration`, the inference-only policy's full check).
`emit` writes into a traced Build row: a switched row's AOT Build (`aot`, `row_tp.TpRow.build`) runs it for each derive.
`row` writes a single-GPU row's whole Build of record from its declared inputs, with no trace, into a dir of its own.
`score` and `build` refuse an output path inside a rows root they read.
"""

from __future__ import annotations

import gzip
import json
import os
import sys
from dataclasses import dataclass, field

from verity_vllm.config import UsageError, option

from . import aot as A
from . import compiled as PC
from . import inventory as INV
from . import node_registration as NR
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
class NodeOptions:
    """Predict a tensor-parallel row's node Program (`node`: its ranks' forwards joined at their collectives) and write its
    descriptor and summary (never into a Build row)."""
    row: str = option("--row")
    config: str = option("--config", help="the pinned checkpoint's config.json")
    wrapper: str = option("--wrapper", default="request", choices=["step", "request"])
    M: int = option("--M", type=int, default=1)
    pos0: int = option("--pos0", type=int, default=8)
    LP: int | None = option("--LP", type=int, default=None)
    T: int | None = option("--T", type=int, default=None)
    out: str = option("--out")
    check_ranks: bool = option("--check-ranks", default=False, action="store_true",
                               help="build each rank's own Program (body only) and hold the node's run of it equal, node for node")
    cmd: str = field(default="node", init=False)


@dataclass(frozen=True, kw_only=True)
class InventoryOptions:
    """The traced facts of every row under a rows root (read-only)."""
    rows: str = option("--rows", help="rows root (<root>/<job>/<row>/build_summary.json)")
    out: str = option("--out", help="inventory JSON(.gz)")
    cmd: str = field(default="inventory", init=False)


COMMANDS = {"build": BuildOptions, "node": NodeOptions, "node-registration": NR.Options, "inventory": InventoryOptions,
            "score": S.Options, "compare": A.CompareOptions, "emit": A.EmitOptions, "row": A.RowOptions, **PC.COMMANDS}


def node_main(a: NodeOptions) -> int:
    """`node_descriptor.json.gz` and `node.json`: the summary, θ's parameters and which rank emitted each run of the body
    Definition's nodes (`spans`; the root's nodes are its input nodes, one per parameter's lowering, then one Call of the body)."""
    from verity.ml.circuits.codec import canonical_json
    from .node import build_node_program, check_ranks
    if os.path.exists(os.path.join(a.out, "build_summary.json")) or os.path.basename(os.path.normpath(a.out)).startswith("build_"):
        raise UsageError(f"--out {a.out} looks like a Build row or Build dir: a prediction is never written into one")
    if a.wrapper == "request" and (a.LP is None or a.T is None):
        raise UsageError("--wrapper request needs --LP and --T")
    phi = A.phi_at(a.row, 0, a.config)
    shape = {"M": a.M, "pos0": a.pos0} if a.wrapper == "step" else {"LP": a.LP, "T": a.T}
    r = build_node_program(phi, a.wrapper, **shape)
    os.makedirs(a.out, exist_ok=True)
    with gzip.open(os.path.join(a.out, "node_descriptor.json.gz"), "wb") as f:
        f.write(canonical_json(r.desc))
    doc = {"schema": "verity-predict/node/v1", "row": a.row, "context_digest": phi.digest(), **r.summary(),
           "theta": list(r.theta), "spans": [list(s) for s in r.spans]}
    if a.check_ranks:
        r.desc = None
        doc["ranks_checked"] = check_ranks(phi, r, **shape)
    with open(os.path.join(a.out, "node.json"), "w") as f:
        json.dump(doc, f, indent=1, sort_keys=True)
    print(json.dumps(r.summary()), file=sys.stderr)
    return 0


def main(a) -> int:
    if isinstance(a, S.Options):
        return S.main(a)
    if isinstance(a, (A.CompareOptions, A.EmitOptions, A.RowOptions)):
        return A.main(a)
    if isinstance(a, (PC.Options, PC.PinOptions)):
        return PC.main(a)
    if isinstance(a, NodeOptions):
        return node_main(a)
    if isinstance(a, NR.Options):
        return NR.main(a)
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
