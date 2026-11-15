"""`verity-vllm predict`: the ahead-of-time predictor's commands.

    verity-vllm predict node --row ROW --config CONFIG_JSON --wrapper step|request [--LP N --T N | --M N --pos0 N] --out DIR
    verity-vllm predict node-registration --row ROW --config CONFIG_JSON --program NODE_DESCRIPTOR --commit-out OUT ...
    verity-vllm predict compare ROW_DIR --row ROW --config CONFIG_JSON --out comparison.json [--workers N] [--predicted DIR]
    verity-vllm predict emit --row ROW --rank R --config CONFIG_JSON --trace DIR --out DIR --wrapper ...   (`aot`)
    verity-vllm predict row --row ROW --out DIR [--config CONFIG_JSON] [--jobs N]                (`aot`: a row's Build of record)
    verity-vllm predict compiled --role ROLE --device DEVICE --workload WL --out build.json      (`compiled`: the served Build)

`node-registration` registers a node Program's θ from its TP Commit of record (`node_registration`, the inference-only policy's full check).
`emit` writes into a traced Build row: a switched row's AOT Build (`aot`, `row_tp.TpRow.build`) runs it for each derive.
`row` writes a single-GPU row's whole Build of record from its declared inputs, with no trace, into a dir of its own.
"""

from __future__ import annotations

import gzip
import json
import os
import sys
from dataclasses import dataclass, field

from verity.circuits.codec import canonical_json

from verity_vllm.config import UsageError, option
from verity_vllm.records.schema_ids import PREDICT_NODE_SCHEMA

from . import aot as A
from . import compiled as PC
from . import node_registration as NR
from .node import build_node_program

__all__ = ["COMMANDS", "main"]


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
    cmd: str = field(default="node", init=False)


COMMANDS = {"node": NodeOptions, "node-registration": NR.Options, "compare": A.CompareOptions, "emit": A.EmitOptions,
            "row": A.RowOptions, **PC.COMMANDS}


def node_main(a: NodeOptions) -> int:
    """`node_descriptor.json.gz` and `node.json`: the summary, θ's parameters and which rank emitted each run of the body
    Definition's nodes (`spans`; the root's nodes are its input nodes, one per parameter's lowering, then one Call of the body)."""
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
    doc = {"schema": PREDICT_NODE_SCHEMA, "row": a.row, "context_digest": phi.digest(), **r.summary(),
           "theta": list(r.theta), "spans": [list(s) for s in r.spans]}
    with open(os.path.join(a.out, "node.json"), "w") as f:
        json.dump(doc, f, indent=1, sort_keys=True)
    print(json.dumps(r.summary()), file=sys.stderr)
    return 0


def main(a) -> int:
    if isinstance(a, (A.CompareOptions, A.EmitOptions, A.RowOptions)):
        return A.main(a)
    if isinstance(a, (PC.Options, PC.PinOptions)):
        return PC.main(a)
    if isinstance(a, NodeOptions):
        return node_main(a)
    return NR.main(a)
