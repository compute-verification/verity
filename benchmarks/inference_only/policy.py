"""An inference-only policy's number: the least slowdown of training run inside the policy's inference blocks.

A policy is `(c, k, n)`: blocks of at most `c` Boolean gates, each reading at most `k` bits from outside itself and
outside θ, at most `n` of them (the Lean's `CompliesAtMost c k n`; `UNITS`). Its number is the slowdown `min_cut.bound`
gives it (`verity/protocols/inference_only/rho/rho/min_cut.py`, the arithmetic of the Lean statement its `HE` names) on
one block's work: `N` the block's work in the policy's own MACs, `O = N o b` its output wires (`o` the output values a
MAC of the model's forward matmuls, `Shape.outputs / Shape.macs`, and `b` a bf16 value's wires) and the circuit's size
`c`. That is `max(1, N / limit)`, `limit` the most of such work one block of the policy finishes
(`Bound.macs_a_block`): the number is 1 where the count lets a block hold all of its own work as training.

`stated` reads a policy's public numbers from a file (`served.toml`, whose keys README.md lists) and says what its `k`
is made of; `built.py` builds a class's policy from this tree. `periods` sets a block's limit beside the blocks a fleet
holds over a monitoring period, on the slowdown calculator's fleets and periods (`verity/protocols/inference_only/rho`).
"""
from __future__ import annotations

import sys
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
RHO = ROOT / "verity" / "protocols" / "inference_only" / "rho"
# first on the path: rho's `periods` imports rho's own `run`, not this benchmark's
sys.path.insert(0, str(RHO))

from rho import hardware, min_cut  # noqa: E402
from rho.params import Registry  # noqa: E402

#: the served policy's public numbers
SERVED = HERE / "served.toml"
#: a block's work by kind, and its total in the policy's unit: the linears are priced at the card's dense int8 rate and
#: the rest at its BF16 rate, half of it, so a linear's MAC counts one and a BF16 MAC two
WORK = ("linears", "attention", "lm_head")
TOTAL = "total"
#: a policy's units. `c` is in two-input Boolean gates, as the Lean count counts a block Definition's by the gate
#: table (`definitions.boolean.gate_table`): not the block's gates at their widths (`Definition.gates`), which
#: `inference_blocks.budgets` sums into a number also called c
UNITS = {"c": "Boolean gates a block", "k": "fresh input bits a block", "n": "blocks"}


@dataclass(frozen=True)
class Shape:
    """A model's forward matmuls at one position: each layer's linears and the LM head, `(outputs, inputs)` each."""

    layers: int
    linears: tuple[tuple[str, int, int], ...]
    head: tuple[int, int]

    @property
    def macs(self) -> int:
        return self.layers * sum(n * k for _, n, k in self.linears) + self.head[0] * self.head[1]

    @property
    def outputs(self) -> int:
        return self.layers * sum(n for _, n, _ in self.linears) + self.head[0]


@dataclass(frozen=True)
class Policy:
    """A policy's public numbers on one class, with what its `k` is made of and what checked it."""

    #: `served`, `stated` or `built`
    label: str
    model: str
    #: the class: slots, prompt ids, outputs
    cls: tuple[int, int, int]
    c: int
    k: int
    n: int
    #: one block's work in MACs by kind (`WORK`), and `TOTAL`
    work: dict[str, float]
    #: `k` by kind, `(kind, bits, how)` each; what they leave of `k` is unexplained
    parts: tuple[tuple[str, int, str], ...]
    #: the bits of `k` that are outputs of linears the class leaves outside the circuit
    outside: int
    source: str
    #: the check's verdict on it: `{"verdict": "refused" | "accepted" | "not run", ...}`
    check: dict[str, Any] = field(default_factory=dict)
    #: which blocks the policy declares
    division: str = "session"

    @property
    def unexplained(self) -> int:
        return self.k - sum(bits for _, bits, _ in self.parts)


def pinned(model: str) -> tuple[str, Path]:
    """The checkpoint ``model`` as the integration's manifests pin it: its revision and its `config.json` of record."""
    from verity_vllm.engine.profiles import dense_generic as DG

    revisions = sorted({str(e["revision"]) for e in DG.checkpoint_entries(DG.MANIFEST) if e.get("repo") == model})
    config = DG.pinned_hf_config(model, revisions[0]) if len(revisions) == 1 else None
    if config is None:
        raise ValueError(f"{model}: the integration's manifests ({DG.MANIFEST.relative_to(ROOT)}) pin "
                         f"{len(revisions) or 'no'} revision{'s' * (len(revisions) != 1)} of it, so its shape isn't known "
                         "here: name a model they pin at one revision, or pin this one there")
    return revisions[0], config


def shape(model: str) -> Shape:
    """``model``'s forward matmuls, from its config of record: the 2-D weights of a layer's record
    (`b1.layer_weights_type`) and the LM head's, the embedding's where the two are tied."""
    from verity_vllm.program.model import config_of

    from verity.definitions.registry import b1

    C, _ = config_of(pinned(model)[1])
    weights = dict(b1.weights_type(C).fields)
    head = weights.get("lm_head", weights["embed"])
    linears = tuple((name.removesuffix("_w"), t.n, t.elem.n) for name, t in weights["layers"].elem.fields
                    if name.endswith("_w"))
    return Shape(weights["layers"].n, linears, (head.n, head.elem.n))


def class_of(cls: tuple[int, int, int]) -> Any:
    """The integration's class of `(slots, prompt ids, outputs)`: its positions and its scan."""
    from verity_vllm.program.predict.window_class import WindowClass

    return WindowClass(*cls)


def scan_state_bits(cls: tuple[int, int, int]) -> int:
    """The bits of the scan state row a block of class ``cls`` reads (`scan_state.words`, u16 words)."""
    from verity_vllm.program.predict import scan_state

    return 16 * scan_state.words(class_of(cls).scan)


def stated(path: Path = SERVED) -> Policy:
    """The policy ``path`` states (`served.toml`'s keys), with its `k` by kind as the model's shape and the class give
    it: the outputs of the linears the class leaves outside the circuit, and the class's scan state row."""
    doc = tomllib.loads(Path(path).read_text())
    try:
        cls, work = tuple(doc["class"]), {x: doc["work"][x] for x in (*WORK, TOTAL)}
        model, c, k, n, source = doc["model"], doc["c"], doc["k"], doc["n"], doc["source"]
    except KeyError as e:
        raise ValueError(f"{path}: a policy file states source, model, class, c, k, n and [work]'s {', '.join(WORK)} and "
                         f"{TOTAL}; it has no {e}") from None
    if len(cls) != 3 or not all(type(x) is int and x > 0 for x in (*cls, c, k, n, *work.values())):
        raise ValueError(f"{path}: class is three positive integers (slots, prompt ids, outputs), and c, k, n and the "
                         "work's MACs positive integers")
    if work[TOTAL] != work["linears"] + 2 * (work["attention"] + work["lm_head"]):
        raise ValueError(f"{path}: [work] {TOTAL} = {work[TOTAL]:,} is not linears + 2 × (attention + lm_head): the "
                         "policy's unit counts a linear's MAC one and a BF16 MAC two")
    sh, positions = shape(model), class_of(cls).positions
    parts = []
    if bits := (doc.get("outside") or {}).get("linears"):
        parts += [(f"{name}'s outputs", sh.layers * positions * n_out * bits,
                   f"{sh.layers} layers × {positions} positions × {n_out:,} values × {bits} bits")
                  for name, n_out, _ in sh.linears]
    state = scan_state_bits(cls)
    parts.append(("the scan state row", state, f"{state // 16:,} words × 16 bits, the class's"))
    label = "served" if Path(path).resolve() == SERVED else "stated"
    why = "this tree builds no circuit of the stated class, so no check ran on it here"
    outside = sum(bits for _, bits, _ in parts[:-1])
    return Policy(label, model, cls, c, k, n, work, tuple(parts), outside, source, {"verdict": "not run", "why": why})


def bound(p: Policy, sh: Shape, blocks: float = 1.0) -> min_cut.Bound:
    """`min_cut.bound` at ``p`` on one block's work: `N` the session's work over the ``blocks`` it is divided into, in
    the policy's own MACs, `O = N o b`, and the circuit's size `c`."""
    N = p.work[TOTAL] / blocks
    return min_cut.bound(p.c, p.k, p.n, N=N, O=N * sh.outputs * min_cut.BF16 / sh.macs, size=p.c)


def periods(work: float, limit: float) -> list[dict[str, Any]]:
    """For each of the slowdown calculator's fleets and monitoring periods (rho's `periods.FLEETS` and `PERIODS`): the
    policy's blocks the fleet holds at capacity, each of ``work`` MACs, the training MACs they hold at most at ``limit``
    a block, and that as a share of the fleet's compute. A fleet's rate is its locations' data-sheet dense rate
    (`location.<name>.flops`, two FLOPs a MAC)."""
    import periods as rho_periods

    reg, rows = Registry.load(), []
    for fleet, label in rho_periods.FLEETS:
        f = hardware.fleet(reg, fleet)
        rate = f.count * reg[f"location.{f.location}.flops"] / 2
        for name in rho_periods.PERIODS:
            blocks = rate * hardware.monitoring_period(reg, name).seconds / work
            rows.append({"where": label, "fleet": fleet, "location": f.location, "locations": f.count, "period": name,
                         "blocks": blocks, "training_macs": blocks * min(limit, work), "share": min(1.0, limit / work)})
    return rows
