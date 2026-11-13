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
`tied` reads the served class under the value tie (`tied.toml`), which `tied.py` prices over those periods.
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
#: the served class under the value tie: its public numbers and the bound on what the tie leaves untied
TIED = HERE / "tied.toml"
#: a block's work by kind, and its total in the policy's unit: the linears are priced at the card's dense int8 rate and
#: the rest at its BF16 rate, half of it, so a linear's MAC counts one and a BF16 MAC two
WORK = ("linears", "attention", "lm_head")
TOTAL = "total"
#: a policy's units. `c` is in two-input Boolean gates, as the Lean count counts a block Definition's by the gate
#: table (`definitions.boolean.gate_table`): not the block's gates at their widths (`Definition.gates`), which
#: `inference_blocks.budgets` sums into a number also called c
UNITS = {"c": "Boolean gates a block", "k": "fresh input bits a block", "n": "blocks"}
#: the work every figure is for, which the bound's premises are taken of (`min_cut`'s docstring has why): a registered
#: weight has no path from a fresh input, and a registered table's row is a function of θ and a few fresh gates
WORK_IS = ("training in bf16 words on the model's own matmul shapes, every weight of it and its embedding table outside "
           "the registered weights θ: (H) is false of a work that reads a registered weight or a registered table's row, "
           "so no figure here covers one")


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

    #: `served`, `tied`, `stated` or `built`
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
    it: the outputs of the linears the class leaves outside the circuit, the class's scan state row, and what the file
    names beside them (`[fresh]`, each key what it is and its value the bits)."""
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
    outside = sum(bits for _, bits, _ in parts)
    state = scan_state_bits(cls)
    parts.append(("the scan state row", state, f"{state // 16:,} words × 16 bits, the class's"))
    fresh = doc.get("fresh") or {}
    if not all(type(bits) is int and bits > 0 for bits in fresh.values()):
        raise ValueError(f"{path}: [fresh] names what a block reads beside its scan state row, each a positive integer "
                         "of bits")
    parts += [(name, bits, "") for name, bits in fresh.items()]
    label = {SERVED: "served", TIED: "tied"}.get(Path(path).resolve(), "stated")
    why = "this tree builds no circuit of the stated class, so no check ran on it here"
    return Policy(label, model, cls, c, k, n, work, tuple(parts), outside, source, {"verdict": "not run", "why": why})


@dataclass(frozen=True)
class Tied:
    """The served class under the value tie (`tied.toml`): its policy, and the bound on the words the tie leaves
    untied, a premise of every figure priced from it."""

    policy: Policy
    #: a node's bound on the untied words of its accepted execution phases, for a phase of `blocks` blocks
    words: int
    blocks: int
    #: an execution phase's seconds
    phase_seconds: int
    #: the MACs that feed one untied word, at most
    macs_a_word: int
    #: an untied word's fresh bits in the one block that reads it, at most
    word_bits: int

    def untied(self, blocks: float, seconds: float) -> float:
        """The bound on the untied words of ``blocks`` blocks served over ``seconds``, every node at its bound and
        every phase accepted. A node's bound is a share of one phase's ticket weight whatever the period's length, so
        a node has `words / blocks` of them for each block of one of its phases, and so have any number of nodes
        together: the untied words a block grow with nothing but the period's shortness."""
        return self.words / self.blocks * blocks / (seconds / self.phase_seconds)


def tied(path: Path = TIED) -> Tied:
    """The policy ``path`` states under the value tie (`stated`), with its `[untied]` table: a node's bound on the
    untied words (`words` for each `blocks` blocks of one execution phase of `phase_seconds`), the MACs that feed one
    at most (`macs_a_word`) and its bits (`word_bits`)."""
    p, keys = stated(path), ("words", "blocks", "phase_seconds", "macs_a_word", "word_bits")
    table = tomllib.loads(Path(path).read_text()).get("untied") or {}
    if set(table) != set(keys) or not all(type(table[x]) is int and table[x] > 0 for x in keys):
        raise ValueError(f"{path}: [untied] states {', '.join(keys)}, each a positive integer, and nothing else")
    if p.unexplained or p.outside:
        raise ValueError(f"{path}: a tied policy's k is its scan state row and what [fresh] names, with no linear left "
                         f"outside the circuit; {p.unexplained:,} of its k = {p.k:,} bits are unexplained and "
                         f"{p.outside:,} are outputs of linears outside the circuit")
    return Tied(p, *(table[x] for x in keys))


def bound(p: Policy, sh: Shape, blocks: float = 1.0) -> min_cut.Bound:
    """`min_cut.bound` at ``p`` on one block's work: `N` the session's work over the ``blocks`` it is divided into, in
    the policy's own MACs, `O = N o b`, and the circuit's size `c`."""
    N = p.work[TOTAL] / blocks
    return min_cut.bound(p.c, p.k, p.n, N=N, O=N * sh.outputs * min_cut.BF16 / sh.macs, size=p.c)


def fleet_periods(work: float) -> list[dict[str, Any]]:
    """For each of the slowdown calculator's fleets and monitoring periods (rho's `periods.FLEETS` and `PERIODS`): the
    period's seconds and the policy's blocks the fleet holds at capacity, each of ``work`` MACs. A fleet's rate is its
    locations' data-sheet dense rate (`location.<name>.flops`, two FLOPs a MAC)."""
    import periods as rho_periods

    reg, rows = Registry.load(), []
    for fleet, label in rho_periods.FLEETS:
        f = hardware.fleet(reg, fleet)
        rate = f.count * reg[f"location.{f.location}.flops"] / 2
        for name in rho_periods.PERIODS:
            seconds = hardware.monitoring_period(reg, name).seconds
            rows.append({"where": label, "fleet": fleet, "location": f.location, "locations": f.count, "period": name,
                         "seconds": seconds, "blocks": rate * seconds / work})
    return rows


def periods(work: float, limit: float) -> list[dict[str, Any]]:
    """`fleet_periods` with the training MACs the fleet's blocks hold at most at ``limit`` a block, and that as a share
    of the fleet's compute."""
    return [{**{x: v for x, v in r.items() if x != "seconds"}, "training_macs": r["blocks"] * min(limit, work),
             "share": min(1.0, limit / work)} for r in fleet_periods(work)]
