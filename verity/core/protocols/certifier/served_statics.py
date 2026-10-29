"""The served rows held to their steps' Programs: each block of G steps of a window's committed served rows (`served_rows`,
one registered row a block) against the RoPE statics of the Programs those steps ran, so a prover can't register cheaper
rows at serve time than the batch it ran. The six-pass run's rows (user-1 never admitted) give step 6 no row for
the 29 prefill tokens its Program computes, and the tie refuses them.

    ServedStaticsTie_v1{R, G, IW, L, T} (pub, srb) -> verdict[16]
        ServedStaticsStep_v1{R, L, T} (rows, statics) -> (count, pos)

**The statics** (`statics_bits`). A step's Program computes each row of its batch, in the runner's batch order (advice v0:
the order of the step's rows), at the positions its RoPE reads as static structure: slot k's n tokens at start_k …
start_k + n_k − 1 (`RotaryEmbeddingRule`: a computed position is refused). That (valid, start, n) per slot is the step's
statics, public, as the Program is, and the verifier holds its own copy (`statics_bits` of the Programs' rows). An idle
step's statics are zero.

**The rows** (each zero-padded to whole 16-bit words):
- ``pub``, public: the block's G steps' statics, each step R slots of (valid, start, n), `served_tie._tie_fields`' order,
  opened to the verifier's own copy;
- ``srb``, the block's served rows, leaf w·(P / G) + j of the served-rows registration (`served_rows.block_bits`: G steps of
  R slots of (valid, request, kind, start, n), `compile_slice.rows_bits`' packing). Its b‖c is held to that leaf by the Lean
  verifier's registered reads (`served_rows.reads_port`, ``flock-verify verify --registered``), so the unit hashes the
  block's rows and no more.

**The statement** (`check`). For every step of the block and every slot k: bit 0 ``count``, the served row's valid and n
are the statics'; bit 1 ``pos``, its start is the statics'. ``ACCEPT`` = 0x3. A slot the Program has no row in is zero in
both. The request ids and kinds are not the Program's to show: ``IoTieRequest`` holds each prefill slot's ids to its
certified request's prompt ids at those positions, and the egress tie each decode slot's input to the request's last
sample.

**Positions.** The tie reads the Program's positions, not the Commit's: on vLLM's eager path no Program reads a committed
position. The compiled path's RoPE reads ``c00.input/positions``, a committed value (`verity_vllm.program.predict.compiled`); there
the committed positions are what the compute read, and `served_tie`'s routed form holds the served rows to them.

**Selection: every unit, always proved**, the idle blocks too (`Selection` All, as a transfer-gate unit is). The tie is a
binding, not a check on work: a block left untied makes its served rows the prover's word, and the compile slice reads
every step's rows, an idle one's included, so nothing but its tie shows an idle block's rows zero. The charge feeds the
network verdict, an AND over every window with no ε. A window is its P / G units, block j = 0 … P / G − 1 each once,
under one template.

**Cost** (`size`): R·(1 + LW + TW) − 2 ANDs a step, the equalities ANDed, and 2·(G − 1) to AND the steps' flags; the
unit's C-Flock block is set by its two rows' in-circuit hashing, G·R·(3 + IW + 2·(LW + TW)) bits.
"""
from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Sequence

from verity.ml.circuits import boolean as BL
from verity.ml.circuits.boolean.forms import Circuit, and_reduce
from verity.ml.circuits.defs import CompositeDefinition, bind
from verity.ml.circuits.evaluation.bits import evaluate_bits
from verity.ml.circuits.types import Array
from verity.ml.commitments.gates.frame_v3 import _part
from verity.core.protocols.certifier import head_check as HC, replay_chain as RC, served_rows as SR, served_tie as ST
from verity.core.protocols.certifier.head_check import BIT

STATICS = ("R", "G", "IW", "L", "T")
STEP = ("R", "L", "T")
WORD = 16
FLAGS = ("count", "pos")
ACCEPT = (1 << len(FLAGS)) - 1


def widths(S) -> SimpleNamespace:
    """A step's served-row bits SB, the TB of them the tie reads (valid, start, n), and LW, TW."""
    if S.G < 1:
        raise ValueError(f"a block is G ≥ 1 steps, not G = {S.G!r}")
    Z = ST._step_widths(S)
    return SimpleNamespace(LW=Z.LW, TW=Z.TW, SB=RC._bits_of(ST._step_fields(S)), TB=RC._bits_of(ST._tie_fields(S)))


def rows(S) -> list[tuple[str, int]]:
    """The unit's rows and their widths in bits."""
    Z = widths(S)
    return [("pub", ST._words(S.G * Z.TB)), ("srb", ST._words(S.G * Z.SB))]


# --- the circuit -------------------------------------------------------------------------------------------------------------

def _sig_step(S):
    TB = RC._bits_of(ST._tie_fields(S))
    return (("rows", Array(TB, BIT)), ("statics", Array(TB, BIT))), Array(2, BIT)


def _build_step(C, S, rows_, statics):
    f, s = RC._split(rows_, ST._tie_fields(S)), RC._split(statics, ST._tie_fields(S))

    def same(*names):
        return and_reduce(C, [(a ^ b).inv() for nm in names for x, y in zip(f[nm], s[nm]) for a, b in zip(x, y)])

    return [[same("valid", "n"), same("start")]]


ServedStaticsStep = _part(
    "ServedStaticsStep", STEP, _sig_step, _build_step,
    "(count, pos) of a step's served rows (its slots' valid, start and n) against its Program's statics: count iff every "
    "slot's valid and n are the statics', pos iff every slot's start is")


def _sig(S):
    return tuple((r, Array(n, BIT)) for r, n in rows(S)), Array(16, BIT)


def _body(B, S, pub, srb):
    Z, K = widths(S), HC._Consts(B)
    pr, sr = HC._refs(pub), HC._refs(srb)
    step = bind(ServedStaticsStep, **{k: getattr(S, k) for k in STEP})
    counts, poss = [], []
    for g in range(S.G):
        tie = ST._tie(S, sr[g * Z.SB:(g + 1) * Z.SB])
        o = HC._refs(B.call(step, HC._coll(Array(Z.TB, BIT), tie), HC._coll(Array(Z.TB, BIT), pr[g * Z.TB:(g + 1) * Z.TB])))
        counts.append(o[0])
        poss.append(o[1])
    return HC._coll(Array(16, BIT), [ST._all(B, counts), ST._all(B, poss), *K.bits(0, 16 - len(FLAGS))])


ServedStaticsTie = CompositeDefinition(
    "ServedStaticsTie", 1, STATICS, _sig, _body, conformance="exact-model-tested",
    doc="a block of G steps' served rows held to their Programs' statics (verdict bits count, pos): each step's "
        "`ServedStaticsStep`, the steps' bits ANDed")


def statics(S, G: int | None = None) -> dict:
    """The tie's statics for a compile slice's statics `S` (its R, P, IW, and ReplayStep_v1's L and T) and a unit's G steps
    (the whole window by default), G dividing P as the served rows' blocks are (`served_rows.register`): the tie's
    statics don't hold P, so this is where G is held to it."""
    return {"R": S.R, "G": SR._blocks(S, G), "IW": S.IW, "L": S.L, "T": S.T}


def unit(**st):
    return bind(ServedStaticsTie, **st)


# --- the specification -------------------------------------------------------------------------------------------------------

Program = Sequence[tuple[int, int]] | None


def statics_bits(S, programs: Sequence[Program]) -> list[int]:
    """The ``pub`` row of G steps' Programs, each its rows in batch order as (start, n) (None: an idle step): per step R slots
    of (valid, start, n) in `served_tie._tie_fields`' order, zero-padded to whole words."""
    Z = widths(S)
    if len(programs) > S.G:
        raise ValueError(f"{len(programs)} steps over a block's G = {S.G}")
    out = []
    for prog in list(programs) + [None] * (S.G - len(programs)):
        prog = list(prog or ())
        if len(prog) > S.R:
            raise ValueError(f"a step of {len(prog)} rows is over R = {S.R}")
        slot = {"valid": [], "start": [], "n": []}
        for k in range(S.R):
            start, n = prog[k] if k < len(prog) else (0, 0)
            slot["valid"].append(ST._bits_int(int(k < len(prog)), 1))
            slot["start"].append(ST._bits_int(start, Z.LW))
            slot["n"].append(ST._bits_int(n, Z.TW))
        out += [b for nm, _c, _w in ST._tie_fields(S) for e in slot[nm] for b in e]
    return out + [0] * (dict(rows(S))["pub"] - len(out))


def program_of(window: Sequence) -> list[Program]:
    """The Programs' rows a window's served rows (`served_rows.windows`' form) describe: each step's (start, n) in slot order,
    as the compiler of an advice whose rows they are lays the step out."""
    return [None if s is None else [(r[3], r[4]) for r in sorted(s)] for s in window]


def check(S, unit_rows: dict) -> dict[str, bool]:
    """The statement's flags on a unit's rows (bits each), in integers."""
    Z = widths(S)
    count = pos = True
    for g in range(S.G):
        f = ST._fields(S, unit_rows["srb"][g * Z.SB:(g + 1) * Z.SB])
        s = {k: [ST._int(e) for e in v]
             for k, v in RC._split(list(unit_rows["pub"][g * Z.TB:(g + 1) * Z.TB]), ST._tie_fields(S)).items()}
        count &= f["valid"] == s["valid"] and f["n"] == s["n"]
        pos &= f["start"] == s["start"]
    return {"count": count, "pos": pos}


def verdict(flags: dict) -> int:
    return sum(int(flags[f]) << k for k, f in enumerate(FLAGS))


@dataclass(frozen=True)
class Case:
    rows: dict
    flags: dict
    verdict: int


def unit_rows(S, srb: Sequence[int], programs: Sequence[Program]) -> dict:
    """A unit's rows: its block's served rows as registered (`served_rows.block_bits`) and its steps' Programs."""
    out = {"pub": statics_bits(S, programs), "srb": list(srb)}
    for r, n in rows(S):
        if len(out[r]) != n:
            raise ValueError(f"row {r}: {len(out[r])} bits, not {n}")
    return out


def case(S, srb: Sequence[int], programs: Sequence[Program]) -> Case:
    out = unit_rows(S, srb, programs)
    flags = check(S, out)
    return Case(out, flags, verdict(flags))


def evaluate(st: dict, cases: Sequence[Case]) -> list[int]:
    S = SimpleNamespace(**st)
    args = [[c.rows[r] for c in cases] for r, _n in rows(S)]
    out = evaluate_bits(unit(**st), *args).reshape(len(cases), -1).tolist()
    return [sum(int(b) << j for j, b in enumerate(o)) for o in out]


def _count(build, params) -> int:
    C = Circuit(cse=True)
    ins = [C.input(f"p{i}", w) for i, t in enumerate(params) for w in BL.elements(t)]
    out = build(C, *ins)
    C.output("out", [b for e in out for b in e])
    return C.counts()["ands"]


def step_size(**st) -> dict[str, int]:
    """One step's ANDs (`ServedStaticsStep`)."""
    S = SimpleNamespace(**{k: st[k] for k in STEP})
    return {"ands": _count(lambda C, *e: _build_step(C, S, *e), [t for _n, t in _sig_step(S)[0]])}


def size(**st) -> dict[str, int]:
    """A unit's ANDs: G steps and their flags ANDed."""
    return {"ands": st["G"] * step_size(**st)["ands"] + 2 * (st["G"] - 1)}
