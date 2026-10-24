"""The served rows' tie to the compute: a window's committed served rows (`served_rows`) held to the positions its steps'
Commit holds, so a prover can't register cheaper rows at serve time than the batch it ran. REDTEAM phase 10's six passes
(user-1 never admitted) leave step 6's 29 prefill tokens of the honest batch uncovered, and the tie refuses them.

    ServedRowsTie_v1{R, P, IW, L, T, IB, PB} (pub, srv, pos_<u>_<j> …) -> verdict[16]

**The step-input port**, the Commit's layout the tie reads (vllm's to adopt; `port_chunks`): each step's stream starts with its
input ids, then its positions, each T little-endian elements (IB and PB bytes) with zeros past the step's N tokens, T the
engine's token budget (``max_num_batched_tokens``). So the positions are the stream's bytes [T·IB, T·(IB + PB)), K chunks
of the step's rows-sha512 tree at fixed leaf indices (`geometry`), and the binding map's entries for the ids and the
positions (``c00.input/ids`` and ``c00.input/positions``; on the eager path ``model.embed_tokens/in`` and the rotary's
positions) name those ranges, where the Programs read them. IO-TIE's ingress tie, ``IoTieRequest``, reads the ids from the
same chunks through the same layout (`positions` and `ids`, the shared read).

**The rows** (each zero-padded to whole 16-bit words):
- ``pub``, public: each step's token count N (TW bits a step), the binding map's element count of the positions;
- ``srv``, the window's served rows, row w of the served-rows registration (`served_rows`: P steps of R slots of (valid,
  request, kind, start, n), `compile_slice.rows_bits`' packing), its b‖c checked natively against the registration's leaf;
- ``pos_<u>_<j>``, step u's j-th positions chunk, a 4096-byte leaf of the step's tree, its b‖c checked natively against
  that leaf (`chunk_ok`) at its public index, and the tree's root through the step root to the run root.

**The circuit.** Per step, the slots in batch order cover the step's tokens in turn: slot k's run is [lo_k, lo_k + n_k),
lo_k the sum of the earlier valid slots' n (an invalid slot's n is taken as 0). Bit 0 ``count``: every step's runs end at its
N. Bit 1 ``pos``: every token t < N is in a valid slot's run whose start puts it at the committed position,
position[t] = start_k + (t − lo_k), the position's high bits zero. ``ACCEPT`` = 0x3. Tokens from N on are padding and
unread. The request ids are not the compute's to show: ``IoTieRequest`` holds each prefill slot's ids, at its tokens, to
its certified request's prompt ids at those positions, and the egress tie each decode slot's input to the request's last
sample.

**Selection: every unit, always proved** (`Selection` All, as a transfer-gate unit is). The tie is a binding, not a check
on work: a step left untied makes that step's served rows the prover's word, and one such step can lower a window's charge
(the six passes are one). The charge it protects feeds the network verdict, an AND over every window with no ε, so a tie
sampled at ε would leave each untied step's charge free with probability 1 − ε.

**Cost**: about P·T·(29·R + 8·PB − W) ANDs (`size`; 193,156 at the slice's class), a comparison of each token with each
slot's run and the position's high bits. N is public, as the Commit's binding map and leaf counts publish it today.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from types import SimpleNamespace
from typing import Sequence

from verity.ml.circuits import boolean as BL
from verity.ml.circuits.boolean.forms import ZERO, Circuit, add, and_reduce, const_bits, or_reduce, sub
from verity.ml.circuits.defs import bind
from verity.ml.circuits.evaluation.bits import evaluate_bits
from verity.ml.circuits.types import Array
from verity.ml.commitments import hm96
from verity.ml.commitments.merkle import CommitmentDomain, merkle_depth
from verity.ml.commitments.rowleaf import SCHEMA_HM96_SHA512_ROW_V2, row_leaf
from verity.core.protocols.certifier import replay_chain as RC
from verity.core.protocols.certifier.head_check import BIT, _part

STATICS = ("R", "P", "IW", "L", "T", "IB", "PB")
CHUNK = 4096
WORD = 16
FLAGS = ("count", "pos")
ACCEPT = (1 << len(FLAGS)) - 1
INST = hm96.SHA512


def _bitlen(v: int) -> int:
    return max(int(v).bit_length(), 1)


def widths(S) -> SimpleNamespace:
    if min(S.R, S.P, S.IW, S.L, S.T, S.IB, S.PB) < 1:
        raise ValueError(f"statics {dict(S.__dict__)}: every size ≥ 1")
    LW, TW = _bitlen(S.L), _bitlen(S.T)
    return SimpleNamespace(LW=LW, TW=TW, W=max(LW, TW + _bitlen(S.R)) + 2)


def geometry(S) -> SimpleNamespace:
    """The positions' chunks of a step's tree, [lo, hi), and their first byte within chunk lo."""
    o0, o1 = S.T * S.IB, S.T * (S.IB + S.PB)
    lo, hi = o0 // CHUNK, -(-o1 // CHUNK)
    return SimpleNamespace(lo=lo, hi=hi, K=hi - lo, off=o0 - lo * CHUNK)


def _step_fields(S) -> list[tuple[str, int, int]]:
    Z = widths(S)
    return [("valid", S.R, 1), ("id", S.R, S.IW), ("kind", S.R, 1), ("start", S.R, Z.LW), ("n", S.R, Z.TW)]


def _words(bits: int) -> int:
    return WORD * -(-bits // WORD)


def rows(S) -> list[tuple[str, int]]:
    """The unit's rows and their widths in bits."""
    Z, G = widths(S), geometry(S)
    step = RC._bits_of(_step_fields(S))
    return ([("pub", _words(S.P * Z.TW)), ("srv", _words(S.P * step))]
            + [(f"pos_{u}_{j}", 8 * CHUNK) for u in range(S.P) for j in range(G.K)])


# --- the shared read ---------------------------------------------------------------------------------------------------------

def positions(S, chunks: Sequence) -> list[list]:
    """Step's T positions (PB·8 bits each, least significant first) from its K positions chunks' bits: the port's read, any
    bits (circuit wires or ints)."""
    G = geometry(S)
    bits = [b for c in chunks for b in c]
    at = 8 * G.off
    return [list(bits[at + 8 * S.PB * t:at + 8 * S.PB * (t + 1)]) for t in range(S.T)]


def ids(S, chunks: Sequence) -> list[list]:
    """Step's T input ids (IB·8 bits each) from the chunks [0, ⌈T·IB / 4096⌉) of its tree: ``IoTieRequest``'s read of the
    same port."""
    bits = [b for c in chunks for b in c]
    return [list(bits[8 * S.IB * t:8 * S.IB * (t + 1)]) for t in range(S.T)]


# --- the circuit -------------------------------------------------------------------------------------------------------------

def _build(C, S, pub, srv, *chunks):
    Z, G = widths(S), geometry(S)
    W, fs = Z.W, _step_fields(S)
    step_bits = RC._bits_of(fs)
    count_ok, pos_ok = [], []
    for u in range(S.P):
        f = RC._split(srv[u * step_bits:(u + 1) * step_bits], fs)
        N = RC._pad(pub[u * Z.TW:(u + 1) * Z.TW], W)
        lo, runs = [ZERO] * W, []
        for k in range(S.R):
            v = f["valid"][k][0]
            end = add(C, lo, RC._pad([C.AND(v, b) for b in f["n"][k]], W), width=W)
            runs.append((lo, end, sub(C, RC._pad(f["start"][k], W), lo, W)[0]))
            lo = end
        count_ok.append(RC._eq(C, lo, N))
        for t, p in enumerate(positions(S, chunks[u * G.K:(u + 1) * G.K])):
            tc = const_bits(t, W)
            hit = [and_reduce(C, [RC._lt(C, tc, a).inv(), RC._lt(C, tc, b), RC._eq(C, p[:W], add(C, off, tc, width=W))])
                   for a, b, off in runs]
            ok = C.AND(RC._zero(C, p[W:]), or_reduce(C, hit))
            pos_ok.append(_or(C, RC._lt(C, tc, N).inv(), ok))
    return [[and_reduce(C, count_ok), and_reduce(C, pos_ok)] + [ZERO] * (16 - len(FLAGS))]


def _or(C, a, b):
    return C.AND(a.inv(), b.inv()).inv()


def _sig(S):
    return tuple((r, Array(n, BIT)) for r, n in rows(S)), Array(16, BIT)


ServedRowsTie = _part(
    "ServedRowsTie", STATICS, _sig, _build,
    "a window's served rows tied to its steps' committed positions: per step, the slots' runs in batch order end at the "
    "step's token count, and each token below it is at its slot's start plus its offset in the run (verdict bits count, pos)")


def statics(S, IB: int = 8, PB: int = 8) -> dict:
    """The tie's statics for a compile slice's statics `S` (its R, P, IW, and ReplayStep_v1's L and T), the port's element
    bytes IB and PB."""
    return {"R": S.R, "P": S.P, "IW": S.IW, "L": S.L, "T": S.T, "IB": IB, "PB": PB}


def unit(**st):
    return bind(ServedRowsTie, **st)


# --- the specification -------------------------------------------------------------------------------------------------------

def _int(bits: Sequence[int]) -> int:
    return sum(int(b) << j for j, b in enumerate(bits))


def check(S, unit_rows: dict) -> dict[str, bool]:
    """The verdict's flags on a unit's rows (bits each), in integers."""
    Z, G, fs = widths(S), geometry(S), _step_fields(S)
    step_bits = RC._bits_of(fs)
    pos_rows = [unit_rows[f"pos_{u}_{j}"] for u in range(S.P) for j in range(G.K)]
    count, pos = True, True
    for u in range(S.P):
        f = {k: [_int(e) for e in v] for k, v in RC._split(unit_rows["srv"][u * step_bits:(u + 1) * step_bits], fs).items()}
        N = _int(unit_rows["pub"][u * Z.TW:(u + 1) * Z.TW])
        lo, runs = 0, []
        for k in range(S.R):
            n = f["n"][k] if f["valid"][k] else 0
            runs.append((lo, lo + n, f["start"][k]))
            lo += n
        count &= lo == N
        got = [_int(p) for p in positions(S, pos_rows[u * G.K:(u + 1) * G.K])]
        pos &= all(any(a <= t < b and got[t] == s + t - a for a, b, s in runs) for t in range(N))
    return {"count": count, "pos": pos}


def verdict(flags: dict) -> int:
    return sum(int(flags[f]) << k for k, f in enumerate(FLAGS))


def _le(values: Sequence[int], width: int, count: int) -> bytes:
    if len(values) > count:
        raise ValueError(f"{len(values)} tokens over the budget T = {count}")
    return b"".join(int(v).to_bytes(width, "little") for v in values) + bytes(width * (count - len(values)))


def port_chunks(S, step_ids: Sequence[int], step_positions: Sequence[int]) -> list[bytes]:
    """The step-input port, the head of a step's stream as the tie reads it: the ids, then the positions, padded to T each
    and to whole chunks."""
    if len(step_ids) != len(step_positions):
        raise ValueError("one position a token")
    head = _le(step_ids, S.IB, S.T) + _le(step_positions, S.PB, S.T)
    head += bytes(-len(head) % CHUNK)
    return [head[k:k + CHUNK] for k in range(0, len(head), CHUNK)]


def _bits(row: bytes) -> list[int]:
    return [(x >> j) & 1 for x in row for j in range(8)]


@dataclass(frozen=True)
class Case:
    rows: dict
    flags: dict
    verdict: int


def case(S, srv_bits: Sequence[int], steps: Sequence[tuple[Sequence[int], Sequence[int]] | None]) -> Case:
    """A unit's rows for a window: the served rows as registered (`served_rows.window_bits`) and each executed step's
    committed (ids, positions); an idle step (None, or past the list) has N = 0 and zero chunks."""
    Z, G = widths(S), geometry(S)
    steps = list(steps) + [None] * (S.P - len(steps))
    out = {"pub": [b for s in steps for b in _bits_int(len(s[1]) if s else 0, Z.TW)],
           "srv": list(srv_bits)}
    for u, s in enumerate(steps):
        chunks = port_chunks(S, *s) if s else [bytes(CHUNK)] * G.hi
        for j in range(G.K):
            out[f"pos_{u}_{j}"] = _bits(chunks[G.lo + j])
    for r, n in rows(S):
        out[r] = list(out[r]) + [0] * (n - len(out[r]))
    flags = check(S, out)
    return Case(out, flags, verdict(flags))


def _bits_int(v: int, w: int) -> list[int]:
    if not 0 <= v < 1 << w:
        raise ValueError(f"{v} doesn't fit {w} bits")
    return [(v >> j) & 1 for j in range(w)]


def evaluate(st: dict, cases: Sequence[Case]) -> list[int]:
    S = SimpleNamespace(**st)
    args = [[c.rows[r] for c in cases] for r, _n in rows(S)]
    out = evaluate_bits(unit(**st), *args).reshape(len(cases), -1).tolist()
    return [sum(int(b) << j for j, b in enumerate(o)) for o in out]


@lru_cache(maxsize=None)
def _counted(key: tuple) -> dict[str, int]:
    S = SimpleNamespace(**dict(zip(STATICS, key)))
    C = Circuit(cse=True)
    ins = [C.input(name, sum(BL.elements(t))) for name, t in _sig(S)[0]]
    C.output("verdict", _build(C, S, *ins)[0])
    return C.counts()


def size(**st) -> dict[str, int]:
    return {"ands": _counted(tuple(st[k] for k in STATICS))["ands"]}


# --- the verifier's native check ---------------------------------------------------------------------------------------------

def chunk_ok(domain: CommitmentDomain, index: int, bc: bytes, path: Sequence[bytes], tree_root: bytes) -> bool:
    """Whether a unit's chunk row, by its b‖c, is leaf ``index`` of the step tree over `domain` (the rows-sha512 step's) with
    root `tree_root`, `path` its siblings leaves first."""
    if not 0 <= index < domain.count or len(path) != merkle_depth(domain.count):
        return False
    digest, cursor = row_leaf(domain, index, INST.tree_leaf(INST.default_key, bytes(bc)), SCHEMA_HM96_SHA512_ROW_V2), index
    for d, sib in enumerate(path):
        digest = domain.node(d, cursor >> 1, *((digest, sib) if cursor % 2 == 0 else (sib, digest)))
        cursor >>= 1
    return digest == bytes(tree_root)
