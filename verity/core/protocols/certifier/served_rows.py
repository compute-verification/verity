"""The served rows: the step rows a served window ran, committed on the Commit side before the sample, which the compile
slice's ``rows`` bit holds the replay to through its ``srv`` row. Were that row the prover's own word, a prover who served
user-1's admission at the sixth ask could put six passes there, and the slice accepts it at 0x3F with a 5-bit code against
the honest 8. With the row registered here in ``srv``, ``rows`` refuses it; with any other
row, the verifier's b‖c check does.

**The rows.** The server keeps each step's rows as the model runner laid them out (`verity_vllm.commit.served_record`'s
`SamplerInputsSource.rows`: the runner's request id, its token rows [lo, hi) of the step's flattened batch, its first
position and its prefill length). `step_rows` gives one step's in `replay_step`'s form, (slot, request, kind, start, n) in
batch order: the request its certified arrival id, n = hi − lo, kind decode iff n = 1 and start ≥ the prefill
(ReplayStep_v1's rule). The slot is the row's batch position: under sync scheduling the scheduler's slot, which the
slice's ``rows`` compares slot for slot; under async (AS = 1) the runner's batch can order the slots' rows otherwise, and
``rows`` matches them by request through ``wit``'s ``map`` (`compile_slice.slot_map`). A value wider than its field is
refused, since `compile_slice.rows_bits` would truncate it. A window
is the slice's P steps in turn, each packed as `rows_bits` packs it (an idle step zero): `window_bits` is the slice's
``srv`` row (`compile_slice.srv_row`).

**The commitment** (`register`), made when the window closes and published with its run root before the advice check and
the sample: each block of G steps of window w (j = 0 … P / G − 1, `block_bits`) as row w·(P / G) + j of one
hm96-sha512/row/v2 registration in a frame-v3-sha512 tree (`verity.ml.commitments.advice`'s construction), its row the
block's bits as u16 words, as a proof commits a row, under `binding` of the run root. G is P unless given, so a window is
one row, the slice's ``srv``; at G < P block j is the slice's ``srv{j}`` (`compile_slice.srv_rows`). `Served.record` is
the public part: the root, the count, the row width and G. The rows and their salts stay the prover's.

**The verifier's side.** `leaf_ok` holds a unit's row by its b‖c to its registered leaf (`leaf`: window w's block j), as
the advice's ``adv`` row is held; `reads_port` is the same registration as the Lean verifier's registered reads take it
(`flock-verify verify --registered`), each unit's leaf its own.

That the served rows are the rows the served compute ran is the tie's to bind (`served_tie`: each row's valid, start and
n against its step's Program, and the ids IoTieRequest's); nothing here checks it.
"""
from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from verity.core.protocols.certifier import compile_slice as CS, replay_step as RS
from verity.core.protocols.sampled_proofs.one_stage import produced, registered as REG
from verity.ml.commitments import advice as ADV
from verity.ml.commitments.frame_v3 import BitRowLeaf
from verity.ml.commitments.rowleaf import bit_row, row_leaf

SCHEMA = "verity/certifier/served-rows/v1"
DOMAIN = b"verity/certifier/served-rows/v1"

Row = tuple[int, int, int, int, int]


def binding(run_root: bytes) -> bytes:
    """The registration's binding: the window's run root, so a registration can't be carried to another run."""
    return hashlib.sha512(DOMAIN + bytes(run_root)).digest()


def step_rows(S, served: Sequence[tuple], ids: Mapping[str, int]) -> tuple[Row, ...]:
    """One step's served rows (request, lo, hi, start, prefill_len), in batch order, as the slice's (slot, request, kind,
    start, n); `ids` maps a runner's request id to its certified arrival id."""
    Z = RS.widths(CS.step_statics(S))
    if len(served) > S.R:
        raise ValueError(f"a step of {len(served)} rows is over R = {S.R}")
    out = []
    for slot, (req, lo, hi, start, prefill) in enumerate(served):
        if req not in ids:
            raise ValueError(f"row {slot}: request {req!r} is no certified arrival")
        n = hi - lo
        row = (slot, ids[req], int(n == 1 and start >= prefill), start, n)
        for name, v, w in (("request", row[1], S.IW), ("start", start, Z.LW), ("n", n, Z.TW)):
            if not 0 <= v < 1 << w:
                raise ValueError(f"row {slot}: {name} {v} doesn't fit its {w} bits")
        out.append(row)
    return tuple(out)


def windows(S, by_step: Mapping[int, Sequence[tuple]], ids: Mapping[str, int], steps: int) -> list[list[tuple[Row, ...] | None]]:
    """The session's ``steps`` executed steps (`SamplerInputsSource.rows`' form, by step) in windows of P, a step with no
    rows idle (None)."""
    rows = [step_rows(S, by_step[u], ids) if by_step.get(u) else None for u in range(steps)]
    return [rows[k:k + S.P] for k in range(0, steps, S.P)]


def window_bits(S, window: Sequence[tuple[Row, ...] | None]) -> list[int]:
    return CS.srv_row(S, window)


def _blocks(S, G: int | None) -> int:
    G = S.P if G is None else G
    if not 1 <= G <= S.P or S.P % G:
        raise ValueError(f"a block is G steps, G dividing P = {S.P}, not G = {G!r}")
    return G


def block_bits(S, window: Sequence[tuple[Row, ...] | None], G: int, j: int) -> list[int]:
    """Block j of a window: its steps [j·G, (j + 1)·G), each `compile_slice.rows_bits` in turn (an idle step zero), zero-padded
    to whole words; at G = P the window's `window_bits`."""
    _blocks(S, G)
    if len(window) > S.P or not 0 <= j < S.P // G:
        raise ValueError(f"block {j} of G = {G} steps of a window of {len(window)} steps, P = {S.P}")
    steps = (list(window) + [None] * S.P)[j * G:(j + 1) * G]
    bits = [b for s in steps for b in CS.rows_bits(S, s)]
    return bits + [0] * (-len(bits) % CS.WORD)


def leaf(S, record: Mapping, w: int, j: int) -> int:
    """Window w's block j's leaf: w·(P / G) + j."""
    U = S.P // _blocks(S, record.get("G", S.P))
    if not 0 <= j < U:
        raise ValueError(f"block {j} of a window of {U}")
    return w * U + j


def _row(bits: Sequence[int]) -> tuple[bytes, int]:
    return bit_row((CS.words(bits), CS.WORD))[0], len(bits)


@dataclass(frozen=True)
class Served:
    """The prover's side: the registration, its rows and their salts, and the steps G a row holds."""

    registration: ADV.Registration
    G: int

    def record(self) -> dict:
        r = self.registration
        return {"schema": SCHEMA, "binding": r.binding.hex(), "root": r.root.hex(), "count": len(r.rows),
                "n_bits": r.n_bits, "G": self.G}

    def bc(self, i: int) -> bytes:
        """Leaf i's row commitment b‖c, as a proof that holds the row under its salt commits it."""
        r = self.registration
        return ADV.INST.commit_string(r.key, BitRowLeaf(r.n_bits).layout().digest(r.rows[i]), r.salts[i])


def register(S, session: Sequence[Sequence[tuple[Row, ...] | None]], run_root: bytes, *, G: int | None = None,
             salts: Sequence[bytes] | None = None) -> Served:
    """Commit the session's windows (`windows`' form) under the run root, a row each block of G steps (P unless given);
    fresh salts unless given."""
    G = _blocks(S, G)
    rows = [_row(block_bits(S, w, G, j)) for w in session for j in range(S.P // G)]
    return Served(ADV.register([r for r, _n in rows], rows[0][1], binding(run_root), salts), G)


def leaf_ok(record: Mapping, run_root: bytes, i: int, bc: bytes, path: Sequence[bytes]) -> bool:
    """Whether b‖c, with `path`, is leaf i's row of the registration `record` publishes under the run root."""
    b, n, n_bits = binding(run_root), record["count"], record["n_bits"]
    if record["binding"] != b.hex() or not 0 <= i < n:
        return False
    lf = row_leaf(ADV.domain(b, n), i, ADV.INST.tree_leaf(ADV.INST.default_key, bytes(bc)), BitRowLeaf(n_bits).hiding_schema)
    return ADV._scheme(b, n, n_bits).verify_path(ADV.PORT, bytes.fromhex(record["root"]), n, i, lf, tuple(path))


def reads_port(record: Mapping, run_root: bytes, reads: Sequence[int | None], reads_from: int = 0) -> dict:
    """The Lean verifier's own ``--registered`` port (`verity/registered-reads/v1`) over the registration `record`
    publishes under the run root: unit ``reads_from + i`` reads leaf ``reads[i]`` (None: none)."""
    b, n, n_bits = binding(run_root), record["count"], record["n_bits"]
    if record["binding"] != b.hex() or n_bits % CS.WORD:
        raise ValueError("a registration under another run root, or rows of part words")
    e = {"leaves": n, "words": n_bits // CS.WORD, "word_bits": CS.WORD, "domain": ADV.domain(b, n).domain_id.hex()}
    port = produced.port(SCHEMA, e, record["root"], [None if r is None else int(r) for r in reads],
                         BitRowLeaf(n_bits).hiding_schema)
    return port | ({REG.READS_FROM: int(reads_from)} if reads_from else {})
