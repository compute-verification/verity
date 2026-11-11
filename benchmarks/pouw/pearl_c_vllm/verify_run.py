"""The reference replay on a vLLM pass's transcript (``pouw_pearl_c_device``'s retained pass): the work sampling design
(`clear_work`), over the pass's whole workload, with the prover's openings read from the pass's files.

  python benchmarks/pouw/pearl_c_vllm/verify_run.py RUN_DIR/manifest.json --coins COINS.json [--draws T] [--jobs J] [--commit SHA] [--fast]

- **The execution phase** is re-started from the manifest's salt coin and every weight's root (`Epoch.start`): the coin must open
  against the execution phase's coin commitment as the challenger published it (``--coins``, not the pass's copy; a live session's
  when its scheme is `window.LIVE_SCHEME`, `epoch_coins`), as the salt coin, opened after the weights' root (a live
  session's after its salt epoch's root), and the salt must be the manifest's.  A pass recorded before the execution phase's coins (a v1 manifest)
  carries a beacon instead, and is read only with ``--recorded`` (`Epoch.unverified`, a salt that claims nothing).
- **The call indices** are each below 2^32 and none twice (`credit.passes_distinct`: ``call-range``, ``repeated-call``);
  `served_passes.py` holds a run's passes of one salt to distinct indices too.
- **The records** (each matmul's real rows, width, excluded and voluntary rows; each weight's excluded and voluntary rows)
  are the manifest's (`records_of`).  Their root must be the one the pass published before any sample.
- **The samples' source** is 32 bytes of the Lean verifier's coins (`pouw.verifier.coins`, `IO.getRandomBytes`,
  `uniform/io-getrandombytes`), the verifier's own, sampled here after the transcript is fixed: a
  `clear_work.RecordedDraw` of round 1, after the commitment's 0.
- **The prover only opens.** Each opening's value and path come from the pass's files: A's rows and tree levels per call,
  B's rows and tree levels per weight, the tile tree.  Each is checked against the published roots.  Under ``hm96-sha512``
  (-h3's) a node is 64 bytes and each value is followed by its salt, `audit.hm96_salt` under the manifest's
  ``hm96_salt_key`` (the prover's, retained for this replay), and the scheme reads the value less its salt.
- **The tiles** are `clear_work.audit`'s: its t work-weighted samples (`draw`) and one tile per excluded row
  (`exclusion_tiles`).  Each is held to `Verifier.check_tile` and `check_opened`, as `audit` does, but mapped over J
  processes, since each costs about 11 s of CPU per 1,024 of k (``--fast``: about a tenth of that on `fast_tile`'s batch
  forms, with the first tile and every ``--cross-check``-th after it, in the draw's order, replayed on the scalar reference
  too).  `tests/test_pearl_c_vllm_verify.py` holds this to `audit`'s own verdict.
- **The served y**, where a call commits it (`served_y`, the manifest's ``y``): an accepted tile's real rows of y open against
  its root, each holds a word for every column of the tile (the opening doesn't bind a row's length: one that ends before
  the tile's columns is refused, ``the opened row holds N words``), and each of the tile's words must be the unscale of
  the tile's checked U (`pearl_c_u.y`, the kernels' epilogue), so the y that leaves the linear is the one the credited
  rows computed.

`verify_passes` is a served run's verdict over all its passes (`served_passes.py verify`): one draw source ("The run's
draw" below), drawn once every pass's roots are read, keys t draws over all the passes' tickets together, and an accept credits
the run's credited MACs at Pearl's Assumption 1, less what ε of the tickets' weight could hold at δ, in core's answer form.
With a window record W (`tie`, Pearl+'s ``verity/registration/v2`` record of the retained rounds, written after the
retention, whose values the one-stage registration takes as outputs), W is held to the passes before the coin: round s's call c (the served run's
s-th pass with calls, its manifest's c-th call) is W's ``pearl-a/<s>/<c>`` and ``pearl-y/<s>/<c>``
(`window_class.PearlTables`), each with its call's root, the domain recomputed from the pass's salt and the call's index,
and its rows, words and word bits; each weight w the passes read is W's ``pearl-b/<w>``, at the root the epoch's weights'
root folds and the tiles' B rows open against, under `audit.weight_binding`'s domain, its n rows of K 32-bit words; and W
names no other Pearl+ value but a dropped round's calls'. Any mismatch rejects the run, since a tie to roots the verdict
didn't credit is vacuous. W states its version (`pearl_c.window.version_of`): this is version 2's tie, and a version-2 W
that registers no ``pearl-b/`` value at all is such a mismatch, named as one (`UNTIED_WEIGHTS`): its calls may all
match, but the run's B is no registered value, so it never reads as a tie. A W that states no version is version 1,
written before W registered the weights: it is held to the calls alone, as it was then, and the tie names its weights
untied (``"weights": WEIGHTS_UNTIED``, in the answer's tie and the verdict's line), so its accept never reads as a tie of
the weights; a version-1 W that registers a ``pearl-b/`` value is refused (`V1_WEIGHTS`).

The prover may drop passes from its retention before the coin (`served_passes.py retain`: a pass its CPU replay didn't
finish, or one left over a cap): the verdict is over the retained passes alone, each at its round, and the dropped ones
credit nothing (their MACs stay in what the run declared).  The run's commitment binds the retained passes and their
records, and the verifier records its coin against it (``record``, ``coin.json``): a verdict over other passes or other
records after the coin is refused (``after-coin``), so neither a retention nor a give-up can follow the coin.

An accepted run's verdict also says where its linears' MACs went uncredited, call by call (`uncredited_calls`, the verify
document's ``uncredited``): excluded, voluntary and side-channel cells, and the deduction's ε, over-cap and ρ terms. It
is a property of the committed records, which the answer never states, and changes nothing the verdict accepts.

**The run's draw** (`verify_passes`' ``draw``) is one source S, and the draws' key is
``derive(S, RUN_DRAW_DOMAIN, {"run": C})`` over the run's commitment C (`run_commitment`, 64 bytes).  S comes from one of:
- ``challenger`` (`bench_arm.LiveDraw`, `served_passes.py verify` under ``PEARLC_CHALLENGER_GATE``): a draw session of
  its own on the auditor host's challenger (`window.DRAW_FAMILIES`, its one coin ``pouw/draw`` 0), reserved withheld
  through the lane's gate once the lane has no period open (so it joins none, and the Pearl-C period's close lists the
  salt sessions alone) and, given the period the salt session joined, as the lane's first reserve after that period
  closed (`pouw_h1_window.after_close`, held before its coin is asked), opened on C, its coin released only after C and
  its commitment, salt and path only at its close. S is that coin's 64 bytes.
- ``node`` (`node_draw`, the default): 32 bytes of the Lean verifier's coins on this node (``uniform/io-getrandombytes``),
  a development draw no verdict of record takes.

**The coin record** (``record``, `COIN_SCHEMA`) is what every reader of the draw reads, the Lean `PouwCheck.PearlPlus`
among them; it is written before any replay, and once asked for a draw is never asked again (`ASKED`):

    {"schema": "verity-pouw/pearl-plus-coin/v2",
     "commitment": C,                      128 lowercase hex
     "source": S,                          lowercase hex: the draw coin's 64 bytes, or the node's 32
     "passes": [...],                      each retained pass's manifest as `read` names it, in order
     "rounds": [...], "dropped": [...],    where the verdict read a retention (`served_passes.py retain`)
     "draw": {"from": "challenger",
              "label": "pouw/draw", "index": 0,
              "reserved": {...},           the gate's reserve answer: session, seq, peer, by, expires_utc, period (null)
              "session": {...},            the gate's open answer without its secret: session, seq, root (= C), port, ...
              "commitment": {...},         `LiveCommitment.to_json`: the draw session's coin commitment, released at its close
              "opening": {...}}            `LiveOpening.to_json`: coin pouw/draw 0, after = C, coin = S, salt and path
           | {"from": "node", "claim": "uniform/io-getrandombytes"}}

A reader holds a challenger draw to: ``opening`` is coin ``pouw/draw`` 0 of ``commitment`` opened after C
(`challenger.live_reason`) and its coin is S; ``commitment`` is the one the session's challenger signed, of a draw
session's layout (`window.draw_epoch_reason`); the auditor's ledger, under the pinned key, opens that session alone on C,
withheld (`challenger.withheld_on`); and that session is its lane's first reserve after the lane's Pearl-C period closed
(`served_verdicts.pearlplus_live_draw`, over `pouw_h1_window.after_close`), so a prover who tries again under another
session holds a draw no reader takes.

Prints one line, ``ACCEPT …`` or ``REJECT …``, naming the pass's run and the manifest (`transcript`, as the panel's
``--transcript`` takes it), writes ``verify.json`` beside the manifest, and exits 0 only on acceptance, or with
``--expect-reject`` (a negative control's pass) only on rejection.  An ACCEPT is replay-verified (development shortcut; the
protocol's check is a ZK proof, PouwCheck).
"""
from __future__ import annotations

import argparse
import bisect
import hashlib
import json
import math
import mmap
import os
import sys
import multiprocessing
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any

import numpy as np
from verity.commitments import Commitment, Opening, VerificationLimits, verify_opening
from verity.protocols.pouw.verifier.audit import (
    HM96,
    HM96_TRANSCRIPT,
    HM96_WEIGHTS,
    ROW_SCHEMA,
    Epoch,
    activation_binding,
    hm96_salt,
    opened_value,
    rows_domain,
    weight_binding,
)
from verity.protocols.pouw.verifier import verifier as PV
from verity.protocols.pouw.verifier.coins import CoinCommitment
from verity.protocols.pouw.verifier.credit import passes_distinct
from verity.protocols.pouw.verifier.protocol import Matmul, Shape, Tile, tagged
from verity.protocols.pouw.prover.schemes import pearl_c_u as PCU
from verity.protocols.pouw.prover.schemes import pearl_c_work as PW
from verity.protocols.pouw.prover.schemes.pearl_c import K_MIN, TILE
from verity.protocols.pouw.verifier.window import LIVE_SCHEME
from verity.protocols.spot.device.challenger import LiveCommitment
from verity.protocols.spot.verifier.one_stage import registered as G, registration as REG
from verity.errors import ProtocolError
from verity.commitments.randomness import derive
from verity_vllm.protocols.pouw import pearl_c as PD
from verity_vllm.protocols.pouw.pearl_c import served_y as SY
from verity_vllm.protocols.pouw.pearl_c import window as PWR
from verity_config.settings import of

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "verity" / "protocols" / "pouw" / "clear"))
import fast_tile as FT
import clear_work as CW
from clear_audit import TileOpening, Transcript, Verifier

settings = of(__file__)
DRAWS = 3
#: a served run's verdict over all its passes (`verify_passes`)
PROTOCOL = "verity/pouw/pearl-plus/v1"
RUN_TAG = "verity/pouw/pearl-plus/run/v1"
RUN_DRAW_DOMAIN = "verity/pouw/pearl-plus-run-draw/v1"
ASSUMPTION = "Pearl's Assumption 1 (pearl_kw)"
DELTA = Fraction(1, 100)
#: `verify_passes`' rules on an opened tile over its inert-work bound (``over_cap``): the run is rejected, or the tile is
#: credited nothing; `served_passes.OVER_CAP` names the same two
REJECT, UNCREDITED = "reject", "uncredited"
OVER_CAP = (REJECT, UNCREDITED)
LIMITS = VerificationLimits()


def pow2(n: int) -> int:
    return 1 << max(n - 1, 0).bit_length()


def opening(tree: bytes, count: int, pos: int, value: bytes, nb: int = 32) -> Opening:
    """Position `pos`'s path in a tree laid out as the device writes it: level 0 padded to a power of two, then each level
    above it, `nb` bytes a node (64 under hm96-sha512)."""
    path, off, width, cur = [], 0, pow2(count), pos
    while width > 1:
        s = off + (cur ^ 1)
        path.append(bytes(tree[nb * s:nb * s + nb]))
        off, width, cur = off + width, width // 2, cur >> 1
    return Opening(pos, value, tuple(path))


class Files:
    """The pass's files, each read once (large ones memory-mapped) and held to its size and, where the manifest has one, its
    SHA-256."""

    def __init__(self, base: Path):
        self.base, self.cache = base, {}

    def __call__(self, spec: dict):
        key = spec["path"]
        if key not in self.cache:
            with open(self.base / key, "rb") as fh:
                data = mmap.mmap(fh.fileno(), 0, access=mmap.ACCESS_READ) if spec["bytes"] > 1 << 24 else fh.read()
            if len(data) != spec["bytes"]:
                raise ValueError(f"{key}: {len(data)} bytes; the manifest says {spec['bytes']}")
            if "sha256" in spec and hashlib.sha256(data).hexdigest() != spec["sha256"]:
                raise ValueError(f"{key}: its SHA-256 is not the manifest's")
            self.cache[key] = data
        return self.cache[key]


def f32_row(data, k: int, i: int) -> bytes:
    """Row i of a BF16 row file as the committed FP32 words (BF16 widened, exact)."""
    return (np.frombuffer(data[2 * k * i:2 * k * (i + 1)], "<u2").astype("<u4") << 16).tobytes()


class Opened:
    """The prover's side of the audit from the pass's files: `open`, as `clear_audit.Prover` has it."""

    def __init__(self, man: dict, files: Files, refs):
        self.man, self.files, self.refs = man, files, refs
        hash = PD.scheme_of(man["scheme"]).commitment_hash
        self.key = bytes.fromhex(man["hm96_salt_key"]) if hash == HM96 else None
        self.nb = 64 if hash == HM96 else 32

    def _salted(self, value: bytes, nonce: int, i: int) -> bytes:
        return value if self.key is None else value + hm96_salt(self.key, nonce, i)

    def open(self, epoch: Epoch, tile_id: int) -> TileOpening:
        ref, nb, sv = self.refs[tile_id], self.nb, self._salted
        u, rec = epoch.workload.matmuls[ref.matmul], self.man["matmuls"][ref.matmul]
        wt = self.man["weights"][str(u.weight)]
        a_rows, a_tree = self.files(rec["rows"]), self.files(rec["tree"])
        b_rows, b_tree = self.files(wt["rows"]), self.files(wt["tree"])
        c = self.man["commitment"]
        leaves = self.files(self.man["leaves"])
        return TileOpening(ref, tuple(opening(a_tree, u.shape.m, i, sv(f32_row(a_rows, u.shape.k, i), u.index, i), nb)
                                      for i in ref.tile.rows),
                           tuple(opening(b_tree, u.shape.n, j, sv(f32_row(b_rows, u.shape.k, j), HM96_WEIGHTS + u.weight, j), nb)
                                 for j in ref.tile.cols),
                           opening(self.files(self.man["tree_t"]), c["count"], tile_id,
                                   sv(bytes(leaves[nb * tile_id:nb * tile_id + nb]), HM96_TRANSCRIPT, tile_id), nb))

    def open_y(self, epoch: Epoch, tile_id: int) -> tuple[Opening, ...]:
        """The tile's real rows of its call's committed y (`served_y`; filler rows aren't served)."""
        ref = self.refs[tile_id]
        u, rec = epoch.workload.matmuls[ref.matmul], self.man["matmuls"][ref.matmul]
        rows, tree, real = self.files(rec["y"]["rows"]), self.files(rec["y"]["tree"]), rec["real_rows"]
        return tuple(opening(tree, real, i, self._salted(f32_row(rows, u.shape.n, i), SY.HM96_Y + u.index, i), self.nb)
                     for i in ref.tile.rows if i < real)


class Refused(ValueError):
    """A pass refused before anything is opened, with `credit.passes_distinct`'s (code, why)."""

    def __init__(self, why: tuple[str, str]):
        super().__init__(": ".join(why))
        self.why = why


def read(path: Path) -> tuple[dict, str, str]:
    """The manifest, its SHA-256, and the name a verdict gives it (``RUN:PATH sha256:DIGEST``)."""
    raw = path.read_bytes()
    man, digest = json.loads(raw), hashlib.sha256(raw).hexdigest()
    return man, digest, f"{man.get('run')}:{path} sha256:{digest}"


def records_of(man: dict, matmuls) -> list[PW.UnitRecord]:
    """The records the manifest declares, a call's and its weight's: real rows, |K|, excluded and voluntary A rows (a call's
    ``excluded_a`` and ``voluntary_a``), and excluded and voluntary B rows (its weight's ``excluded_b`` and ``voluntary_b``)."""
    def weight(mm):
        return man["weights"][str(mm.weight)]
    return [PW.UnitRecord(mm, u["real_rows"], u["k_kept"], tuple(u["excluded_a"]), tuple(weight(mm)["excluded_b"]),
                          tuple(u.get("voluntary_a", ())), tuple(weight(mm).get("voluntary_b", ())))
            for mm, u in zip(matmuls, man["matmuls"])]


@dataclass
class Pass:
    """One pass as the verifier holds it once its manifest is read: its epoch, records and transcript, checked against
    what the pass published, and the prover's openings from its files."""

    path: Path
    man: dict
    salt: str
    scheme: object
    epoch: Epoch
    records: list
    transcript: Transcript
    verifier: Verifier
    prover: Opened

    def tile_ids(self, picks: list[tuple[int, Tile]]) -> list[tuple[int, int]]:
        ids = {(ref.matmul, ref.tile.rows[0], ref.tile.cols[0]): ref.id for ref in self.verifier.refs}
        return [(pos, ids[(pos, t.rows[0], t.cols[0])]) for pos, t in picks]


def epoch_coins(path: Path) -> CoinCommitment | LiveCommitment:
    """The epoch's coin commitment in the coins file at `path` (a run's ``coins.json``): a live session's when its scheme is
    `window.LIVE_SCHEME`, else an in-process challenger's."""
    doc = json.loads(Path(path).read_text())
    live = isinstance(doc, dict) and doc.get("scheme") == LIVE_SCHEME
    return (LiveCommitment if live else CoinCommitment).from_json(doc)


def load(path: Path, coins: CoinCommitment | LiveCommitment | None = None, recorded: bool = False,
         man: dict | None = None) -> Pass:
    """The pass at manifest `path`, its epoch re-started under `coins` (`pouw_pearl_c_device.manifest_epoch`); raises
    `Refused` or ValueError on a pass that can't be checked."""
    man = read(path)[0] if man is None else man
    PD.manifest_schema(man, recorded)
    why = passes_distinct([man])
    if why is not None:
        raise Refused(why)
    scheme = PD.scheme_of(man["scheme"], **{k: tuple(v) if isinstance(v, list) else v for k, v in man.get("instance", {}).items()})
    matmuls = tuple(Matmul(u["index"], Shape(u["m"], u["k"], u["n"]), u["weight"]) for u in man["matmuls"])
    epoch, salt = PD.manifest_epoch(man, coins, recorded)
    records = records_of(man, matmuls)
    splits = {w: PW.Split(man["weights"][str(w)]["k"]) for w in epoch.workload.weights()}
    reason = CW.check_records(epoch, splits, records)
    if reason is None and PW.records_root(epoch.salt, epoch.weights_root, splits, records).hex() != man["records_root"]:
        reason = "the records are not the ones whose root the pass published"
    if reason is not None:
        raise ValueError(reason)
    c = man["commitment"]
    transcript = Transcript(tuple(bytes.fromhex(u["root_a"]) for u in man["matmuls"]), Commitment(bytes.fromhex(c["root"]), c["count"]),
                            ())
    verifier = Verifier(CW._TileRuleView(scheme), epoch)
    if len(verifier.refs) != c["count"]:
        raise ValueError(f"the commitment counts {c['count']} tiles; the workload has {len(verifier.refs)}")
    return Pass(path, man, salt, scheme, epoch, records, transcript, verifier, Opened(man, Files(path.parent), verifier.refs))


_JOB: dict = {}


def check(job: tuple[int, tuple[int, int, int]]) -> tuple[int, int, bool, str | None, bool, bool | None]:
    """The verdict of the ``i``-th tile in the draw's order, ``job`` = (i, (pass, record position, tile id)): (pass, tile
    id, ok, reason, over, same) (`_check`'s, then ``same``), where on `--fast`, for the first tile and every
    `--cross-check`-th after it, ``same`` says whether the scalar reference's verdict is the same one (None where it was
    not replayed twice).  By place, not tile id: an
    excluded row's tile starts its row of a matmul, at an id that is a multiple of the matmul's column tiles (256 for the
    1B's gate_up), so a rule on ids takes whole layouts' tiles or none."""
    i, pick = job
    got = _check(pick)
    every, undo = _JOB.get("cross_check", 0), _JOB.get("undo")
    if undo is None or not every or i % every:
        return *got, None
    undo()
    try:
        same = _check(pick) == got
    finally:
        _JOB["undo"] = FT.install(_JOB["scheme"])
    return *got, same


def _check(pick: tuple[int, int, int]) -> tuple[int, int, bool, str | None, bool]:
    """One tile, (pass, record position, tile id), as `clear_work.audit` checks it: the openings and digest
    (`Verifier.check_tile`), then the tile rule, its rows and then its inert-work bound, then its y.  (pass, tile id, ok,
    reason, over): ``over`` where the tile fails its bound (`clear_work.cap_failure`, the reason) and passes every other
    check, y included, which `verify_passes`' ``over_cap`` decides."""
    i, pos, tid = pick
    p = _JOB["passes"][i]
    scheme, epoch, transcript = p.scheme, p.epoch, p.transcript
    ref = p.verifier.refs[tid]
    try:
        opened = p.prover.open(epoch, tid)
    except (OSError, ValueError) as ex:
        return i, tid, False, f"the prover's files don't open it: {type(ex).__name__}: {ex}", False
    v = p.verifier.check_tile(transcript, opened)
    if not v.ok:
        return i, tid, False, v.reason, False
    h = scheme.commitment_hash
    a, b = [opened_value(h, o.value) for o in opened.a], [opened_value(h, o.value) for o in opened.b]
    record = p.records[pos]
    reason = CW.rows_failure(scheme, record, ref.tile, a, b, v.ops)
    over = None if reason is not None else CW.cap_failure(scheme, record, ref.tile, v.ops, getattr(scheme, "r1_tolerance", 0))
    if reason is None and "y" in p.man["matmuls"][pos]:
        try:
            reason = check_y(p, pos, ref.tile, v, p.prover.open_y(epoch, tid))
        except (OSError, ValueError) as ex:
            reason = f"the prover's files don't open its y: {type(ex).__name__}: {ex}"
    if reason is None and over is not None:
        return i, tid, False, over, True
    return i, tid, reason is None, reason, False


def check_y(p: Pass, pos: int, tile: Tile, v, opened: tuple[Opening, ...]) -> str | None:
    """The served y on an accepted tile: its real rows open against the call's y root, each holds a word for every column
    of the tile, and each of those words is the unscale of the tile's checked U at its row's and column's α (`pearl_c_u.y`,
    the kernels' epilogue).  The opening binds a row's value and not how many words it has (an hm96 leaf takes a preimage
    of any length), so a row is held to the tile's columns, ``width`` their end, before any of its words is read: one
    that ends sooner is refused, as the Lean's `Rule.yReason` refuses it.  A row's words are its whole ones (the Lean's
    `words32`)."""
    if p.scheme.commitment_hash != HM96 or p.scheme.forming != "v1":
        return f"y is checked under {HM96} and forming v1, not {p.scheme.name}"
    u, rec = p.epoch.workload.matmuls[pos], p.man["matmuls"][pos]
    real = rec["real_rows"]
    d, c = SY.y_domain(p.epoch.salt, u.index, real), Commitment(bytes.fromhex(rec["y"]["root"]), real)
    try:
        if not all(verify_opening(d, c, o, ROW_SCHEMA, LIMITS) for o in opened):
            return "y opening"
    except ProtocolError:
        return "y opening"
    width = max((j + 1 for j in tile.cols), default=0)
    for x, (o, urow) in enumerate(zip(opened, v.checked.u[:len(opened)], strict=True)):
        value = opened_value(HM96, o.value)
        words = np.frombuffer(value, "<u4", len(value) // 4)
        if len(words) < width:
            return f"y[{tile.rows[x]}]: the opened row holds {len(words)} words, the tile's columns end at {width}"
        for z, j in enumerate(tile.cols):
            try:
                want = PCU.y(urow[z], v.ops.a.alpha[x], v.ops.b.alpha[z]) << 16
            except PCU.NonFinite:
                return f"y[{tile.rows[x]}, {j}]: the unscale of U leaves the finite range"
            if int(words[j]) != want:
                return f"y[{tile.rows[x]}, {j}] is not the unscale of the tile's U"
    return None


def replay(passes: list[Pass], todo: list[tuple[int, int, int]], jobs: int, fast: bool = False, cross_check: int = 0
           ) -> tuple[list[tuple[int, int, bool, str | None, bool, bool | None]], bool]:
    """`check` on each (pass, record position, tile id) of `todo` over J processes, and whether the batch forms ran:
    `fast` replays on `fast_tile`'s batch forms where they take the scheme (Pearl-C v1 on a promoted device; otherwise the
    scalar reference), and the first tile of `todo` and every `cross_check`-th after it (`check`) on the scalar reference
    too."""
    scheme = passes[0].scheme
    if any(p.scheme.name != scheme.name for p in passes):
        raise ValueError("the passes are not all of one scheme")
    fast = fast and scheme.forming == "v1" and scheme.device.g is not None
    _JOB.clear()
    _JOB.update(passes=passes, scheme=scheme, cross_check=cross_check, undo=FT.install(scheme) if fast else None)
    try:
        if jobs > 1 and len(todo) > 1:
            # fork: the workers unpickle their function from this module, which callers load from its path, not by an importable name
            with multiprocessing.get_context("fork").Pool(min(jobs, len(todo))) as pool:
                return pool.map(check, list(enumerate(todo)), chunksize=1), fast
        return [check(job) for job in enumerate(todo)], fast
    finally:
        if _JOB.get("undo") is not None:
            _JOB.pop("undo")()


def verify(path: Path, draws: int, jobs: int, commit: str, source: bytes | None = None, fast: bool = False,
           cross_check: int = 0, coins: CoinCommitment | LiveCommitment | None = None, recorded: bool = False) -> dict:
    """The pass's verdict, its execution phase re-started under `coins`, the execution phase's published coin commitment
    (`pouw_pearl_c_device.manifest_epoch`; `recorded` reads a pass recorded before the execution phase's coins).  `fast` replays its tiles on `fast_tile`'s batch forms where they take the scheme (Pearl-C v1 on a
    promoted device; otherwise the scalar reference, ``replay``), and the first of its tiles and every `cross_check`-th after
    it in the draw's order (`check`) on the scalar reference too: any verdict that differs is ``cross_check_failed``, and the
    pass is not accepted."""
    man, digest, named = read(path)
    out = {"manifest": str(path), "transcript": named, "transcript_sha256": digest, "verifier_commit": commit, "draws": draws,
           "replay": "batch" if fast else "scalar"}
    try:
        p = load(path, coins, recorded, man)
        scheme, records, c, out["salt"] = p.scheme, p.records, man["commitment"], p.salt
        source = PV.coins(32) if source is None else source
        key = CW.replay_key(CW.RecordedDraw(1, source), p.transcript.commitment.root, bytes.fromhex(man["records_root"]), p.epoch.salt)
        drawn, excluded = CW.draw(scheme, records, key, draws), CW.exclusion_tiles(scheme, records)
        todo = [(0, pos, tid) for pos, tid in p.tile_ids(drawn + excluded)]
        verdicts, fast = replay([p], todo, jobs, fast, cross_check)
        out["replay"] = "batch" if fast else "scalar"
        differ = [t for _, t, _, _, _, same in verdicts if same is False]
        ok = all(v[2] for v in verdicts) and not differ
        w_ref, credit = CW.work(scheme, records)
        bad = "; ".join(f"tile {t}: {r}" for _, t, good, r, *_ in verdicts if not good)
        if differ:
            bad = "; ".join(filter(None, [bad, f"the batch replay's verdict is not the scalar one's on tiles {differ}"]))
            out["cross_check_failed"] = differ
        if fast:
            out["cross_checked"] = [t for _, t, _, _, _, same in verdicts if same is not None]
        at = CW.draw_wref(scheme)
        out.update(accepted=ok, scheme=scheme.name, source=source.hex(), tiles=[t for _, t, *_ in verdicts],
                   draw_wref=at.statement if at else "wref",
                   drawn=len(drawn), exclusion_tiles=len(excluded),
                   verdicts=[{"tile": t, "ok": g, "reason": r} for _, t, g, r, *_ in verdicts], w_ref=str(w_ref), credit=str(credit))
        out["line"] = (f"{'ACCEPT' if ok else 'REJECT'} {scheme.name} {man['entry']}: clear_work's tiles ({len(drawn)} drawn, "
                       f"{len(excluded)} for excluded rows) of {c['count']} over {len(records)} matmuls"
                       f"{'' if ok else ' (' + bad + ')'}; transcript root {c['root'][:16]}, records {man['records_root'][:16]}, "
                       f"source {source.hex()[:16]}; transcript {named}; verifier {commit}"
                       f"{'; batch replay, ' + str(len(out['cross_checked'])) + ' tiles cross-checked' if fast else ''}")
    except Exception as ex:  # noqa: BLE001 - a transcript that can't be checked is rejected, with the reason
        if isinstance(ex, Refused):
            out["refused"] = list(ex.why)
        out.update(accepted=False, error=f"{type(ex).__name__}: {ex}",
                   line=f"REJECT {man.get('scheme')} {man.get('entry')}: {type(ex).__name__}: {ex}; transcript {named}; verifier {commit}")
    return out


def y_roots(p: Pass) -> list[str | None]:
    """Each call's committed y root (`served_y`), None where the pass commits none."""
    return [u["y"]["root"] if "y" in u else None for u in p.man["matmuls"]]


def run_commitment(passes: list[Pass]) -> bytes:
    """What the run committed before its draw: each pass's transcript root, records root, salt and its calls' y roots (empty
    where none), in order."""
    return tagged(RUN_TAG, *(x for p in passes for x in (p.transcript.commitment.root, bytes.fromhex(p.man["records_root"]),
                                                         p.epoch.salt, *(bytes.fromhex(r or "") for r in y_roots(p)))))


TIE_A, TIE_Y, TIE_B = "pearl-a/{s}/{c}", "pearl-y/{s}/{c}", "pearl-b/{w}"
#: W's Pearl+ values: a call's (a round's, `TIE_CALLS`) and a weight's
TIE_PREFIXES = ("pearl-a/", "pearl-y/", "pearl-b/")
TIE_CALLS = TIE_PREFIXES[:2]
#: why `tie` refuses a version-2 W that ties the calls alone: it reads as untied weights, never as a tie
UNTIED_WEIGHTS = "W registers no weight's B rows (no pearl-b/ value: a window record written before W registered them)"
#: the tie's ``weights`` for a version-1 W (`pearl_c.window.version_of`), which ties the calls alone: named in the
#: verdict, never read as a tie of the weights
WEIGHTS_UNTIED = "weights untied (W v1)"
#: why `tie` refuses a version-1 W that registers a weight's B rows
V1_WEIGHTS = "W states no version (version 1), whose values are the calls' alone, and registers"
#: the coin record's schema (module docstring, "The coin record")
COIN_SCHEMA = "verity-pouw/pearl-plus-coin/v2"
#: the coin record's ``draw.from``: a draw session's coin on the auditor host's challenger, or this node's coins
DRAW_CHALLENGER, DRAW_NODE = "challenger", "node"
#: the claim instance the node's draw source rests on (`pouw.verifier.coins`)
NODE_CLAIM = "uniform/io-getrandombytes"
#: beside the coin record, the mark written just before a draw is asked for: a draw asked and never recorded is not
#: asked again
ASKED = ".asked"
#: why `verify_passes` refuses passes other than those its recorded coin was drawn on
AFTER_COIN = ("after-coin", "the retained passes or their records are not those the verifier's coin was drawn on: a "
              "retention change or a give-up after the coin is refused, and a retry never redraws")
#: why `verify_passes` refuses a run whose draw was asked for and never recorded
ASKED_ONCE = ("asked-once", "the verifier's draw was asked for and never recorded (a draw that failed or was cut short), "
              "and a retry never redraws")


def node_draw(root: bytes, asking: Any) -> tuple[bytes, dict]:
    """The development draw (`DRAW_NODE`): 32 bytes of the Lean verifier's coins on this node, after `asking` marks it."""
    asking({})
    return PV.coins(32), {"from": DRAW_NODE, "claim": NODE_CLAIM}


def tie_entries(passes: list[Pass], rounds: list[int] | None = None, weights: bool = True) -> dict[str, dict]:
    """What a window record must register for each call of the run (`tie`): A, its m16 committed rows of K FP32 words under
    `activation_binding`'s domain, and y, its real rows of N FP32 words under `served_y.y_domain`'s, each at its root; then,
    with ``weights`` (a version-2 W's), for each weight the passes read (pass by pass, by ascending id), B, its n rows of K
    FP32 words under `weight_binding`'s domain, at the root the pass's epoch commits it under. Pass i is round
    ``rounds[i]`` (default i): its place among the served run's passes with calls. ValueError for a weight two passes
    commit under another root or shape."""
    want, ss = {}, list(range(len(passes)) if rounds is None else rounds)
    for s, p in zip(ss, passes, strict=True):
        if p.scheme.commitment_hash != HM96:
            raise ValueError(f"a tie is to {HM96} rows, not pass {s}'s {p.scheme.commitment_hash}")
        for c, (r, rec) in enumerate(zip(p.records, p.man["matmuls"], strict=True)):
            u = r.matmul
            if "y" not in rec:
                raise ValueError(f"pass {s}'s call {c} (index {u.index}) commits no y to tie")
            a = rows_domain(activation_binding(p.epoch.salt, u.index), u.index, u.shape.m, HM96)
            want[TIE_A.format(s=s, c=c)] = {"root": rec["root_a"], "leaves": u.shape.m, "words": u.shape.k, "word_bits": 32,
                                            "domain": a.domain_id.hex()}
            want[TIE_Y.format(s=s, c=c)] = {"root": rec["y"]["root"], "leaves": rec["real_rows"], "words": u.shape.n,
                                            "word_bits": 32,
                                            "domain": SY.y_domain(p.epoch.salt, u.index, rec["real_rows"]).domain_id.hex()}
    bs: dict[str, dict] = {}
    for s, p in zip(ss, passes) if weights else ():
        for w, (n, k) in sorted(p.epoch.workload.weights().items()):
            x = {"root": p.epoch.weight_roots[w].hex(), "leaves": n, "words": k, "word_bits": 32,
                 "domain": rows_domain(weight_binding(w), -2, n, HM96).domain_id.hex()}
            if bs.setdefault(TIE_B.format(w=w), x) != x:
                raise ValueError(f"pass {s} commits weight {w}'s B rows under another root or shape than an earlier pass, "
                                 "and W registers one")
    return {**want, **bs}


def _owner(v: str) -> str:
    return "weight" if v.startswith(TIE_PREFIXES[2]) else "call"


def tie(w: dict, passes: list[Pass], rounds: list[int] | None = None, dropped: tuple[int, ...] = ()) -> dict:
    """``{"record": digest(W), "tables": n}`` for a window record W at version 2 (`pearl_c.window.version_of`) that
    registers exactly the run's calls' A rows and y and its weights' B rows (`tie_entries`, pass i as round ``rounds[i]``)
    among its ``pearl-a/``, ``pearl-y/`` and ``pearl-b/`` values, beside any of the ``dropped`` rounds' calls' (passes
    left out of the retention before the coin, which credit nothing). For a W at version 1, which registers no ``pearl-b/``
    value, the tie is of the calls alone, as it was at that version, and the answer names the weights untied:
    ``"weights": WEIGHTS_UNTIED``. ValueError naming the first mismatch otherwise."""
    if not isinstance(w, dict) or w.get("format") != REG.FORMAT_V2 or not isinstance(w.get("registered"), list):
        raise ValueError(f"W is no {REG.FORMAT_V2} record with registered values")
    tied = PWR.version_of(w) == PWR.VERSION
    got: dict[str, dict] = {}
    for e in w["registered"]:
        v = e.get("value") if isinstance(e, dict) else None
        if isinstance(v, str) and v.startswith(TIE_PREFIXES):
            if v in got:
                raise ValueError(f"W registers {v} twice")
            got[v] = e
    if not tied and (b := next((v for v in got if v.startswith(TIE_PREFIXES[2])), None)):
        raise ValueError(f"{V1_WEIGHTS} {b}: a W that registers the weights states version {PWR.VERSION}")
    want, skip = tie_entries(passes, rounds, tied), {str(s) for s in dropped}
    if extra := sorted(v for v in set(got) - set(want) if not (v.startswith(TIE_CALLS) and v.split("/")[1] in skip)):
        raise ValueError(f"W registers {extra[0]}, no {_owner(extra[0])} of the run's")
    if (b := next((v for v in want if v.startswith(TIE_PREFIXES[2])), None)) and not any(
            v.startswith(TIE_PREFIXES[2]) for v in got):
        raise ValueError(f"{UNTIED_WEIGHTS}, so the run's rootB is no W's ({b} first)")
    for v, x in want.items():
        if (e := got.get(v)) is None:
            raise ValueError(f"W registers no {v}")
        have = {**e, "word_bits": e.get(G.WIDTH_KEY, G.WORD_BITS)}
        if bad := [k for k in x if type(have.get(k)) is not type(x[k]) or have[k] != x[k]]:
            raise ValueError(f"W's {v}: its {', '.join(bad)} not the {_owner(v)}'s")
    return {"record": REG.digest(w), "tables": len(want), **({} if tied else {"weights": WEIGHTS_UNTIED})}


def epsilon(t: int, delta: Fraction, over: int = 0) -> Fraction:
    """One more than the least k with Σ_{i ≤ over} C(t, i) (k/10⁹)^i (1 − k/10⁹)^(t − i) ≤ δ, over 10⁹, exactly (at
    ``over`` = 0, (1 − ε)^t ≤ δ): of t work-weighted draws, ``over`` landed on tiles over their inert-work bound and the
    rest passed, and the tiles wrong or over the bound hold more than ε of the tickets' weight only with δ.  The Lean's
    `epsilon`, step for step."""
    g, num, den = 10 ** 9, delta.numerator, delta.denominator

    def ok(k: int) -> bool:
        mass = sum(math.comb(t, i) * k ** i * (g - k) ** (t - i) for i in range(min(over, t) + 1))
        return mass * den <= num * g ** t

    lo, hi = 0, g
    for _ in range(31):
        if lo < hi:
            mid = (lo + hi) // 2
            lo, hi = (lo, mid) if ok(mid) else (mid + 1, hi)
    return Fraction(lo + 1, g)


def credited_cells(scheme: Any, r: PW.UnitRecord) -> list[int]:
    """Each of a record's tiles' credited cells, its credited rows × columns (`UnitRecord.credited`)."""
    return [len(rows) * len(cols) for rows, cols in (r.credited(x) for x in scheme.tiles(r.matmul.shape))]


def pass_macs(p: Pass) -> int:
    """One pass's credited cells' MACs, each cell at its record's kept K (`credit_at_assumption`'s ``macs`` over it alone)."""
    return sum(sum(credited_cells(p.scheme, r)) * r.k_kept for r in p.records)


def _taken(passes: list[Pass], t: int, delta: Fraction, over: int) -> tuple[dict[tuple[int, int], tuple[list[int], Fraction]], Fraction]:
    """`credit_at_assumption`'s deduction record by record: each record with a credited cell, by (pass, position), its
    tiles' credited cells and the MACs taken off it, and ε."""
    units = []
    for i, p in enumerate(passes):
        for pos, r in enumerate(p.records):
            cells = credited_cells(p.scheme, r)
            if sum(cells):
                ticket = CW.ticket_cell(p.scheme, r)
                units.append(((i, pos), cells, r.k_kept / ticket, sum(cells) * ticket))
    eps = epsilon(t, delta, over)
    budget, taken = eps * sum(u[3] for u in units), {}
    for at, cells, ratio, w in sorted(units, key=lambda u: (u[2], u[3]), reverse=True):
        take = min(budget, w)
        taken[at], budget = (cells, take * ratio), budget - take
    return taken, eps


def credit_at_assumption(passes: list[Pass], t: int, delta: Fraction, over: int = 0) -> dict:
    """The run's credit at Pearl's Assumption 1 (`pearl_kw`): a tile that passes the rule cost its credited cells' MACs,
    (1 − ρ) of them at the scheme's cap.  Of t draws by ticket weight, ``over`` (X_O) landed on tiles over their
    inert-work bound, credited nothing, and the rest passed: the tiles wrong or over the bound hold at most ε of the
    tickets' weight except with δ (`epsilon`); the most MACs that weight can hold, filled from the records whose cells hold
    the most MACs a ticket, are taken off (`_taken`)."""
    taken, eps = _taken(passes, t, delta, over)
    macs = sum(sum(cells) * passes[i].records[pos].k_kept for (i, pos), (cells, _) in taken.items())
    tiles = sum(1 for cells, _ in taken.values() for c in cells if c)
    wrong, rho = sum((w for _, w in taken.values()), Fraction(0)), passes[0].scheme.cap
    credited = (1 - rho) * max(macs - wrong, Fraction(0))
    return {"credited": math.floor(credited), "credited_tiles": tiles, "macs": macs, "rho": str(rho), "epsilon": str(eps),
            "over_cap_draws": over}


#: the causes `uncredited_calls` splits a call's real MACs the credit leaves out by, beside the deduction's three terms
UNCREDITED_ROWS = ("excluded_a", "excluded_b", "voluntary", "kept_width")
UNCREDITED_SCHEMA = "verity-pouw/pearl-plus-uncredited/v1"


def uncredited_cells(r: PW.UnitRecord) -> tuple[int, dict[str, int]]:
    """(credited cells, the real cells credited nothing by cause) of a record, by `UnitRecord.credited`'s band rule over
    its real rows × n columns: a cell in a column block holding an excluded B row is ``excluded_b``; else one in a block
    holding a voluntary B row, or in a row band holding a voluntary A row, is ``voluntary``; else one in an excluded A
    row is ``excluded_a``."""
    n, band, real = r.matmul.shape.n, PW.BAND, r.real_rows
    eb = {j // band for j in r.excluded_b}
    vb = {j // band for j in r.voluntary_b} - eb
    bands = {i // band for i in r.voluntary_a}
    width = {b: min(band, n - b * band) for b in eb | vb}
    cols_eb, cols_vb = sum(width[b] for b in eb), sum(width[b] for b in vb)
    live = n - cols_eb - cols_vb
    rows_vb = sum(min(band, real - b * band) for b in bands if b * band < real)
    ea = sum(1 for i in r.excluded_a if i // band not in bands)
    return (real - rows_vb - ea) * live, {"excluded_a": ea * live, "excluded_b": real * cols_eb,
                                          "voluntary": real * cols_vb + rows_vb * live}


def uncredited_calls(passes: list[Pass], t: int, delta: Fraction, over: int = 0) -> dict:
    """Where an accepted run's linears' MACs go uncredited, call by call (the verify document's ``uncredited``): each call's
    real MACs (its real rows × n × its weight's k), the MACs its credited cells hold at |K| (``checked_macs``, summing to
    the credit's ``macs``), its real MACs the credit leaves out by cause (`uncredited_cells` at |K|, and ``kept_width``,
    its real rows × n × the k − |K| side channels), and the credit's three terms on it: ``epsilon_macs``, what
    `credit_at_assumption`'s deduction takes off it at ε with no draw over the inert-work bound; ``over_cap_macs``, what
    the ``over`` draws on tiles over it add; ``rho_macs``, ρ of the rest; and ``credited_macs``, (1 − ρ) of the rest.  The
    terms are exact fractions, so a call's real MACs are its checked MACs and its causes, and its checked MACs its three
    terms and its credit; ``totals`` sums them, ``credited`` floored as the credit is, which is the credit's.  It reads
    only the records the run committed and the verdict's draws, δ and over-cap draws: properties of values the answer
    never states, so it stays in the verify document."""
    full, _ = _taken(passes, t, delta, over)
    clear, _ = _taken(passes, t, delta, 0)
    rho = passes[0].scheme.cap
    calls, sums = [], {}
    for i, p in enumerate(passes):
        for pos, r in enumerate(p.records):
            s, k = r.matmul.shape, p.man["weights"][str(r.matmul.weight)]["k"]
            cells, left = uncredited_cells(r)
            tiles, wrong = full.get((i, pos), ([], Fraction(0)))
            sampled = clear.get((i, pos), ([], Fraction(0)))[1]
            if cells != sum(tiles):
                raise ValueError(f"pass {i} call {r.matmul.index}: the band rule's {cells} credited cells are not its "
                                 f"tiles' {sum(tiles)}")
            checked = cells * r.k_kept
            causes = {**{c: m * r.k_kept for c, m in left.items()}, "kept_width": r.real_rows * s.n * (k - r.k_kept)}
            terms = {"epsilon_macs": sampled, "over_cap_macs": wrong - sampled, "rho_macs": rho * (checked - wrong),
                     "credited_macs": (1 - rho) * (checked - wrong)}
            calls.append({"pass": i, "call": r.matmul.index, "weight": r.matmul.weight, "m": s.m, "k": k, "n": s.n,
                          "real_rows": r.real_rows, "k_kept": r.k_kept,
                          "rows": {"excluded_a": len(r.excluded_a), "excluded_b": len(r.excluded_b),
                                   "voluntary_a": len(r.voluntary_a), "voluntary_b": len(r.voluntary_b)},
                          "macs": r.real_rows * s.n * k, "checked_macs": checked, "uncredited_macs": causes,
                          **{x: str(v) for x, v in terms.items()}})
            for x, v in (("macs", r.real_rows * s.n * k), ("checked_macs", checked), *causes.items(), *terms.items()):
                sums[x] = sums.get(x, 0) + v
    totals = {x: str(v) if isinstance(v, Fraction) else v for x, v in sums.items()}
    return {"schema": UNCREDITED_SCHEMA, "rho": str(rho), "over_cap_draws": over, "calls": calls,
            "totals": {**totals, "credited": math.floor(sums["credited_macs"])}}


def rejected(reason: str, commit: str) -> dict:
    """A run's verdict that rejects for ``reason``: core's answer form, with neither outputs nor profile, and its line."""
    return {"answer": {"protocol": PROTOCOL, "verdict": "reject", "reason": reason, "commitment": None, "public": None,
                       "profile": None}, "line": f"REJECT {PROTOCOL}: {reason}; verifier {commit}"}


def verify_passes(paths: list[Path], draws: int, jobs: int, commit: str, coins: CoinCommitment | LiveCommitment,
                  source: bytes | None = None,
                  fast: bool = False, cross_check: int = 0, delta: Fraction = DELTA, window: dict | None = None,
                  rounds: list[int] | None = None, dropped: tuple[int, ...] = (), record: Path | None = None,
                  draw: Any = node_draw, over_cap: str = UNCREDITED, test_instance: bool = False) -> dict:
    """One served run's verdict over all its passes, in core's answer form (`core_check.answer_of`): every pass is read
    and held to what it published, and to the window record `window` where there is one (`tie`), then one source of the
    verifier's coins (`pouw.verifier.coins`), drawn after, keys `draws` work-weighted tiles over every pass's tickets
    together (`clear_work.draw` on their records in order), and each pass opens one tile per excluded row.  It accepts iff
    every opened tile passes the tile rule, and credits the run at Pearl's assumption (`credit_at_assumption`) at δ.

    ``over_cap`` is the rule on an opened tile that passes every check but its inert-work bound (`clear_work.cap_failure`):
    ``uncredited`` (CMP-311 option A, the default) accepts it with its bound's reason and credits it nothing, the credit
    taking X_O, the draws that landed on such tiles; ``reject`` rejects the run.  The verdict records the rule it ran under
    (``over_cap``, and the profile's).  The run of record is ``uncredited`` over passes with no voluntary row, since one
    uncredits its whole band or block (`pearl_c_work.UnitRecord.credited`; ``served_passes.py verify``).

    The passes are all of one salt, and each one's instance is the served one, ``k_min`` `K_MIN` at `TILE` × `TILE` tiles:
    a pass of another is a test instance's, read only with ``test_instance``, whose verdict credits nothing (``credited``
    0), as ``--recorded``'s salt claims nothing.

    ``paths`` are the passes the prover retained, pass i the served run's round ``rounds[i]`` (default i), the rounds
    ``dropped`` left out before the coin: they credit nothing, and W may register them (`tie`).  ``record`` is the
    verifier's record of its coin: the first verdict writes the run's commitment and the source there before any replay,
    and a later one over passes whose commitment is another (a retention or a give-up changed after the coin) is refused
    (`AFTER_COIN`) while one over the same takes the same source, so a retry never redraws.  ``draw(C, asking)`` is the
    source's draw on the run's commitment C (module docstring, "The run's draw"): it calls ``asking(info)`` just before it
    asks, which marks the record (`ASKED`) so a draw that fails or is cut short is never asked again (`ASKED_ONCE`), and
    returns (S, the record's ``draw``).  A given ``source`` (a test's) is recorded as the node's."""
    if over_cap not in OVER_CAP:
        raise ValueError(f"over_cap {over_cap!r} is none of {', '.join(OVER_CAP)}")
    named = [read(x)[2] for x in paths]
    out = {"passes": named, "verifier_commit": commit, "draws": draws, "delta": str(delta), "over_cap": over_cap,
           **({"test_instance": True} if test_instance else {}),
           **({"rounds": list(rounds), "dropped": list(dropped)} if rounds is not None else {})}

    def reject(reason: str) -> dict:
        out.update(rejected(reason, commit))
        return out

    if not paths:
        return reject("the run holds no pass with calls")
    try:
        why = passes_distinct([read(x)[0] for x in paths])
        if why is not None:
            out["refused"] = list(why)
            raise Refused(why)
        passes = [load(x, coins) for x in paths]
        if len({p.scheme.name for p in passes}) != 1:
            raise ValueError("the passes are not all of one scheme")
        if len({p.epoch.salt for p in passes}) != 1:
            raise ValueError("the passes are not all of one salt")
        if not test_instance and (odd := next(((i, p.scheme) for i, p in enumerate(passes)
                                               if (p.scheme.k_min, tuple(p.scheme.tile)) != (K_MIN, (TILE, TILE))), None)):
            i, s = odd
            raise ValueError(f"pass {i}'s instance is k_min {s.k_min} at {s.tile[0]} x {s.tile[1]} tiles, not the served "
                             f"{K_MIN} at {TILE} x {TILE}: a test instance is read only as one, crediting nothing")
        tied = None if window is None else tie(window, passes, rounds, dropped)
    except Exception as ex:  # noqa: BLE001 - a run that can't be checked is rejected, with the reason
        return reject(f"{type(ex).__name__}: {ex}")
    if tied is not None:
        out["tie"] = tied
    scheme, root = passes[0].scheme, run_commitment(passes)
    asked = None if record is None else record.with_name(record.name + ASKED)
    if record is not None and record.exists():
        held = json.loads(record.read_text())
        if held["commitment"] != root.hex() or (source is not None and source.hex() != held["source"]):
            out["refused"] = list(AFTER_COIN)
            return reject(f"{AFTER_COIN[0]}: {AFTER_COIN[1]} (its coin was drawn on run {held['commitment'][:16]}, "
                          f"these passes are run {root.hex()[:16]})")
        source, drew = bytes.fromhex(held["source"]), held.get("draw")
    elif asked is not None and asked.exists():
        out["refused"] = list(ASKED_ONCE)
        return reject(f"{ASKED_ONCE[0]}: {ASKED_ONCE[1]} (asked on run {root.hex()[:16]}: {asked.read_text().strip()})")
    else:
        def asking(info: dict) -> None:
            if asked is not None:
                asked.parent.mkdir(parents=True, exist_ok=True)
                asked.write_text(json.dumps({"commitment": root.hex(), **info}) + "\n")

        if source is not None:
            drew = {"from": DRAW_NODE, "claim": NODE_CLAIM}
        else:
            try:
                source, drew = draw(root, asking)
            except Exception as ex:  # noqa: BLE001 - a draw that fails is a visible rejection, never redrawn
                return reject(f"the verifier's draw: {type(ex).__name__}: {ex}")
        if record is not None:
            record.parent.mkdir(parents=True, exist_ok=True)
            tmp = record.with_name(record.name + ".tmp")
            tmp.write_text(json.dumps({"schema": COIN_SCHEMA, "commitment": root.hex(), "source": source.hex(), "passes": named,
                                       **({"rounds": list(rounds), "dropped": list(dropped)} if rounds is not None else {}),
                                       "draw": drew}, indent=1) + "\n")
            os.replace(tmp, record)
    out["draw"] = (drew or {}).get("from")
    key = derive(source, RUN_DRAW_DOMAIN, {"run": root})
    at = [0]
    for p in passes:
        at.append(at[-1] + len(p.records))
    records = [r for p in passes for r in p.records]
    drawn: dict[int, list] = {}
    for pos, tile in CW.draw(scheme, records, key, draws):
        i = bisect.bisect_right(at, pos) - 1
        drawn.setdefault(i, []).append((pos - at[i], tile))
    todo, picked = [], []
    for i, p in enumerate(passes):
        mine, ex = p.tile_ids(drawn.get(i, [])), p.tile_ids(CW.exclusion_tiles(p.scheme, p.records))
        todo += [(i, pos, tid) for pos, tid in mine + ex]
        picked += [True] * len(mine) + [False] * len(ex)
    excluded = len(todo) - sum(map(len, drawn.values()))
    verdicts, fast = replay(passes, todo, jobs, fast, cross_check)
    uncredited = over_cap == UNCREDITED
    bad = [f"pass {i} tile {t}: {r}" for i, t, good, r, over, _ in verdicts if not good and not (over and uncredited)]
    bad += [f"pass {i} tile {t}: the batch replay's verdict is not the scalar one's" for i, t, *_, same in verdicts if same is False]
    over_draws = sum(1 for v, d in zip(verdicts, picked, strict=True) if d and v[4])
    credit = credit_at_assumption(passes, draws, delta, over_draws)
    if test_instance:
        credit["credited"] = 0
    out.update(scheme=scheme.name, source=source.hex(), commitment=root.hex(), replay="batch" if fast else "scalar",
               verdicts=[{"pass": i, "tile": t, "ok": g or (over and uncredited), "reason": r}
                         for i, t, g, r, over, _ in verdicts])
    if bad:
        return reject("; ".join(bad))
    public = {"work": credit, "unit": "MAC", "assumption": ASSUMPTION, "scheme": scheme.name,
              "passes": [{"transcript": p.man["commitment"]["root"], "records": p.man["records_root"], "salt": p.epoch.salt.hex(),
                          "activations": [u["root_a"] for u in p.man["matmuls"]], "y": y_roots(p)} for p in passes],
              **({"tie": tied} if tied is not None else {}),
              **({"rounds": list(rounds), "dropped": list(dropped)} if rounds is not None else {})}
    profile = {"delta": str(delta), "draws": draws, "draw": "work-weighted, with replacement", "source": source.hex(),
               "source_from": out["draw"], "over_cap": over_cap, **({"test_instance": True} if test_instance else {}),
               "exclusion_tiles": excluded, "tiles": [[i, t] for i, t, *_ in verdicts]}
    ys = [r for p in passes for r in y_roots(p)]
    tie_clause = "" if tied is None else (f", tied to W {tied['record'][:16]} on its {tied['tables']} tables"
                                          + (f", {tied['weights']}" if "weights" in tied else ""))
    if dropped:
        tie_clause += f", {len(dropped)} passes dropped before the coin (rounds {', '.join(map(str, dropped))})"
    if over_draws:
        tie_clause += f", {over_draws} draws on tiles over their inert-work bound credited nothing"
    if test_instance:
        tie_clause += ", a test instance's, credited nothing"
    else:
        out["uncredited"] = uncredited_calls(passes, draws, delta, over_draws)
    out.update(answer={"protocol": PROTOCOL, "verdict": "accept", "reason": None, "commitment": root.hex(), "public": public,
                       "profile": profile},
               line=(f"ACCEPT {PROTOCOL} {scheme.name}: {len(todo) - excluded} drawn and {excluded} exclusion tiles over "
                     f"{len(passes)} passes, y committed and checked on {sum(r is not None for r in ys)} of {len(ys)} calls"
                     f"{tie_clause}; "
                     f"{credit['credited']:,} of {credit['macs']:,} MACs credited at ε {float(Fraction(credit['epsilon'])):.4g}, "
                     f"δ {delta} (Pearl's Assumption 1); run {root.hex()[:16]}, source {source.hex()[:16]}; verifier {commit}"))
    return out


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("manifest", type=Path)
    p.add_argument("--draws", type=int, default=DRAWS)
    p.add_argument("--jobs", type=int, default=os.cpu_count() or 1)
    p.add_argument("--commit", default=settings.RESEARCH_SOURCE_SHA or "unknown")
    p.add_argument("--expect-reject", action="store_true", help="the pass is a negative control: exit 0 only if it is rejected")
    p.add_argument("--fast", action="store_true", help="replay the tiles on fast_tile's batch forms (the scalar reference's verdicts)")
    p.add_argument("--cross-check", type=int, default=64, metavar="N",
                   help="with --fast, replay the first tile and every N-th after it, in the draw's order, on the scalar "
                        "reference too; any verdict that differs rejects the pass and exits 2 (0: none)")
    p.add_argument("--coins", type=Path, help="the epoch's coin commitment as its challenger published it (e2e.py's coins.json)")
    p.add_argument("--recorded", action="store_true", help="the pass was recorded before epoch coins: its salt is its beacon's")
    a = p.parse_args(argv)
    coins = epoch_coins(a.coins) if a.coins else None
    r = verify(a.manifest, a.draws, a.jobs, a.commit, fast=a.fast, cross_check=a.cross_check, coins=coins, recorded=a.recorded)
    print(r["line"], flush=True)
    (a.manifest.parent / "verify.json").write_text(json.dumps(r, indent=1) + "\n")
    if r.get("cross_check_failed"):
        return 2
    return 0 if r["accepted"] != a.expect_reject else 1


if __name__ == "__main__":
    sys.exit(main())
