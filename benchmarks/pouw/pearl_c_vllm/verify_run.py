"""The reference replay on a vLLM pass's transcript (``pouw_pearl_c_device``'s retained pass): the work law
(`clear_work`), over the pass's whole workload, with the prover's openings read from the pass's files.

  python benchmarks/pouw/pearl_c_vllm/verify_run.py RUN_DIR/manifest.json --coins COINS.json [--draws T] [--jobs J] [--commit SHA] [--fast]

- **The epoch** is re-started from the manifest's salt coin and every weight's root (`Epoch.start`): the coin must open
  against the epoch's coin commitment as the challenger published it (``--coins``, not the pass's copy), as the salt coin,
  opened after the weights' root, and the salt must be the manifest's.  A pass recorded before epoch coins (a v1 manifest)
  carries a beacon instead, and is read only with ``--recorded`` (`Epoch.unverified`, a salt that claims nothing).
- **The call indices** are each below 2^32 and none twice (`credit.passes_distinct`: ``call-range``, ``repeated-call``);
  `served_passes.py` holds a run's passes of one salt to distinct indices too.
- **The records** (each matmul's real rows, width and excluded rows; each weight's excluded rows) are the manifest's.
  Their root must be the one the pass published before any draw.
- **The draws' source** is 32 bytes of the Lean verifier's coins (`pouw.verifier.coins`, `IO.getRandomBytes`,
  `uniform/io-getrandombytes`), the verifier's own, drawn here after the transcript is fixed: a
  `clear_work.RecordedDraw` of round 1, after the commitment's 0.
- **The prover only opens.** Each opening's value and path come from the pass's files: A's rows and tree levels per call,
  B's rows and tree levels per weight, the tile tree.  Each is checked against the published roots.  Under ``hm96-sha512``
  (-h3's) a node is 64 bytes and each value is followed by its salt, `audit.hm96_salt` under the manifest's
  ``hm96_salt_key`` (the prover's, retained for this replay), and the scheme reads the value less its salt.
- **The tiles** are `clear_work.audit`'s: its t work-weighted draws (`draw`) and one tile per excluded row
  (`exclusion_tiles`).  Each is held to `Verifier.check_tile` and `check_opened`, as `audit` does, but mapped over J
  processes, since each costs about 11 s of CPU per 1,024 of k (``--fast``: about a tenth of that on `fast_tile`'s batch
  forms, with every ``--cross-check``-th tile id replayed on the scalar reference too).  `tests/test_pearl_c_vllm_verify.py` holds this to `audit`'s
  own verdict.
- **The served y**, where a call commits it (`served_y`, the manifest's ``y``): an accepted tile's rows of y open against
  its root, and each of the tile's words must be the unscale of the tile's checked U (`pearl_c_u.y`, the kernels'
  epilogue), so the y that leaves the linear is the one the credited rows computed.

`verify_passes` is a served run's verdict over all its passes (`served_passes.py verify`): one source of the verifier's
coins, drawn once every pass's roots are read, keys t draws over all the passes' tickets together, and an accept credits
the run's credited MACs at Pearl's Assumption 1, less what ε of the tickets' weight could hold at δ, in core's answer form.

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

import numpy as np
from verity.ml.commitments import Commitment, Opening, VerificationLimits, verify_opening
from verity.core.protocols.pouw.audit import (
    HM96,
    HM96_TRANSCRIPT,
    HM96_WEIGHTS,
    ROW_SCHEMA,
    Epoch,
    hm96_salt,
    opened_value,
)
from verity.core.protocols.pouw import verifier as PV
from verity.core.protocols.pouw.coins import CoinCommitment
from verity.core.protocols.pouw.credit import passes_distinct
from verity.core.protocols.pouw.protocol import Matmul, Shape, Tile, tagged
from verity.core.protocols.pouw.schemes import pearl_c_u as PCU
from verity.core.protocols.pouw.schemes import pearl_c_work as PW
from verity.errors import ProtocolError
from verity.ml.randomness import derive
from verity_vllm.protocols.pouw import pearl_c as PD
from verity_vllm.protocols.pouw.pearl_c import served_y as SY
from verity_config.settings import of

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "clear"))
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
        """The tile's rows of its call's committed y (`served_y`)."""
        ref = self.refs[tile_id]
        u, y = epoch.workload.matmuls[ref.matmul], self.man["matmuls"][ref.matmul]["y"]
        rows, tree = self.files(y["rows"]), self.files(y["tree"])
        return tuple(opening(tree, u.shape.m, i, self._salted(f32_row(rows, u.shape.n, i), SY.HM96_Y + u.index, i), self.nb)
                     for i in ref.tile.rows)


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


def load(path: Path, coins: CoinCommitment | None = None, recorded: bool = False, man: dict | None = None) -> Pass:
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
    records = [PW.UnitRecord(mm, u["real_rows"], u["k_kept"], tuple(u["excluded_a"]),
                             tuple(man["weights"][str(mm.weight)]["excluded_b"])) for mm, u in zip(matmuls, man["matmuls"])]
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


def check(pick: tuple[int, int, int]) -> tuple[int, int, bool, str | None, bool | None]:
    """One tile's verdict, (pass, tile id, ok, reason, same): on `--fast`, for every `--cross-check`-th tile id whether the
    scalar reference's verdict is the same one (None where it was not replayed twice)."""
    got = _check(pick)
    every, undo = _JOB.get("cross_check", 0), _JOB.get("undo")
    if undo is None or not every or got[1] % every:
        return *got, None
    undo()
    try:
        same = _check(pick) == got
    finally:
        _JOB["undo"] = FT.install(_JOB["scheme"])
    return *got, same


def _check(pick: tuple[int, int, int]) -> tuple[int, int, bool, str | None]:
    """One tile, (pass, record position, tile id), as `clear_work.audit` checks it: the openings and digest
    (`Verifier.check_tile`), then the tile rule."""
    i, pos, tid = pick
    p = _JOB["passes"][i]
    scheme, epoch, transcript = p.scheme, p.epoch, p.transcript
    ref = p.verifier.refs[tid]
    try:
        opened = p.prover.open(epoch, tid)
    except (OSError, ValueError) as ex:
        return i, tid, False, f"the prover's files don't open it: {type(ex).__name__}: {ex}"
    v = p.verifier.check_tile(transcript, opened)
    if not v.ok:
        return i, tid, False, v.reason
    h = scheme.commitment_hash
    a, b = [opened_value(h, o.value) for o in opened.a], [opened_value(h, o.value) for o in opened.b]
    reason = CW.check_opened(scheme, p.records[pos], ref.tile, a, b, v.ops)
    if reason is None and "y" in p.man["matmuls"][pos]:
        try:
            reason = check_y(p, pos, ref.tile, v, p.prover.open_y(epoch, tid))
        except (OSError, ValueError) as ex:
            reason = f"the prover's files don't open its y: {type(ex).__name__}: {ex}"
    return i, tid, reason is None, reason


def check_y(p: Pass, pos: int, tile: Tile, v, opened: tuple[Opening, ...]) -> str | None:
    """The served y on an accepted tile: its rows open against the call's y root, and each of the tile's words is the unscale
    of the tile's checked U at its row's and column's α (`pearl_c_u.y`, the kernels' epilogue)."""
    if p.scheme.commitment_hash != HM96 or p.scheme.forming != "v1":
        return f"y is checked under {HM96} and forming v1, not {p.scheme.name}"
    u, root = p.epoch.workload.matmuls[pos], bytes.fromhex(p.man["matmuls"][pos]["y"]["root"])
    d, c = SY.y_domain(p.epoch.salt, u.index, u.shape.m), Commitment(root, u.shape.m)
    try:
        if not all(verify_opening(d, c, o, ROW_SCHEMA, LIMITS) for o in opened):
            return "y opening"
    except ProtocolError:
        return "y opening"
    for x, (o, urow) in enumerate(zip(opened, v.checked.u, strict=True)):
        words = np.frombuffer(opened_value(HM96, o.value), "<u4")
        for z, j in enumerate(tile.cols):
            try:
                want = PCU.y(urow[z], v.ops.a.alpha[x], v.ops.b.alpha[z]) << 16
            except PCU.NonFinite:
                return f"y[{tile.rows[x]}, {j}]: the unscale of U leaves the finite range"
            if int(words[j]) != want:
                return f"y[{tile.rows[x]}, {j}] is not the unscale of the tile's U"
    return None


def replay(passes: list[Pass], todo: list[tuple[int, int, int]], jobs: int, fast: bool = False, cross_check: int = 0
           ) -> tuple[list[tuple[int, int, bool, str | None, bool | None]], bool]:
    """`check` on each (pass, record position, tile id) of `todo` over J processes, and whether the batch forms ran:
    `fast` replays on `fast_tile`'s batch forms where they take the scheme (Pearl-C v1 on a promoted device; otherwise the
    scalar reference), and every `cross_check`-th tile id on the scalar reference too."""
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
                return pool.map(check, todo, chunksize=1), fast
        return [check(p) for p in todo], fast
    finally:
        if _JOB.get("undo") is not None:
            _JOB.pop("undo")()


def verify(path: Path, draws: int, jobs: int, commit: str, source: bytes | None = None, fast: bool = False,
           cross_check: int = 0, coins: CoinCommitment | None = None, recorded: bool = False) -> dict:
    """The pass's verdict, its epoch re-started under `coins`, the epoch's published coin commitment
    (`pouw_pearl_c_device.manifest_epoch`; `recorded` reads a pass recorded before epoch coins).  `fast` replays its tiles on `fast_tile`'s batch forms where they take the scheme (Pearl-C v1 on a
    promoted device; otherwise the scalar reference, ``replay``), and every `cross_check`-th tile id on the scalar reference
    too: any verdict that differs is ``cross_check_failed``, and the pass is not accepted."""
    man, digest, named = read(path)
    out = {"manifest": str(path), "transcript": named, "transcript_sha256": digest, "verifier_commit": commit, "draws": draws,
           "replay": "batch" if fast else "scalar"}
    try:
        p = load(path, coins, recorded, man)
        scheme, records, c, out["salt"] = p.scheme, p.records, man["commitment"], p.salt
        source = PV.coins(32) if source is None else source
        key = CW.replay_key(CW.RecordedDraw(1, source), p.transcript.commitment.root, bytes.fromhex(man["records_root"]), p.epoch.salt)
        todo = [(0, pos, tid) for pos, tid in p.tile_ids(CW.draw(scheme, records, key, draws) + CW.exclusion_tiles(scheme, records))]
        verdicts, fast = replay([p], todo, jobs, fast, cross_check)
        out["replay"] = "batch" if fast else "scalar"
        differ = [t for _, t, _, _, same in verdicts if same is False]
        ok = all(v[2] for v in verdicts) and not differ
        w_ref, credit = CW.work(scheme, records)
        bad = "; ".join(f"tile {t}: {r}" for _, t, good, r, _ in verdicts if not good)
        if differ:
            bad = "; ".join(filter(None, [bad, f"the batch replay's verdict is not the scalar one's on tiles {differ}"]))
            out["cross_check_failed"] = differ
        if fast:
            out["cross_checked"] = [t for _, t, _, _, same in verdicts if same is not None]
        at = CW.draw_wref(scheme)
        out.update(accepted=ok, scheme=scheme.name, source=source.hex(), tiles=[t for _, t, *_ in verdicts],
                   draw_wref=at.statement if at else "wref",
                   drawn=draws, exclusion_tiles=len(todo) - draws,
                   verdicts=[{"tile": t, "ok": g, "reason": r} for _, t, g, r, _ in verdicts], w_ref=str(w_ref), credit=str(credit))
        out["line"] = (f"{'ACCEPT' if ok else 'REJECT'} {scheme.name} {man['entry']}: clear_work's tiles ({draws} drawn, "
                       f"{len(todo) - draws} for excluded rows) of {c['count']} over {len(records)} matmuls"
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


def epsilon(t: int, delta: Fraction) -> Fraction:
    """The smallest ε (rounded up to 10^-9) with (1 − ε)^t ≤ δ: t accepted work-weighted draws leave more than ε of the
    tickets' weight on wrong tiles except with δ."""
    return Fraction(math.ceil(-math.expm1(math.log(delta) / t) * 1e9) + 1, 10 ** 9)


def credit_at_assumption(passes: list[Pass], t: int, delta: Fraction) -> dict:
    """The run's credit at Pearl's Assumption 1 (`pearl_kw`): a tile that passes the rule cost its credited cells' MACs,
    (1 − ρ) of them at the scheme's cap.  t accepted draws by ticket weight leave at most ε of the tickets' weight on wrong
    tiles except with δ (`epsilon`); the most MACs that weight can hold, filled from the records whose cells hold the most
    MACs a ticket, are taken off."""
    macs, tiles, units = 0, 0, []
    for p in passes:
        for r in p.records:
            cells = [len(rows) * len(cols) for rows, cols in (r.credited(x) for x in p.scheme.tiles(r.matmul.shape))]
            if sum(cells):
                ticket = CW.ticket_cell(p.scheme, r)
                macs += sum(cells) * r.k_kept
                tiles += sum(1 for c in cells if c)
                units.append((r.k_kept / ticket, sum(cells) * ticket))
    eps, rho = epsilon(t, delta), passes[0].scheme.cap
    budget, wrong = eps * sum(w for _, w in units), Fraction(0)
    for ratio, w in sorted(units, reverse=True):
        take = min(budget, w)
        wrong, budget = wrong + take * ratio, budget - take
    credited = (1 - rho) * max(macs - wrong, Fraction(0))
    return {"credited": math.floor(credited), "credited_tiles": tiles, "macs": macs, "rho": str(rho), "epsilon": str(eps)}


def verify_passes(paths: list[Path], draws: int, jobs: int, commit: str, coins: CoinCommitment, source: bytes | None = None,
                  fast: bool = False, cross_check: int = 0, delta: Fraction = DELTA) -> dict:
    """One served run's verdict over all its passes, in core's answer form (`core_check.answer_of`): every pass is read
    and held to what it published, then one source of the verifier's coins (`pouw.verifier.coins`), drawn after, keys
    `draws` work-weighted tiles over every pass's tickets together (`clear_work.draw` on their records in order), and each
    pass opens one tile per excluded row.  It accepts iff every opened tile passes the tile rule, and credits the run at
    Pearl's assumption (`credit_at_assumption`) at δ."""
    named = [read(x)[2] for x in paths]
    out = {"passes": named, "verifier_commit": commit, "draws": draws, "delta": str(delta)}

    def reject(reason: str) -> dict:
        out.update(answer={"protocol": PROTOCOL, "verdict": "reject", "reason": reason, "commitment": None, "public": None,
                           "profile": None}, line=f"REJECT {PROTOCOL}: {reason}; verifier {commit}")
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
    except Exception as ex:  # noqa: BLE001 - a run that can't be checked is rejected, with the reason
        return reject(f"{type(ex).__name__}: {ex}")
    scheme, root = passes[0].scheme, run_commitment(passes)
    source = PV.coins(32) if source is None else source
    key = derive(source, RUN_DRAW_DOMAIN, {"run": root})
    at = [0]
    for p in passes:
        at.append(at[-1] + len(p.records))
    records = [r for p in passes for r in p.records]
    drawn: dict[int, list] = {}
    for pos, tile in CW.draw(scheme, records, key, draws):
        i = bisect.bisect_right(at, pos) - 1
        drawn.setdefault(i, []).append((pos - at[i], tile))
    todo = [(i, pos, tid) for i, p in enumerate(passes)
            for pos, tid in p.tile_ids(drawn.get(i, []) + CW.exclusion_tiles(p.scheme, p.records))]
    excluded = len(todo) - sum(map(len, drawn.values()))
    verdicts, fast = replay(passes, todo, jobs, fast, cross_check)
    bad = [f"pass {i} tile {t}: {r}" for i, t, good, r, _ in verdicts if not good]
    bad += [f"pass {i} tile {t}: the batch replay's verdict is not the scalar one's" for i, t, _, _, same in verdicts if same is False]
    credit = credit_at_assumption(passes, draws, delta)
    out.update(scheme=scheme.name, source=source.hex(), commitment=root.hex(), replay="batch" if fast else "scalar",
               verdicts=[{"pass": i, "tile": t, "ok": g, "reason": r} for i, t, g, r, _ in verdicts])
    if bad:
        return reject("; ".join(bad))
    public = {"work": credit, "unit": "MAC", "assumption": ASSUMPTION, "scheme": scheme.name,
              "passes": [{"transcript": p.man["commitment"]["root"], "records": p.man["records_root"], "salt": p.epoch.salt.hex(),
                          "activations": [u["root_a"] for u in p.man["matmuls"]], "y": y_roots(p)} for p in passes]}
    profile = {"delta": str(delta), "draws": draws, "draw": "work-weighted, with replacement", "source": source.hex(),
               "exclusion_tiles": excluded, "tiles": [[i, t] for i, t, *_ in verdicts]}
    ys = [r for p in passes for r in y_roots(p)]
    out.update(answer={"protocol": PROTOCOL, "verdict": "accept", "reason": None, "commitment": root.hex(), "public": public,
                       "profile": profile},
               line=(f"ACCEPT {PROTOCOL} {scheme.name}: {len(todo) - excluded} drawn and {excluded} exclusion tiles over "
                     f"{len(passes)} passes, y committed and checked on {sum(r is not None for r in ys)} of {len(ys)} calls; "
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
                   help="with --fast, replay every tile whose id is a multiple of N on the scalar reference too; any verdict "
                        "that differs rejects the pass and exits 2 (0: none)")
    p.add_argument("--coins", type=Path, help="the epoch's coin commitment as its challenger published it (e2e.py's coins.json)")
    p.add_argument("--recorded", action="store_true", help="the pass was recorded before epoch coins: its salt is its beacon's")
    a = p.parse_args(argv)
    coins = CoinCommitment.from_json(json.loads(a.coins.read_text())) if a.coins else None
    r = verify(a.manifest, a.draws, a.jobs, a.commit, fast=a.fast, cross_check=a.cross_check, coins=coins, recorded=a.recorded)
    print(r["line"], flush=True)
    (a.manifest.parent / "verify.json").write_text(json.dumps(r, indent=1) + "\n")
    if r.get("cross_check_failed"):
        return 2
    return 0 if r["accepted"] != a.expect_reject else 1


if __name__ == "__main__":
    sys.exit(main())
