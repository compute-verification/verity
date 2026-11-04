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

Prints one line, ``ACCEPT …`` or ``REJECT …``, naming the pass's run and the manifest (`transcript`, as the panel's
``--transcript`` takes it), writes ``verify.json`` beside the manifest, and exits 0 only on acceptance, or with
``--expect-reject`` (a negative control's pass) only on rejection.  An ACCEPT is replay-verified (development shortcut; the
protocol's check is a ZK proof, PROTOCOL.md).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import mmap
import os
import sys
import multiprocessing
from pathlib import Path

import numpy as np
from verity.ml.commitments import Commitment, Opening
from verity.core.protocols.pouw.audit import (
    HM96,
    HM96_TRANSCRIPT,
    HM96_WEIGHTS,
    Epoch,
    hm96_salt,
    opened_value,
)
from verity.core.protocols.pouw import verifier as PV
from verity.core.protocols.pouw.coins import CoinCommitment
from verity.core.protocols.pouw.credit import passes_distinct
from verity.core.protocols.pouw.protocol import Context, Matmul, Shape
from verity.core.protocols.pouw.schemes import pearl_c_work as PW
from verity_vllm.protocols.pouw import pearl_c as PD
from verity_config.settings import of

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "clear"))
import fast_tile as FT
import clear_work as CW
from clear_audit import TileOpening, Transcript, Verifier

settings = of(__file__)
DRAWS = 3


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


_JOB: dict = {}


def check(pick: tuple[int, int]) -> tuple[int, bool, str | None, bool | None]:
    """One tile's verdict and, on `--fast`, for every `--cross-check`-th tile id whether the scalar reference's verdict is the
    same one (None where it was not replayed twice)."""
    tid, ok, reason = _check(pick)
    every, undo = _JOB.get("cross_check", 0), _JOB.get("undo")
    if undo is None or not every or tid % every:
        return tid, ok, reason, None
    undo()
    try:
        same = _check(pick) == (tid, ok, reason)
    finally:
        _JOB["undo"] = FT.install(_JOB["scheme"])
    return tid, ok, reason, same


def _check(pick: tuple[int, int]) -> tuple[int, bool, str | None]:
    """One tile, as `clear_work.audit` checks it: the openings and digest (`Verifier.check_tile`), then the tile rule."""
    j = _JOB
    scheme, epoch, transcript, prover, records, verifier = (j[x] for x in ("scheme", "epoch", "transcript", "prover", "records",
                                                                           "verifier"))
    pos, tid = pick
    ref = verifier.refs[tid]
    opened = prover.open(epoch, tid)
    v = verifier.check_tile(transcript, opened)
    if not v.ok:
        return tid, False, v.reason
    u = epoch.workload.matmuls[pos]
    ctx = Context(epoch.salt, u, transcript.activation_roots[pos], epoch.weight_roots[u.weight])
    h = scheme.commitment_hash
    a, b = [opened_value(h, o.value) for o in opened.a], [opened_value(h, o.value) for o in opened.b]
    reason = CW.check_opened(scheme, records[pos], ref.tile, a, b, scheme.encode(ctx, scheme.noise(ctx), a, b, ref.tile))
    return tid, reason is None, reason


def verify(path: Path, draws: int, jobs: int, commit: str, source: bytes | None = None, fast: bool = False,
           cross_check: int = 0, coins: CoinCommitment | None = None, recorded: bool = False) -> dict:
    """The pass's verdict, its epoch re-started under `coins`, the epoch's published coin commitment
    (`pouw_pearl_c_device.manifest_epoch`; `recorded` reads a pass recorded before epoch coins).  `fast` replays its tiles on `fast_tile`'s batch forms where they take the scheme (Pearl-C v1 on a
    promoted device; otherwise the scalar reference, ``replay``), and every `cross_check`-th tile id on the scalar reference
    too: any verdict that differs is ``cross_check_failed``, and the pass is not accepted."""
    raw = path.read_bytes()
    man, digest = json.loads(raw), hashlib.sha256(raw).hexdigest()
    named = f"{man.get('run')}:{path} sha256:{digest}"
    out = {"manifest": str(path), "transcript": named, "transcript_sha256": digest, "verifier_commit": commit, "draws": draws,
           "replay": "batch" if fast else "scalar"}
    try:
        PD.manifest_schema(man, recorded)
        why = passes_distinct([man])
        if why is not None:
            out["refused"] = list(why)
            raise ValueError(": ".join(why))
        scheme = PD.scheme_of(man["scheme"], **{k: tuple(v) if isinstance(v, list) else v for k, v in man.get("instance", {}).items()})
        matmuls = tuple(Matmul(u["index"], Shape(u["m"], u["k"], u["n"]), u["weight"]) for u in man["matmuls"])
        epoch, out["salt"] = PD.manifest_epoch(man, coins, recorded)
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
        source = PV.coins(32) if source is None else source
        key = CW.replay_key(CW.RecordedDraw(1, source), transcript.commitment.root, bytes.fromhex(man["records_root"]), epoch.salt)
        ids = {(ref.matmul, ref.tile.rows[0], ref.tile.cols[0]): ref.id for ref in verifier.refs}
        picks = CW.draw(scheme, records, key, draws)
        picks += CW.exclusion_tiles(scheme, records)
        todo = [(pos, ids[(pos, t.rows[0], t.cols[0])]) for pos, t in picks]
        if fast and (scheme.forming != "v1" or scheme.device.g is None):
            fast, out["replay"] = False, "scalar"
        _JOB.clear()
        _JOB.update(scheme=scheme, epoch=epoch, transcript=transcript, records=records, verifier=verifier,
                    prover=Opened(man, Files(path.parent), verifier.refs), cross_check=cross_check,
                    undo=FT.install(scheme) if fast else None)
        try:
            if jobs > 1 and len(todo) > 1:
                # fork: the workers unpickle their function from this module, which callers load from its path, not by an importable name
                with multiprocessing.get_context("fork").Pool(min(jobs, len(todo))) as pool:
                    verdicts = pool.map(check, todo, chunksize=1)
            else:
                verdicts = [check(p) for p in todo]
        finally:
            if _JOB.get("undo") is not None:
                _JOB.pop("undo")()
        differ = [t for t, _, _, same in verdicts if same is False]
        ok = all(v[1] for v in verdicts) and not differ
        w_ref, credit = CW.work(scheme, records)
        bad = "; ".join(f"tile {t}: {r}" for t, good, r, _ in verdicts if not good)
        if differ:
            bad = "; ".join(filter(None, [bad, f"the batch replay's verdict is not the scalar one's on tiles {differ}"]))
            out["cross_check_failed"] = differ
        if fast:
            out["cross_checked"] = [t for t, _, _, same in verdicts if same is not None]
        at = CW.draw_wref(scheme)
        out.update(accepted=ok, scheme=scheme.name, source=source.hex(), tiles=[t for t, *_ in verdicts],
                   draw_wref=at.statement if at else "wref",
                   drawn=draws, exclusion_tiles=len(todo) - draws,
                   verdicts=[{"tile": t, "ok": g, "reason": r} for t, g, r, _ in verdicts], w_ref=str(w_ref), credit=str(credit))
        out["line"] = (f"{'ACCEPT' if ok else 'REJECT'} {scheme.name} {man['entry']}: clear_work's tiles ({draws} drawn, "
                       f"{len(todo) - draws} for excluded rows) of {c['count']} over {len(matmuls)} matmuls"
                       f"{'' if ok else ' (' + bad + ')'}; transcript root {c['root'][:16]}, records {man['records_root'][:16]}, "
                       f"source {source.hex()[:16]}; transcript {named}; verifier {commit}"
                       f"{'; batch replay, ' + str(len(out['cross_checked'])) + ' tiles cross-checked' if fast else ''}")
    except Exception as ex:  # noqa: BLE001 - a transcript that can't be checked is rejected, with the reason
        out.update(accepted=False, error=f"{type(ex).__name__}: {ex}",
                   line=f"REJECT {man.get('scheme')} {man.get('entry')}: {type(ex).__name__}: {ex}; transcript {named}; verifier {commit}")
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
