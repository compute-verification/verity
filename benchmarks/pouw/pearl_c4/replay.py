"""Pearl-C4's reference replay of a device run's own transcript under the work law (`clear_work.audit`), with
Pearl-C4's domain and debit (`schemes.pearl_c4`).  It opens rows and words and recomputes them in the clear, so its
ACCEPT is replay-verified (development shortcut; the protocol's check is a ZK proof, PROTOCOL.md), never the protocol's
verdict.

**The dump** (`write_dump`) is what a timed call emits, as the device holds it after the call: A's rows (BF16, row-major,
the unit's m rows with its zero filler) and B's, the device's A and B roots (`audit.commit_rows`, SHA-256 whole-row leaves),
and its tile leaves (`pearl_c.leaf`, one per tile in the scheme's order), beside ``call.json`` (the scheme, the unit's
shape, matmul index and weight, its real rows, and the epoch beacon the call's salt was derived from).  The scale bytes
the device's forming wrote may ride along; they are not committed (the verifier forms its own), and `prove` reports how
many fall outside D-SB's byte range (`device_scales`).

**`prove`** makes transcript v0 (`FORMAT`) from a dump, on the host:

* the epoch: B's root as the device committed it, and the salt keyed by the call's recorded epoch beacon
  (`audit.Epoch.unverified`: a recorded replay's salt, with no coin behind it, not the epoch's live coin);
* the transcript commitment over the device's leaves (`audit.transcript_domain`);
* the records: one unit, no channel split, its real rows, and as excluded the rows that fail `row_passes` (D-NF) and the
  rows that pass it but not R1 (`PearlC4.r1_row`, row by row over the workers), as `clear_work.declare` has them;
* the draws' beacon, Fiat-Shamir over the commitment (`draw_beacon`), and the openings of the tiles the work law opens
  there: t draws by the records' credit and one tile per excluded row.

**`verify`** recomputes everything from the transcript and the scheme: the salt, the records against the workload, the
draws, and on every opened tile (`clear_work.check_drawn`) A's and B's openings against the device's roots, the leaf's
against the commitment, the tile's C̃ and U through the scalar reference to the device's leaf, zero filler rows, credited
rows passing the per-row rules and R1 and excluded rows failing one of them, and the replayed debit
(`pearl_c4.tile_debit`: F1′'s base split under the exact noise law given the drawn E lines and the realised scale bytes,
F2's int8 route on flat scales, identity atoms, dead × dead pairs, salt-dead or unneeded A forming, sub-grid blocks, D-24's
pair-sparse row windows) under the cap, ρ = 1/400 of the credited cells' creditOf.  Its report carries every tile's debit and the
accept line (`accept_line`).

  python -m benchmarks.pouw.pearl_c4.replay prove DUMP --run RUN --out transcript.json [--t 8] [--workers N]
  python -m benchmarks.pouw.pearl_c4.replay verify transcript.json --out verify.json [--path P] [--commit SHA]

Shortcuts, stated in every transcript and report (`SHORTCUTS`).
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import sys
from collections.abc import Callable, Sequence
from concurrent.futures import ProcessPoolExecutor
from fractions import Fraction
from functools import partial
from multiprocessing import get_context
from pathlib import Path

from verity.ml.commitments import Commitment, MerkleTree, Opening

from verity.core.protocols.pouw import audit as au
from verity.core.protocols.pouw.protocol import Context, Matmul, Shape, tagged
from verity.core.protocols.pouw.schemes import get
from verity.core.protocols.pouw.schemes import pearl_c4 as F
from verity.core.protocols.pouw.schemes import pearl_c_work as W
from verity_config.settings import of

# the work law and the clear audit are a reference tool's (benchmarks/pouw/clear), imported by name
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "clear"))
import clear_audit as CA
import clear_work as CW

settings = of(__file__)

FORMAT = "verity/pouw/pearl-c4-transcript/v0"
REPORT = "verity/pouw/pearl-c4-replay/v0"
DRAW_BEACON_TAG = "verity/pouw/pearl-c4-replay/draw-beacon/v0"
COMMITTED_ROUND = 0
SHORTCUTS = (
    "the draws' beacon is Fiat-Shamir over the transcript root and the records root (draw_beacon), not a public beacon "
    "round after the commitment",
    "the epoch beacon is a label of the benchmark's operand set, not a public beacon round after the weights",
    "the transcript tree over the tile leaves and the records (row_passes and R1 per row) are computed on the host after "
    "timing",
    "verify opens the drawn tiles' rows and words and recomputes them in the clear: its ACCEPT is replay-verified "
    "(development shortcut; the protocol's check is a ZK proof, PROTOCOL.md)",
)


def _hex(b: bytes) -> str:
    return b.hex()


def _opening_json(o: Opening) -> dict:
    return {"position": o.position, "value": base64.b64encode(o.value).decode(), "path": [p.hex() for p in o.path]}


def _opening(d: dict) -> Opening:
    return Opening(d["position"], base64.b64decode(d["value"]), tuple(bytes.fromhex(p) for p in d["path"]))


def _rows(data: bytes, count: int, k: int) -> list[bytes]:
    if len(data) != count * 2 * k:
        raise ValueError(f"{len(data)} bytes are not {count} rows of {k} BF16 words")
    return [data[i * 2 * k:(i + 1) * 2 * k] for i in range(count)]


# ---- the dump ---------------------------------------------------------------------------------------------------------

def write_dump(path: str | os.PathLike, call: dict, rows_a: bytes, rows_b: bytes, root_a: bytes, root_b: bytes,
               leaves: bytes, scales_a: bytes | None = None, scales_b: bytes | None = None) -> None:
    """A call's dump: ``call.json`` (``scheme``, ``shape`` [m, k, n] of the unit, ``index``, ``weight``, ``real_rows``,
    ``epoch_beacon`` hex, and anything else the caller records) and the device's bytes, with the scale bytes its forming
    wrote when given (not committed: `prove` counts them, `device_scales`)."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    for name, data in (("xa.bin", rows_a), ("xb.bin", rows_b), ("root_a.bin", root_a), ("root_b.bin", root_b),
                       ("leaves.bin", leaves), ("scales_a.bin", scales_a), ("scales_b.bin", scales_b)):
        if data is not None:
            (p / name).write_bytes(data)
    (p / "call.json").write_text(json.dumps(call, indent=1, sort_keys=True) + "\n")


def device_scales(fmt: F.Fmt, data: bytes) -> dict:
    """What a device's formed scale bytes hold against D-SB's range (`pearl_c4.scale_bytes_ok`)."""
    counts = [data.count(bytes([b])) for b in range(256)]
    out = {"bytes": len(data), "outside_range": sum(c for b, c in enumerate(counts) if not F.scale_bytes_ok(fmt, [b])),
           "min": min((b for b in range(256) if counts[b]), default=None),
           "max": max((b for b in range(256) if counts[b]), default=None)}
    if fmt is F.NVFP4:
        out |= {"bit7": sum(counts[128:]), "zero": counts[0], "nan": counts[0x7F]}
    else:
        out |= {"nan": counts[0xFF]}
    return out


def draw_beacon(transcript_root: bytes, records_digest: bytes) -> CW.RecordedDraw:
    return CW.RecordedDraw(COMMITTED_ROUND + 1, tagged(DRAW_BEACON_TAG, transcript_root, records_digest))


def _pool(workers: int, chunksize: int = 1) -> tuple[Callable, Callable[[], None]]:
    """(map, shutdown): the builtin map for one worker, else a process pool's (forkserver: its workers never fork a
    multi-threaded parent)."""
    if workers <= 1:
        return map, lambda: None
    ex = ProcessPoolExecutor(max_workers=workers, mp_context=get_context("forkserver"))
    return partial(ex.map, chunksize=chunksize), partial(ex.shutdown, wait=True)


def _passes(fmt: str, k: int, row: bytes) -> bool:
    return F.row_passes(F.FORMATS[fmt], F.parse_row(row, k))


def _over_r1(s: F.PearlC4, ctx: Context, k_kept: int, row: tuple[str, int, bytes]) -> bool:
    return s.r1_row(ctx, *row, k_kept)


def _scheme(t: dict, scheme: F.PearlC4 | None) -> F.PearlC4:
    s = scheme or get(t["scheme"])
    if not isinstance(s, F.PearlC4) or s.name != t["scheme"]:
        raise ValueError(f"{t['scheme']} is not a Pearl-C4 scheme")
    return s


def _epoch(t: dict, workload: au.Workload) -> au.Epoch:
    roots = {int(w): bytes.fromhex(r) for w, r in t["weight_roots"].items()}
    return au.Epoch.unverified(workload, roots, bytes.fromhex(t["epoch_beacon"]))


def _unit(t: dict) -> Matmul:
    m, k, n = t["shape"]
    return Matmul(t["index"], Shape(m, k, n), t["weight"])


# ---- prove ------------------------------------------------------------------------------------------------------------

def prove(dump: str | os.PathLike, run: str, t: int = 8, workers: int = 1, scheme: F.PearlC4 | None = None) -> dict:
    """Transcript v0 of a dump (the module docstring)."""
    p = Path(dump)
    call = json.loads((p / "call.json").read_text())
    s = _scheme(call, scheme)
    unit = _unit(call)
    m, k, n = call["shape"]
    rows_a, rows_b = _rows((p / "xa.bin").read_bytes(), m, k), _rows((p / "xb.bin").read_bytes(), n, k)
    root_a, root_b, leaves = ((p / nm).read_bytes() for nm in ("root_a.bin", "root_b.bin", "leaves.bin"))
    workload = au.Workload((unit,))
    epoch = au.Epoch.unverified(workload, {unit.weight: root_b}, bytes.fromhex(call["epoch_beacon"]))
    refs = au.tile_refs(s, workload)
    if len(leaves) != 32 * len(refs):
        raise ValueError(f"{len(leaves)} bytes of leaves are not {len(refs)} tile leaves")
    tree = MerkleTree(au.transcript_domain(epoch, len(refs)), {i: leaves[32 * i:32 * i + 32] for i in range(len(refs))},
                      lambda _: au.TILE_SCHEMA)
    tree_a = au.commit_rows(au.activation_binding(epoch.salt, unit.index), unit.index, rows_a)
    tree_b = au.commit_rows(au.weight_binding(unit.weight), -2, rows_b)
    real = call["real_rows"]
    fmap, done = _pool(workers, max(1, (real + n) // (4 * max(workers, 1))))
    try:
        passes = list(fmap(partial(_passes, s.fmt.name, k), rows_a[:real] + rows_b))
        live = [("A", i, rows_a[i]) for i in range(real) if passes[i]]
        live += [("B", j, rows_b[j]) for j in range(n) if passes[real + j]]
        r1 = partial(_over_r1, s, Context(epoch.salt, unit, root_a, root_b), k)
        over = {(side, i) for (side, i, _), o in zip(live, fmap(r1, live)) if o}
    finally:
        done()
    record = W.UnitRecord(unit, real, k, tuple(i for i in range(real) if not passes[i] or ("A", i) in over),
                          tuple(j for j in range(n) if not passes[real + j] or ("B", j) in over))
    splits = {unit.weight: W.Split(k)}
    digest = W.records_root(epoch.salt, epoch.weights_root, splits, [record])
    beacon = draw_beacon(tree.commitment.root, digest)
    picks = (CW.draw(s, [record], CW.replay_key(beacon, tree.commitment.root, digest, epoch.salt), t)
             + CW.exclusion_tiles(s, [record]))
    ids = {(r.matmul, r.tile.rows[0], r.tile.cols[0]): r.id for r in refs}
    opened = sorted({ids[(pos, tile.rows[0], tile.cols[0])] for pos, tile in picks})
    openings = [{"tile": i, "a": [_opening_json(tree_a.open(r)) for r in refs[i].tile.rows],
                 "b": [_opening_json(tree_b.open(c)) for c in refs[i].tile.cols],
                 "leaf": _opening_json(tree.open(i))} for i in opened]
    return {
        "format": FORMAT, "run": run, "scheme": s.name, "shape": [m, k, n], "index": unit.index, "weight": unit.weight,
        "call": call, "epoch_beacon": call["epoch_beacon"], "weight_roots": {str(unit.weight): _hex(root_b)},
        "activation_roots": [_hex(root_a)], "transcript": {"root": _hex(tree.commitment.root), "count": len(refs)},
        "leaves_sha256": hashlib.sha256(leaves).hexdigest(),
        "splits": {str(unit.weight): {"k": k, "side": [], "calibration": ""}},
        "records": [{"real_rows": real, "k_kept": k, "excluded_a": list(record.excluded_a),
                     "excluded_b": list(record.excluded_b)}],
        "committed_round": COMMITTED_ROUND, "draw": {"round": beacon.round, "value": _hex(beacon.value)}, "t": t,
        "openings": openings, "shortcuts": list(SHORTCUTS),
        "host_roots_match": {"a": tree_a.commitment.root == root_a, "b": tree_b.commitment.root == root_b},
        "device_scales": {side: device_scales(s.fmt, (p / f"scales_{side}.bin").read_bytes())
                          for side in ("a", "b") if (p / f"scales_{side}.bin").exists()},
    }


# ---- verify -----------------------------------------------------------------------------------------------------------

class _Openings:
    """The transcript's openings, served to the work law as its prover's."""

    def __init__(self, refs: Sequence[au.TileRef], openings: Sequence[dict]):
        self.refs, self.by_id = refs, {o["tile"]: o for o in openings}

    def open(self, epoch: au.Epoch, tile_id: int) -> CA.TileOpening:
        if tile_id not in self.by_id:
            raise KeyError(f"the transcript does not open tile {tile_id}")
        o = self.by_id[tile_id]
        return CA.TileOpening(self.refs[tile_id], tuple(_opening(x) for x in o["a"]), tuple(_opening(x) for x in o["b"]),
                              _opening(o["leaf"]))


def _check_with_debit(f: Callable, pick) -> tuple[CA.TileCheck, dict | None]:
    scheme = f.args[0]
    scheme.last_debit = None
    v = f(pick)
    return v, None if scheme.last_debit is None else scheme.last_debit.to_json()


def verify(t: dict, workers: int = 1, scheme: F.PearlC4 | None = None) -> dict:
    """The report on transcript v0 `t`: ``accepted``, the reason when it is not, every opened tile's verdict and debit."""
    s = _scheme(t, scheme)
    unit = _unit(t)
    workload = au.Workload((unit,))
    epoch = _epoch(t, workload)
    rec = t["records"][0]
    records = [W.UnitRecord(unit, rec["real_rows"], rec["k_kept"], tuple(rec["excluded_a"]), tuple(rec["excluded_b"]),
                            tuple(rec.get("voluntary_a", ())), tuple(rec.get("voluntary_b", ())))]
    splits = {int(w): W.Split(sp["k"], tuple(sp["side"]), bytes.fromhex(sp["calibration"]))
              for w, sp in t["splits"].items()}
    commitment = Commitment(bytes.fromhex(t["transcript"]["root"]), t["transcript"]["count"])
    transcript = CA.Publication(tuple(bytes.fromhex(r) for r in t["activation_roots"]), commitment, ())
    beacon = CW.RecordedDraw(t["draw"]["round"], bytes.fromhex(t["draw"]["value"]))
    report = {"format": REPORT, "scheme": s.name, "run": t["run"], "shape": t["shape"], "t": t["t"],
              "shortcuts": list(SHORTCUTS), "salt": epoch.salt.hex()}
    fs = draw_beacon(commitment.root, W.records_root(epoch.salt, epoch.weights_root, splits, records))
    if beacon != fs:
        return report | {"accepted": False, "reason": "the draws' beacon is not the Fiat-Shamir of the commitment"}
    debits: list = []
    fmap, done = _pool(workers)

    def tile_map(f, picks):
        out = list(fmap(partial(_check_with_debit, f), picks))
        debits.extend(d for _, d in out)
        return [v for v, _ in out]

    try:
        out = CW.audit_replay(s, epoch, _Openings(au.tile_refs(s, workload), t["openings"]), transcript, splits, records,
                      t["committed_round"], beacon, t["t"], tile_map=tile_map)
    except KeyError as e:
        return report | {"accepted": False, "reason": str(e)}
    finally:
        done()
    refs = au.tile_refs(s, workload)
    cell = CW.cell_credit(s, records[0])
    tiles = []
    for v, d in zip(out.audit.verdicts, debits or [None] * len(out.audit.verdicts)):
        row = {"tile": v.tile, "ok": v.ok, "reason": v.reason}
        if v.tile >= 0:
            ref = refs[v.tile]
            rows, cols = records[0].credited(ref.tile)
            cap = F.CAP * cell * len(rows) * len(cols)
            row |= {"rows": [ref.tile.rows[0], ref.tile.rows[-1]], "cols": [ref.tile.cols[0], ref.tile.cols[-1]],
                    "credited": [len(rows), len(cols)], "cap": str(cap)}
            if d is not None:
                row |= {"debit": d, "debit_over_cap": float(Fraction(d["total"]) / cap) if cap else None}
        tiles.append(row)
    bad = next((v.reason for v in out.audit.verdicts if not v.ok), "")
    ratios = [x["debit_over_cap"] for x in tiles if x.get("debit_over_cap") is not None]
    return report | {
        "accepted": out.audit.accepted, "reason": bad, "tiles": tiles, "opened": len(tiles),
        "max_debit_over_cap": max(ratios) if ratios else None, "wref": str(out.wref), "credit": str(out.credit),
        "job": out.job, "profile": out.audit.profile.to_json() if out.audit.profile else None,
    }


def accept_line(report: dict, transcript: str, sha: str, commit: str) -> str:
    """One line: ACCEPT (replay-verified) or REJECT, the scheme, the transcript checked, what was opened, the largest
    debit/cap, and the replay's commit."""
    m, k, n = report["shape"]
    verdict = "ACCEPT" if report["accepted"] else "REJECT"
    tail = (f"opened={report.get('opened', 0)} t={report['t']} max_debit_over_cap={report.get('max_debit_over_cap')}"
            if report["accepted"] else f"reason={report['reason']!r}")
    return (f"{verdict} {REPORT} {report['scheme']} {transcript} sha256:{sha} m={m} k={k} n={n} {tail} "
            f"verifier={commit}")


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m benchmarks.pouw.pearl_c4.replay")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prove")
    p.add_argument("dump")
    p.add_argument("--run", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--t", type=int, default=8)
    p.add_argument("--workers", type=int, default=os.cpu_count() or 1)
    v = sub.add_parser("verify")
    v.add_argument("transcript")
    v.add_argument("--out", required=True)
    v.add_argument("--path", help="the transcript as the accept line names it (default: <run>:<file name>)")
    v.add_argument("--commit", default=settings.VERITY_COMMIT or "unknown")
    v.add_argument("--workers", type=int, default=os.cpu_count() or 1)
    args = ap.parse_args(argv)
    if args.cmd == "prove":
        t = prove(args.dump, args.run, args.t, args.workers)
        Path(args.out).write_text(json.dumps(t) + "\n")
        print(f"transcript {args.out}: {len(t['openings'])} tiles opened, excluded rows "
              f"{len(t['records'][0]['excluded_a'])} of A, {len(t['records'][0]['excluded_b'])} of B, host roots "
              f"{t['host_roots_match']}, device scale bytes {t['device_scales']}")
        return 0
    raw = Path(args.transcript).read_bytes()
    t = json.loads(raw)
    report = verify(t, args.workers)
    line = accept_line(report, args.path or f"{t['run']}:{Path(args.transcript).name}", hashlib.sha256(raw).hexdigest(),
                       args.commit)
    Path(args.out).write_text(json.dumps(report | {"accept_line": line}, indent=1) + "\n")
    print(line)
    return 0 if report["accepted"] else 1


if __name__ == "__main__":
    sys.exit(main())
