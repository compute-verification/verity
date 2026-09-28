"""The reference replay on pearlc_arm's transcripts: `verity.core.protocols.pouw.audit.Verifier(scheme, epoch).audit(prover, transcript,
Sampled(t), source)`, #449's `pearl_c` audit, unchanged.  The epoch is re-started from the transcript's salt coin and weight
root (`Epoch.start`: the coin must open against the transcript's coin commitment, as the salt coin, after the weight root;
the arm is its run's own challenger, a development shortcut); a transcript recorded before epoch coins (v1) carries a beacon
instead and is read only with --recorded (`Epoch.unverified`, a salt that claims nothing).  The transcript is A's root and
the tile tree's commitment; the draw's source is `os.urandom(32)`, drawn here, after the transcript is fixed (the arm's
challenger opens no draw coin for this process, so the draw is a replay's, as `verify_run`'s is).  The prover only opens: each opening's value and path are read
from the transcript's files (the device's tree levels, the tile leaves, A's and B's rows, every file held to its SHA-256
first).  The verifier checks every path against the published roots and recomputes every drawn tile through the scheme on the
CPU (about 11 s per tile per 1,024 of k).

  python verify.py RUN_DIR [--draws T] [--jobs J] [--commit SHA] [--run ID] [--recorded]

Prints one line per transcript under RUN_DIR/transcripts/*/*/manifest.json and per negative control under
RUN_DIR/negative/*/*/manifest.json (the arm's dump of the call that writes none of the commitment into its poisoned
buffers), `ACCEPT ...` or `REJECT ...`, and writes RUN_DIR/verify.json.  An ACCEPT is replay-verified (development shortcut;
the protocol's check is a ZK proof, PROTOCOL.md).  Exit 0 only if at least one transcript was found, every one was
accepted, and each has a negative control beside it that was rejected; otherwise a `REJECT` line names the
transcript's SHA-256 with the reason.  --commit (default RESEARCH_SOURCE_SHA) names the verifier's tree and --run (default
RESEARCH_RUN_ID, else RUN_DIR's name) the run, in every line.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import mmap
import os
import sys
import multiprocessing
from pathlib import Path

from verity.ml.commitments import Commitment, Opening
from verity.core.protocols.pouw.audit import Epoch, Sampled, TileOpening, Transcript, Verifier, Workload
from verity.core.protocols.pouw.coins import CoinCommitment, CoinOpening
from verity.core.protocols.pouw.protocol import Matmul, Shape
from verity.core.protocols.pouw.schemes import pearl_c as C
from verity.core.protocols.pouw.schemes.pearl_c_device import DEVICES

SCHEMA = "verity-pouw/pearl-c-sm120-transcript/v2"
#: a transcript recorded before epoch coins, salted from a beacon: read only as a recorded one (``--recorded``)
SCHEMA_RECORDED = "verity-pouw/pearl-c-sm120-transcript/v1"
DRAWS = 3


def pow2(n: int) -> int:
    return 1 << max(n - 1, 0).bit_length()


def opening(tree: bytes, count: int, pos: int, value: bytes) -> Opening:
    """Position `pos`'s path in a frame-b3 tree's bytes as the device lays them out: level 0 padded to a power of two, then
    each level above it, 32 bytes a node."""
    path, off, width, cur = [], 0, pow2(count), pos
    while width > 1:
        s = off + (cur ^ 1)
        path.append(bytes(tree[32 * s:32 * s + 32]))
        off, width, cur = off + width, width // 2, cur >> 1
    return Opening(pos, value, tuple(path))


class Opened:
    """The prover's side of the audit from a transcript's files: `digest` and `open`, as `verity.core.protocols.pouw.audit.Prover` has them."""

    def __init__(self, files: dict, verifier: Verifier, matmul: Matmul, count: int):
        self.f, self.refs, self.mm, self.count = files, verifier.refs, matmul, count

    def digest(self, tile_id: int) -> bytes:
        return bytes(self.f["leaves"][32 * tile_id:32 * tile_id + 32])

    def open(self, epoch: Epoch, tile_id: int) -> TileOpening:
        ref, s = self.refs[tile_id], self.mm.shape
        rb = 4 * s.k
        row = lambda name, i: bytes(self.f[name][rb * i:rb * (i + 1)])  # noqa: E731
        return TileOpening(ref, tuple(opening(self.f["tree_a"], s.m, i, row("rows_a", i)) for i in ref.tile.rows),
                           tuple(opening(self.f["tree_b"], s.n, j, row("rows_b", j)) for j in ref.tile.cols),
                           opening(self.f["tree_t"], self.count, tile_id, self.digest(tile_id)))


def load(base: Path, spec: dict):
    """A transcript file, held to its size and SHA-256 (large ones memory-mapped)."""
    path = Path(spec["path"]) if Path(spec["path"]).is_absolute() else base / spec["path"]
    with open(path, "rb") as fh:
        data = mmap.mmap(fh.fileno(), 0, access=mmap.ACCESS_READ) if spec["bytes"] > 1 << 24 else fh.read()
    h = hashlib.sha256()
    for lo in range(0, len(data), 1 << 24):
        h.update(data[lo:lo + (1 << 24)])
    if len(data) != spec["bytes"] or h.hexdigest() != spec["sha256"]:
        raise ValueError(f"{path}: {len(data)} bytes, sha256 {h.hexdigest()}; the manifest says {spec['bytes']}, {spec['sha256']}")
    return data


def transcript_epoch(man: dict, mm: Matmul, recorded: bool = False) -> Epoch:
    """The transcript's epoch: from its salt coin under its coin commitment (`SCHEMA`), or, for a transcript recorded before
    epoch coins (`SCHEMA_RECORDED`) and only when `recorded`, from its beacon; its salt must be the epoch's."""
    roots = {mm.weight: bytes.fromhex(man["weight_root"])}
    schema = man.get("schema")
    if schema == SCHEMA:
        epoch, how = Epoch.start(Workload((mm,)), roots, CoinCommitment.from_json(man["coins"]),
                                 CoinOpening.from_json(man["salt_opening"])), "salt coin"
    elif schema == SCHEMA_RECORDED and recorded:
        epoch, how = Epoch.unverified(Workload((mm,)), roots, bytes.fromhex(man["beacon"])), "recorded beacon"
    elif schema == SCHEMA_RECORDED:
        raise ValueError(f"the transcript is {schema}, recorded before epoch coins: only --recorded reads it")
    else:
        raise ValueError(f"schema {schema!r}, not {SCHEMA}" + (f" or {SCHEMA_RECORDED}" if recorded else ""))
    if epoch.salt.hex() != man["salt"]:
        raise ValueError(f"the transcript's salt is not its {how}'s and weight root's")
    return epoch


def verify(path: Path, draws: int, commit: str, run: str = "", recorded: bool = False) -> dict:
    raw = path.read_bytes()
    man, digest = json.loads(raw), hashlib.sha256(raw).hexdigest()
    s, label = man["shape"], f"{man.get('arm')}/{man['shape']['label']}"
    negative = "negative_control" in man
    tag, tail = "negative control " if negative else "", f"; transcript sha256:{digest}; verifier {commit}" + (f"; run {run}" if run else "")
    out = {"manifest": str(path), "transcript_sha256": digest, "arm": man.get("arm"), "shape": s, "verifier_commit": commit,
           "draws": draws, "negative_control": negative}
    try:
        mm = Matmul(man["matmul"]["index"], Shape(s["m"], s["k"], s["n"]), man["matmul"]["weight"])
        epoch = transcript_epoch(man, mm, recorded)
        a = man["scheme_args"]
        scheme = C.PearlC(forming=a["forming"], device=DEVICES[a["device"]], hashing=a["hashing"])
        if scheme.name != man["scheme"]:
            raise ValueError(f"scheme {man['scheme']!r} is not {scheme.name!r}")
        c = man["commitment"]
        transcript = Transcript((bytes.fromhex(man["activation_root"]),), Commitment(bytes.fromhex(c["root"]), c["count"]), ())
        # The arm's challenger, its run's own, opens no draw coin this process can ask for: like verify_run's replay
        # (`RecordedDraw`), the draw is this verifier's own randomness, drawn after the transcript is fixed.
        verifier = Verifier(scheme, dataclasses.replace(epoch, coins=None))
        files = {name: load(path.parent, spec) for name, spec in man["files"].items()}
        source = os.urandom(32)
        audit = verifier.audit(Opened(files, verifier, mm, c["count"]), transcript, Sampled(draws), source)
        out.update(scheme=scheme.name, accepted=audit.accepted, selected=list(audit.selected), source=source.hex(),
                   verdicts=[{"tile": v.tile, "ok": v.ok, "reason": v.reason} for v in audit.verdicts],
                   profile=audit.profile.to_json() if audit.profile else None)
        tiles = ", ".join(str(t) for t in audit.selected)
        bad = "; ".join(f"tile {v.tile}: {v.reason}" for v in audit.verdicts if not v.ok)
        out["line"] = (f"{'ACCEPT' if audit.accepted else 'REJECT'} {tag}{scheme.name} {s['label']}: verity.core.protocols.pouw.audit.Verifier, "
                       f"Sampled({draws}) drew tiles {tiles} of {c['count']}{'' if audit.accepted else ' (' + bad + ')'}; "
                       f"root_A {man['activation_root'][:16]}, tile root {c['root'][:16]}, source {source.hex()[:16]}{tail}")
    except Exception as ex:  # noqa: BLE001 - a transcript that can't be checked is rejected, with the reason
        out.update(accepted=False, error=f"{type(ex).__name__}: {ex}", line=f"REJECT {tag}{label}: {type(ex).__name__}: {ex}{tail}")
    return out


def _one(job):
    return verify(*job)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("run_dir", type=Path)
    p.add_argument("--draws", type=int, default=DRAWS)
    p.add_argument("--jobs", type=int, default=8)
    p.add_argument("--commit", default=os.environ.get("RESEARCH_SOURCE_SHA", "unknown"))
    p.add_argument("--run", default=os.environ.get("RESEARCH_RUN_ID"))
    p.add_argument("--recorded", action="store_true", help="read transcripts recorded before epoch coins (their beacon's salt)")
    a = p.parse_args(argv)
    run = a.run or a.run_dir.resolve().name
    found = sorted(a.run_dir.glob("transcripts/*/*/manifest.json"))
    negatives = sorted(a.run_dir.glob("negative/*/*/manifest.json"))
    errors = sorted(str(e.parent.relative_to(a.run_dir)) for e in a.run_dir.glob("transcripts/*/*/error.txt"))
    # fork: the workers unpickle their function from this module, which callers load from its path, not by an importable name
    with multiprocessing.get_context("fork").Pool(max(1, min(a.jobs, len(found) + len(negatives)))) as pool:
        results = pool.map(_one, [(f, a.draws, a.commit, run, a.recorded) for f in found + negatives], chunksize=1)
    results, controls = results[:len(found)], results[len(found):]
    for r in results + controls:
        print(r["line"], flush=True)
    for e in errors:
        print(f"REJECT {e}: the arm wrote no transcript ({(a.run_dir / e / 'error.txt').read_text().strip()}); run {run}", flush=True)
    by = {(c["arm"], c["shape"]["label"]): c for c in controls}
    uncontrolled = []
    for r in results:
        c = by.get((r["arm"], r["shape"]["label"]))
        why = "no negative control beside it" if c is None else "its negative control was accepted" if c["accepted"] else None
        if why:
            uncontrolled.append(r["manifest"])
            print(f"REJECT {r['arm']}/{r['shape']['label']}: {why}; transcript sha256:{r['transcript_sha256']}; "
                  f"verifier {a.commit}; run {run}", flush=True)
    ok = (bool(results) and not errors and all(r["accepted"] for r in results) and not uncontrolled
          and not any(c["accepted"] for c in controls))
    (a.run_dir / "verify.json").write_text(json.dumps({"verifier_commit": a.commit, "draws": a.draws, "run": run, "ok": ok,
                                                       "transcripts": results, "negative_controls": controls,
                                                       "uncontrolled": uncontrolled, "missing": errors}, indent=1) + "\n")
    if not found:
        print(f"REJECT: no transcript under {a.run_dir}/transcripts", flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
