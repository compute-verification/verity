"""`members_files.sh`'s input: one per-member case as the executable's `check` takes it, in DIR: params.json (the
parameters, canonical), commitment.bin (the registration record) and blobs.txt (the blobs' paths, a line each).

  python members_files.py DIR --members M --bin FLOCK_CIRCUIT --lean-bin LEAN_BIN_DIR
  python members_files.py DIR --case CASE

With --members, the case is a stand-in window's (`benchmarks/one_stage/members_standin.py`'s: M members of one unit
over two templates, half each, under ``stratified:1``, so one member of each template is sampled and the rest share
their template's circuit file unsampled), audited as a served window's audit step runs it (``records_audit.py audit``
on a local challenger, ``--deployed-key``, FLOCK_CHALLENGER=1) into DIR/audit, which writes the case. With --case, it
is one a driver wrote (an audit's ``window/case-v2-members.json``). Either way v2's ``check`` must accept the case
(`case_files.run`, each blob under a short name), so the pass `members-files-time` times is an accepted audit's; it
refuses otherwise. Prints one JSON line: the case's members, blobs and their bytes, the audit's seconds, and the
check's seconds and peak RSS.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "benchmarks" / "one_stage"))


def audited(out: Path, members: int, flock: Path, lean: Path) -> tuple[Path, float]:
    """The per-member case of the stand-in window of ``members`` members, audited into ``out``/audit -> its path and
    the audit's seconds."""
    import members_standin as MS

    out.mkdir(parents=True, exist_ok=True)
    win = MS.window(out / "window", members)
    main = out / "main.txt"
    main.write_text(subprocess.run(["git", "-C", str(ROOT), "rev-list", "--first-parent", "HEAD"], capture_output=True,
                                   text=True, check=True).stdout)
    env = {k: v for k, v in os.environ.items() if not k.startswith(("FC_", "FLOCK_")) and k != "PYTHONPATH"}
    env.update(RAYON_NUM_THREADS=str(len(os.sched_getaffinity(0))), RUST_MIN_STACK=str(64 << 20), FLOCK_CHALLENGER="1")
    t = time.monotonic()
    p = subprocess.run(MS.audit_argv(win, out / "audit", main, flock, lean), capture_output=True, text=True, env=env)
    case = out / "audit" / "window" / "case-v2-members.json"
    if p.returncode != 0 or not case.is_file():
        why = out / "audit" / "window" / "case-v2-members-error.txt"
        said = why.read_text() if why.is_file() else "no reason written"
        raise SystemExit(f"the audit of the stand-in window wrote no per-member case (exit {p.returncode}): {said}\n"
                         f"{p.stdout[-3000:]}\n{p.stderr[-3000:]}")
    return case, round(time.monotonic() - t, 1)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dir", type=Path)
    ap.add_argument("--case", type=Path, help="a driver's per-member case")
    ap.add_argument("--members", type=int, help="the stand-in window's members")
    ap.add_argument("--bin", type=Path, help="Flock's prover, for the stand-in window's audit")
    ap.add_argument("--lean-bin", type=Path, help="the directory of flock-verify and flock-firewall, for that audit")
    a = ap.parse_args()
    if (a.case is None) == (a.members is None) or (a.members is not None and not (a.bin and a.lean_bin)):
        ap.error("give --case, or --members with --bin and --lean-bin")
    import case_files as CF
    from verity.protocols.proofs.spot.verifier.one_stage import proofs as PF

    path, audit_s = (a.case, None) if a.case else audited(a.dir, a.members, a.bin, a.lean_bin)
    case = json.loads(path.read_text())
    got = CF.run(case, PF.v2_executable(), "check", a.dir / "check")
    line = json.loads(got["line"]) if got["line"] else {}
    if got["rc"] != 0 or line.get("verdict") != "accept":
        raise SystemExit(f"v2's check does not accept the case (exit {got['rc']}: {str(got['line'])[:1500]}), so its "
                         f"pass is no accepted audit's and nothing is timed")
    params = json.dumps(case["params"], sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False)
    (a.dir / "params.json").write_text(params)
    (a.dir / "commitment.bin").write_bytes(bytes.fromhex(case["commitment"]))
    (a.dir / "blobs.txt").write_text("".join(f"{b}\n" for b in case["blobs"]))
    print(json.dumps({"members": len(case["params"]["proofs"]["members"]), "blobs": got["linked"],
                      "blob_bytes": got["bytes"], "audit_s": audit_s,
                      "check": {x: got[x] for x in ("rc", "seconds", "peak_rss_kb")} | {"verdict": line["verdict"]}}))


if __name__ == "__main__":
    main()
