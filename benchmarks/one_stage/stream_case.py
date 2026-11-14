"""The streamed check beside `check` on one per-member case: v2's ``check`` (`case_files.run`), then its ``plan`` and
``check-stream`` (`Core.StreamProtocol.main` at `Spot.OneStageZk.Streamed.protocol`, two commands of the same
executable), its blobs listed in the plan's order (the pinned tables the case holds first, in their order), and two
feeds it must refuse: two planned blobs swapped, and one planned blob with a byte changed.

    stream_case.py DIR --case CASE
    stream_case.py DIR --members M --bin FLOCK_CIRCUIT --lean-bin LEAN_BIN_DIR

With --members the case is `members_files.audited`'s stand-in window. Prints one JSON line a command: its exit, its
last line, its seconds and peak RSS; the stream's rows say whether its line is ``check``'s.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Sequence

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "benchmarks" / "one_stage"))
sys.path.insert(0, str(ROOT / "benchmarks" / "verifier_cost"))

import case_files as CF  # noqa: E402


def timed(args: Sequence[str], d: Path, name: str) -> dict[str, Any]:
    """`args` run from `d`, stdout and stderr in ``d/name.out`` and ``d/name.err`` -> exit, last line, seconds, peak."""
    t = time.monotonic()
    with open(d / f"{name}.out", "w") as out, open(d / f"{name}.err", "w") as err:
        p = subprocess.Popen(list(args), cwd=d, stdout=out, stderr=err, preexec_fn=CF._stack)
        _, status, use = os.wait4(p.pid, 0)
    lines = (d / f"{name}.out").read_text().strip().splitlines()
    errs = (d / f"{name}.err").read_text().strip().splitlines()
    return {"command": name, "rc": os.waitstatus_to_exitcode(status), "line": lines[-1] if lines else None,
            "stderr": errs[-1][:2000] if errs else None, "seconds": round(time.monotonic() - t, 1),
            "peak_rss_kb": use.ru_maxrss}


def sha(p: Path) -> str:
    h = hashlib.sha512()
    with open(p, "rb") as f:
        while chunk := f.read(1 << 24):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dir", type=Path)
    ap.add_argument("--case", type=Path)
    ap.add_argument("--members", type=int)
    ap.add_argument("--bin", type=Path)
    ap.add_argument("--lean-bin", type=Path)
    a = ap.parse_args()
    from verity.protocols.proofs.spot.verifier.one_stage import proofs as PF

    if a.case:
        path, audit_s = a.case, None
    else:
        import members_files as MF
        path, audit_s = MF.audited(a.dir, a.members, a.bin, a.lean_bin)
    case = json.loads(path.read_text())
    print(json.dumps({"case": str(path), "audit_s": audit_s, "members": len(case["params"]["proofs"]["members"]),
                      "blobs": len(case["blobs"])}), flush=True)

    exe = PF.v2_executable()
    check = CF.run(case, exe, "check", a.dir / "check")
    print(json.dumps({"command": "check"} | check), flush=True)

    d = a.dir / "stream"
    d.mkdir(parents=True, exist_ok=True)
    params = json.dumps(case["params"], sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False)
    for name, b in (("p", params.encode()), ("c", bytes.fromhex(case["commitment"])),
                    ("e", bytes.fromhex(case["evidence"]))):
        (d / name).write_bytes(b)
    exe = str(exe)
    plan = timed([exe, "plan", "p", "c", "e"], d, "plan")
    lines = (d / "plan.out").read_text().splitlines()
    tables = [x.split()[1] for x in lines if x.startswith("table ")]
    planned = [x.split()[1] for x in lines if x.startswith("blob ")]
    by_sha: dict[str, Path] = {}
    for b in dict.fromkeys(case["blobs"]):
        by_sha.setdefault(sha(Path(b)), Path(b).resolve())
    fed_tables = [by_sha[t] for t in tables if t in by_sha]
    missing = [h for h in planned if h not in by_sha]
    unplanned = sorted(set(by_sha) - set(planned) - set(tables))
    print(json.dumps(plan | {"tables": len(tables), "tables_in_case": len(fed_tables), "planned": len(planned),
                             "planned_missing": len(missing), "case_blobs_unplanned": len(unplanned)}), flush=True)
    if plan["rc"] != 0 or missing:
        raise SystemExit("the plan refused, or names a blob the case lacks: nothing to stream")
    order = fed_tables + [by_sha[h] for h in planned]
    (d / "list").write_text("".join(f"{p}\n" for p in order))
    got = timed([exe, "check-stream", "p", "c", "e", "list"], d, "check-stream")
    print(json.dumps(got | {"fed": len(order), "same_line_as_check": got["line"] == check["line"]}), flush=True)

    i = len(fed_tables) + 1
    swapped = order[:i] + [order[i + 1], order[i]] + order[i + 2:]
    (d / "list-swapped").write_text("".join(f"{p}\n" for p in swapped))
    got = timed([exe, "check-stream", "p", "c", "e", "list-swapped"], d, "swapped")
    print(json.dumps(got | {"swapped": [i, i + 1]}), flush=True)

    changed = d / "changed.bin"
    raw = bytearray(order[i].read_bytes())
    raw[len(raw) // 2] ^= 1
    changed.write_bytes(raw)
    edited = order[:i] + [changed] + order[i + 1:]
    (d / "list-changed").write_text("".join(f"{p}\n" for p in edited))
    got = timed([exe, "check-stream", "p", "c", "e", "list-changed"], d, "changed")
    print(json.dumps(got | {"changed_blob": i, "changed_byte": len(raw) // 2}), flush=True)


if __name__ == "__main__":
    main()
