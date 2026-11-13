"""`members_files.sh`'s input: one per-member case as the executable's `check` takes it, in DIR: params.json (the
parameters, canonical), commitment.bin (the registration record) and blobs.txt (the blobs' paths, a line each).

  python members_files.py DIR --case CASE
  python members_files.py DIR --bin FLOCK_CIRCUIT --lean FLOCK_VERIFY --firewall FLOCK_FIREWALL

With --case, the case is one a driver wrote (an audit's `window/case-v2-members.json`: `{"blobs", "commitment",
"evidence", "params"}`). Without it, the case is the one the two-member audit of `benchmarks/one_stage`'s test writes
(`test_audit_two_members`: a window of two templates of one proof block each, each member proved by Flock's prover
behind the ZK server and verified by the Lean verifier, its sessions on the window's challenger session), run here
into DIR/audit on the three binaries. Either way the compiled check must accept the case (`members.case_check`), so the
pass `members-files-time` times is an accepted audit's; it refuses otherwise.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BINARIES = ("bin", "lean", "firewall")


def audited(out: Path, binaries: dict[str, Path]) -> Path:
    """The per-member case of the test's two-member audit, run into ``out``/audit on ``binaries``."""
    sys.path.insert(0, str(ROOT / "benchmarks" / "one_stage" / "tests"))
    import test_audit_two_members as T

    out.mkdir(parents=True, exist_ok=True)
    rc, row, where, said = T._audit(out, T._window(out / "window"), binaries, "--deployed-key",
                                    more={"FLOCK_CHALLENGER": "1"})
    case = where / "window" / "case-v2-members.json"
    if rc != 0 or row.get("ok") is not True or not case.is_file():
        raise SystemExit(f"the two-member audit wrote no per-member case (exit {rc}, ok {row.get('ok')}):\n{said}")
    return case


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dir", type=Path)
    ap.add_argument("--case", type=Path, help="a driver's per-member case; without it, the two-member audit's")
    for name in BINARIES:
        ap.add_argument(f"--{name}", type=Path, help="the two-member audit's (test_audit_two_members' `binaries`)")
    a = ap.parse_args()
    binaries = {name: getattr(a, name) for name in BINARIES}
    if a.case is None and not all(binaries.values()):
        ap.error("without --case, the two-member audit needs --bin, --lean and --firewall")
    from verity.protocols.spot.verifier.one_stage import members as M

    case = json.loads((a.case or audited(a.dir, binaries)).read_text())
    verdict, why = M.case_check(case)
    if verdict != "accept":
        raise SystemExit(f"the compiled check does not accept the case ({verdict}: {why}), so its pass is no accepted "
                         f"audit's and nothing is timed")
    a.dir.mkdir(parents=True, exist_ok=True)
    params = json.dumps(case["params"], sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False)
    (a.dir / "params.json").write_text(params)
    (a.dir / "commitment.bin").write_bytes(bytes.fromhex(case["commitment"]))
    (a.dir / "blobs.txt").write_text("".join(f"{b}\n" for b in case["blobs"]))
    print(json.dumps({"members": len(case["params"]["proofs"]["members"]), "blobs": len(case["blobs"]),
                      "blob_bytes": sum(Path(b).stat().st_size for b in case["blobs"]), "verdict": verdict}))


if __name__ == "__main__":
    main()
