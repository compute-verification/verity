"""NET-190's checks of one served run's serve dir, on its node, after the serve step (odserve_check.sh runs it in the run's
tree): the network certifier's records and every request the endpoint served, from the front's kept fetch.

    odserve_check.py SERVE_DIR PRIVATE_DIR

1. The kept fetch: exactly one under SERVE_DIR/private/certifier (`certifier_piece.kept_run`), unpacked in PRIVATE_DIR.
2. The gate's own check passed it (`certifier_piece.rehearsed`: its end.json's exit 0 and accepted).
3. `warden_serve.check` again, here, on the fetch's witness alone: every window of the three records has a certificate
   that verifies, every window of the wire's two a registration whose root is its rows', the audit accepts, and each
   connection's client got exactly its egress cells; the result held equal to the gate's own check.json.
4. Every request in the endpoint's requests.jsonl is a certified one (`certifier_piece.bind` over `warden_chain.io`:
   its user one the network certifier certified, with the same prompt and streamed ids), each certified request is one
   the endpoint served, and each names the network certifier's release run (deployment.json's ``run_id``), read by the
   tree's own `certifier_piece.release_run`.

Prints one JSON line, the public summary: verdicts, the link-windows' counts and the check's own counts of records,
never a count of requests or windows served; the rest goes to PRIVATE_DIR/check.json. Exit 0 only when every check
passed.
"""
from __future__ import annotations

import json
import sys
import tarfile
from pathlib import Path

TREE = Path.cwd()
sys.path[:0] = [str(TREE / "benchmarks" / "network_traces")]
import certifier_piece as CP  # noqa: E402
import warden_chain as WC  # noqa: E402
import warden_serve as WS  # noqa: E402


def lines(p: Path) -> list[dict]:
    return [json.loads(x) for x in p.read_text().splitlines() if x.strip()] if p.is_file() else []


def unpacked(fetch: Path, into: Path) -> tuple[Path, Path]:
    """The fetch's network certifier run and its witness directory, extracted in ``into``."""
    with tarfile.open(fetch, "r:gz") as tar:
        tar.extractall(into, filter="data")
    hit = next(into.rglob("certified.jsonl"), None)
    if hit is None:
        raise CP.NotRun(f"{fetch}: no network certifier's run (a certified.jsonl) in the fetch")
    return (hit.parent.parent, hit.parent) if hit.parent.name == "witness" else (hit.parent, hit.parent)


def checks(serve: Path, private: Path) -> tuple[dict, dict]:
    run = CP.kept_run(serve)
    root, wit = unpacked(serve / CP.PRIVATE / run / "fetch.tar.gz", private / "fetch")
    CP.rehearsed(root)
    gate = json.loads((wit / "check.json").read_text()) if (wit / "check.json").is_file() else None
    mine = WS.check(wit)
    (private / "warden-check.json").write_text(json.dumps(mine, indent=1))
    agrees = gate is not None and (gate["accepted"], gate["summary"]) == (mine["accepted"], mine["summary"])
    windows = mine["windows"]
    dep, _ = WC.deployment(root)
    io = WC.io(root)
    reqs = lines(serve / "requests.jsonl")
    try:
        bound = CP.bind(reqs, io)
        bind_why = None
    except CP.Rejected as e:
        bound, bind_why = {}, str(e)
    released = {str(CP.release_run(r)) for r in reqs}
    unserved = sorted({r["user"] for r in io["records"]} - set(bound)) if bound else []
    cells = mine["summary"]["cell_rows"]
    public = {
        "schema": "odserve/check/v0",
        "gate_rehearsal": "accepted",
        "records": {"accepted": mine["accepted"], "problems": len(mine["problems"]),
                    "link_windows_certified": sum(bool(x["certified"]) for x in windows), "link_windows": len(windows),
                    "records": sorted({x["record"] for x in windows}),
                    "cell_rows_registered": cells["registered"] == cells["of"],
                    "gate_check_agrees": agrees},
        "io_read_whole": bool(io["accepted"]),
        "every_request_certified": bool(reqs) and bind_why is None,
        "every_certified_request_served": bool(reqs) and bind_why is None and not unserved,
        "every_request_released_by_the_network_certifier": bool(reqs) and released == {str(dep["run_id"])},
    }
    public["passed"] = (public["records"]["accepted"] and public["records"]["gate_check_agrees"]
                        and public["records"]["cell_rows_registered"]
                        and public["records"]["link_windows_certified"] == public["records"]["link_windows"] > 0
                        and public["io_read_whole"] and public["every_request_certified"]
                        and public["every_certified_request_served"] and public["every_request_released_by_the_network_certifier"])
    full = public | {"run": run, "problems": mine["problems"], "bind_why": bind_why, "unserved": unserved,
                     "released": sorted(released), "certifier_run_id": dep["run_id"], "requests": len(reqs),
                     "certified": len(io["records"]), "io_violations": io["violations"][:20]}
    return public, full


def main(argv: list[str]) -> int:
    serve, private = map(Path, argv[1:3])
    try:
        public, full = checks(serve, private)
    except (CP.NotRun, CP.Rejected, SystemExit, OSError, ValueError, KeyError) as e:
        (private / "check.json").write_text(json.dumps({"refused": f"{type(e).__name__}: {e}"}, indent=1))
        print(json.dumps({"schema": "odserve/check/v0", "passed": False,
                          "refused": f"{type(e).__name__} (its reason in the serve dir's private/odserve/check.json)"}))
        return 1
    (private / "check.json").write_text(json.dumps(full, indent=1))
    print(json.dumps(public))
    return 0 if public["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
