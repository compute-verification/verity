"""prove.py BINARY DIR N [--zk] [--runs R] [--selftest [CASE ...]]: one loopback session of DIR's statement sampling all N
instances (``serve`` on 127.0.0.1, then ``prove``; ``--zk`` the served path's --zk session), with the prover's peak memory (its
rusage) and the verifier's (VmHWM), or the selftest's negative cases (``--selftest``, `class_statement.prove`: every case, or
each CASE in its own process). A form the
soundness proof does not cover is refused by the verifier (its verdict ``refused: ...``); the verdict it would have given is
the record's ``without_refusal``, written here as ``verdict_without_refusal``. Writes DIR/prove[-zk].json or DIR/selftest.json."""
from __future__ import annotations

import json
import os
import resource
import subprocess
import sys
import time
from pathlib import Path

from verity_flock import circuit as FC
from verity_flock import class_statement as CS

KEYS = ("vus_per_block", "k_log", "nbl", "dense_m", "witness_s", "prove_s", "prove_total_s", "proof_bytes", "wire_sent", "wire_recv")


def session(binary: str, d: Path, n: int, zk: bool, runs: int) -> dict:
    circ, inst, pub = d / "circuit.txt", d / f"inst-{n}.bin", d / f"pub-{n}.bin"
    pin, tag = FC.digest(circ.read_text()), "-zk" if zk else ""
    z = ["--zk"] if zk else []
    log = d / f"serve{tag}.log"
    srv = subprocess.Popen([binary, "serve", *z, "--listen", "127.0.0.1:0", "--out", str(d / f"sessions{tag}"), "--instances", str(pub),
                            "--circuit", str(circ), "--pin", pin, "--operator", "loopback", "--draw", f"subset:{n}"],
                           stdout=open(log, "w"), stderr=subprocess.STDOUT)
    addr = None
    while addr is None and srv.poll() is None:
        addr = next((ln.split()[1] for ln in log.read_text().splitlines() if ln.startswith("SERVING ")), None)
        time.sleep(0.2)
    if addr is None:
        return {"accepted": False, "why": "serve exited: " + log.read_text()[-400:]}
    t0 = time.monotonic()
    with open(d / f"prove{tag}.out", "w") as fo, open(d / f"prove{tag}.err", "w") as fe:
        rc = subprocess.run([binary, "prove", *z, "--verifier", addr, "--instances", str(inst), "--circuit", str(circ),
                             "--runs", str(runs), "--warm", "0"], stdout=fo, stderr=fe).returncode
    wall = time.monotonic() - t0
    prover_gb = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss / 2**20
    verifier_gb = CS._vm_hwm(srv.pid)
    srv.kill()
    srv.wait()
    live = [json.loads(ln[5:]) for ln in (d / f"prove{tag}.out").read_text().splitlines() if ln.startswith("LIVE")]
    if rc or not live:
        return {"accepted": False, "why": f"prove exited {rc}: " + (d / f"prove{tag}.err").read_text()[-400:]}
    v = live[-1]
    verdict = v.get("verdict") or {}
    w = v.get("without_refusal") or verdict.get("without_refusal") or {}
    return {"zk": zk, "accepted": all(x.get("accepted") is True for x in live), "why": v.get("why"),
            "verdict_without_refusal": {k: w.get(k) for k in ("accepted", "why") if k in w},
            "verify_s": verdict.get("verify_s"), "prover_peak_gb": round(prover_gb, 2), "verifier_peak_gb": verifier_gb,
            "wall_s": round(wall, 2), "runs": [{k: x.get(k) for k in KEYS} for x in live]} | {k: v.get(k) for k in KEYS}


def selftest(binary: str, d: Path, n: int, cases: list[str]) -> dict:
    """Every case in one selftest process, or each of ``cases`` in its own (``--only``), run in DIR: flock-circuit reads
    ``circuit-forged.txt`` and ``tables`` from its working directory, and skips ``tail_swapped`` without the first."""
    os.chdir(d)
    if not cases:
        return CS.prove(binary, d, n, 0, selftest=True, timeout=7200)
    each = {c: CS.prove(binary, d, n, 0, selftest=c, timeout=7200) for c in cases}
    failed = [c for c, r in each.items() if r.get("all_pass") is not True]
    return {"all_pass": not failed, "cases": len(cases), "failed": failed, "each": each,
            "wall_s": round(sum(r.get("wall_s") or 0 for r in each.values()), 3)}


def main(argv: list[str]) -> int:
    binary, d, n = argv[0], Path(argv[1]).resolve(), int(argv[2])
    if "/" in binary:
        binary = str(Path(binary).resolve())
    if "--selftest" in argv:
        res, name = selftest(binary, d, n, [a for a in argv[argv.index("--selftest") + 1:] if not a.startswith("--")]), "selftest.json"
    else:
        zk = "--zk" in argv
        runs = int(argv[argv.index("--runs") + 1]) if "--runs" in argv else 1
        res, name = session(binary, d, n, zk, runs), f"prove{'-zk' if zk else ''}.json"
    (d / name).write_text(json.dumps(res, indent=1, sort_keys=True))
    print(json.dumps({k: v for k, v in res.items() if k != "runs"}))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
