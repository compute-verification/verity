"""torch-loop-smoke: the network certifier's rehearsal in front of the torch-loop stub (`torch_loop_stub.py`), timed.

    python torch_loop_smoke.py --out DIR [--stub-steps 2] -- [certifier_serve.py rehearse's options, less --upstream
        and --out]

It starts the stub on loopback, then runs `certifier_serve.py rehearse --upstream STUB --out DIR/rehearsal` with the
rest of the arguments in this process, timing each key's making or loading (`load_key`), each strategy's run
(`rehearse_one`: serve, client, check) and each check. Then for each strategy it counts the certifier's
`certified.jsonl` and `clocks.jsonl` and the client's kept `client/certificates.jsonl` and `client/clocks.jsonl`, holds
the kept ones to the deployment's key and session again (`check_certificates`), and holds one kept certificate with a
bit of its signature flipped, which must be refused. `DIR/smoke.json` has the CPUs, the timings, the counts, the
verdicts and the rehearsal's exit, which is this script's. On a network-pool slot, from a clone (r20261025-191204-3c34):

    research run --on vy-nebius-cpu-1 --project verity --source . --cwd clone --mem-gb 16 -- \\
      env CHECK_SLOTS=/workspace/research/locks/network/slots python3 tools/verity/check/slot.py --by network-accounting -- \\
      bash -c 'export PATH=$HOME/.elan/bin:$PATH; uv sync -q --all-packages --extra torch-cpu && \\
        .venv/bin/python benchmarks/network_traces/torch_loop_smoke.py --out $RESEARCH_RUN_DIR/out/h20 -- \\
        --traffic varied --compact 256 --requests 2 --tokens 2 --prompt-range 1:1 --seed 1 --height 20'
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import certifier_serve as cs  # noqa: E402


def lines(path: Path) -> list[dict]:
    return [json.loads(x) for x in path.read_text().splitlines()] if path.exists() else []


def kept_answers(run: Path, deployment: dict) -> dict[int, dict]:
    """The client's kept certificates (`client/`) as the reveal endpoint's answer for each window."""
    answers = {w: {"window": w, "session": deployment["session"], "certificates": [], "clocks": [], "unsigned": []}
               for w in range(deployment["windows"])}
    for d in lines(run / cs.CLIENT / "certificates.jsonl"):
        if "unsigned" in d:
            answers[d["unsigned"][2]]["unsigned"].append(d["unsigned"])
        else:
            answers[d["window"]]["certificates"].append({k: v for k, v in d.items() if k != "session"})
    for d in lines(run / cs.CLIENT / "clocks.jsonl"):
        answers[d["certificate"]["window"]]["clocks"].append(d)
    return answers


def recheck(run: Path) -> dict:
    deployment = json.loads((run / "deployment.json").read_text())
    kept, refused, flipped = defaultdict(int), [], None
    for w, answer in kept_answers(run, deployment).items():
        got, synced, unsigned, no = cs.check_certificates(deployment, w, answer)
        kept["certificates"] += len(got)
        kept["clocks"] += len(synced)
        kept["unsigned"] += len(unsigned)
        refused += no
        if flipped is None and answer["certificates"]:
            bad = json.loads(json.dumps(answer))
            sig = bytearray.fromhex(bad["certificates"][0]["signature"])
            sig[len(sig) // 2] ^= 1
            bad["certificates"][0]["signature"] = sig.hex()
            flipped = cs.check_certificates(deployment, w, bad)[3]
    return {"windows": deployment["windows"], "kept": dict(kept), "refused": refused, "flipped_refused": flipped}


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    ours, rest = (argv[:argv.index("--")], argv[argv.index("--") + 1:]) if "--" in argv else (argv, [])
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--stub-steps", type=int, default=2)
    a = ap.parse_args(ours)
    a.out.mkdir(parents=True, exist_ok=True)
    stub = subprocess.Popen([sys.executable, str(HERE / "torch_loop_stub.py"), "--steps", str(a.stub_steps)],
                            stdout=subprocess.PIPE, text=True)
    timing: dict = {"keys": [], "runs": {}, "checks": []}
    load_key, rehearse_one, check = cs.load_key, cs.rehearse_one, cs.check

    def timed_load_key(directory, height=cs.HEIGHT):
        made, t0 = not (Path(directory) / "public.json").exists(), time.monotonic()
        key = load_key(directory, height)
        timing["keys"].append({"key": Path(directory).name, "made": made, "height": key.params.H,
                               "seconds": round(time.monotonic() - t0, 3)})
        return key

    def timed_rehearse_one(out, strategy, **kw):
        t0 = time.monotonic()
        try:
            return rehearse_one(out, strategy, **kw)
        finally:
            timing["runs"][strategy] = round(time.monotonic() - t0, 3)

    def timed_check(run, challenger):
        t0 = time.monotonic()
        try:
            return check(run, challenger)
        finally:
            timing["checks"].append({"run": Path(run).name, "seconds": round(time.monotonic() - t0, 3)})

    cs.load_key, cs.rehearse_one, cs.check = timed_load_key, timed_rehearse_one, timed_check
    try:
        listening = json.loads(stub.stdout.readline())
        args = cs.parser().parse_args(["rehearse", "--upstream", listening["listening"],
                                       "--out", str(a.out / "rehearsal"), *rest])
        t0 = time.monotonic()
        rc = cs.rehearse(args)
        timing["rehearsal"] = round(time.monotonic() - t0, 3)
    finally:
        stub.terminate()
        stub.wait()
    summary = json.loads((a.out / "rehearsal" / "summary.json").read_text())
    runs = {}
    for s in args.strategies.split(","):
        run = a.out / "rehearsal" / s
        runs[s] = {"verdict": summary["runs"][s]["verdict"], "exit": summary["runs"][s]["exit"],
                   "certifier": {f: len(lines(run / f)) for f in ("certified.jsonl", "clocks.jsonl")},
                   "client": {f: len(lines(run / cs.CLIENT / f)) for f in ("certificates.jsonl", "clocks.jsonl")}
                   | {"peer.json": (run / cs.CLIENT / "peer.json").exists()},
                   "recheck": recheck(run)}
    doc = {"stub": listening, "cpus": len(os.sched_getaffinity(0)), "keygen_workers": cs.KEYGEN_WORKERS,
           "argv": rest, "rc": rc, "timing": timing, "windows": summary["windows"], "varied": summary["varied"],
           "compact": summary["compact"], "lms_height": summary["lms_height"], "runs": runs}
    (a.out / "smoke.json").write_text(json.dumps(doc, indent=1))
    print(json.dumps(doc, indent=1), flush=True)
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
