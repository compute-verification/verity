"""The store binding's decision at a hello at a store's size, timed through both verifiers: a synthetic pinned model
whose arena is `--weights` chunks (random digests) then zeros to `--arena-blocks`, `--engines` arenas of it, then
`--host` host chunks, under one payload key, `store.Binding` over it as it stands past what it reads of the files, and
the Lean verifier (`remote-audit-verifier`) bound to it (`verifier.bind_lean`). Each phase is one hello's
decision, timed as a whole, with each verifier's own time (store.py's `recompute_s`, the Lean's `lean_s`) and CPU
seconds (store.py's processes, the Lean's threads):

~~~text
sequential  one after the other: `Binding.hold` (its root in `--workers` processes), then the Lean's `hold` (its root on
            `--workers` threads), on a hello whose root is 64 zero bytes: both refuse, at the root they computed
concurrent  `verifier.hold_store`, the two at once on the same cores, on the same hello: both refuse
bound       `verifier.hold_store` on a hello carrying that root: both bind
split       `verifier.hold_store` on the bound hello with the cores split: store.py's root in round(--python-share *
            --workers) processes, a second Lean process's on the rest of `--workers` as threads
~~~

A phase may be listed again, so that on a shared machine the orders bracket a drift in its load. A phase whose verifiers
disagree, or that decides otherwise than its row says, fails the bench. Writes OUT/binding_bench.json after each phase
(the cores, the load average before and after each); exits 1 unless every phase decided as it should.

    python binding_bench.py --out OUT [--engines 8 --arena-blocks 16250000 --weights 4000000 --host 0 --workers N]
                            [--phases sequential,concurrent,bound,split] [--python-share 0.59]

The defaults are a whole node's 130M chunks: eight arenas of an 8B BF16 model's 4M weights chunks.

    uv sync -q --all-packages --extra torch-cpu && uv run python benchmarks/remote_session/binding_bench.py --out OUT
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import resource
import sys
import tempfile
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1]))

from verity.core.protocols.remote_bandwidth import store  # noqa: E402

import lean_verifier as lv  # noqa: E402
import verifier as V  # noqa: E402

ROLE = "SYNTHETIC_BF16"
NODE = b"binding-bench-node"
PHASES = ("sequential", "concurrent", "bound", "split")
PYTHON_SHARE = 0.59
"""store.py's share of the cores in `split`: its root's time over both roots' on 11 cores each, 432 s and 306 s at
130M chunks."""


def pin_data() -> bytes:
    files = {n: hashlib.sha512(n.encode()).hexdigest() for n in ("config.json", "model.safetensors")}
    return json.dumps({"name": ROLE, "checkpoint": files, "checkpoint_sha512": store.checkpoint_sha512(files)}).encode()


def lean_cpu(ln: lv.Lean) -> float:
    """The Lean process's CPU seconds so far, all its threads (`/proc/PID/stat`'s utime and stime)."""
    fields = Path(f"/proc/{ln.proc.pid}/stat").read_text().rpartition(")")[2].split()
    return (int(fields[11]) + int(fields[12])) / os.sysconf("SC_CLK_TCK")


def python_cpu() -> float:
    """This process's CPU seconds so far and its reaped children's: store.py's worker processes once their pool ends."""
    me, kids = resource.getrusage(resource.RUSAGE_SELF), resource.getrusage(resource.RUSAGE_CHILDREN)
    return me.ru_utime + me.ru_stime + kids.ru_utime + kids.ru_stime


def binding(weights: int, engines: int, arena_blocks: int, workers: int, digests: Path) -> store.Binding:
    """A `store.Binding` of the synthetic pin whose arena is `weights` chunks of the digests in the file `digests`, as
    `Binding` leaves it once it has read the files."""
    b = store.Binding.__new__(store.Binding)
    b.pin_data, b.pin_path, b.role = pin_data(), Path("pin.json"), ROLE
    b.pin = store.model_pin(b.pin_data, str(b.pin_path), ROLE)
    b.arena, b.engines, b.arena_blocks, b.workers = store.Arena("", (), weights), engines, arena_blocks, workers
    b.digests, b.store, b.state, b.log = digests, None, {}, lambda _m: None
    return b


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--engines", type=int, default=8)
    ap.add_argument("--arena-blocks", type=int, default=16_250_000)
    ap.add_argument("--weights", type=int, default=4_000_000)
    ap.add_argument("--host", type=int, default=0)
    ap.add_argument("--workers", type=int, default=len(os.sched_getaffinity(0)))
    ap.add_argument("--phases", default=",".join(PHASES),
                    help=f"a comma-separated list of {PHASES}, run in that order, any of them again")
    ap.add_argument("--python-share", type=float, default=PYTHON_SHARE)
    a = ap.parse_args()
    phases = a.phases.split(",")
    if any(p not in PHASES for p in phases) or phases[0] not in ("sequential", "concurrent"):
        ap.error(f"--phases lists {PHASES}, the first sequential or concurrent (the others take its root)")
    a.out.mkdir(parents=True, exist_ok=True)
    blocks = a.engines * a.arena_blocks + a.host
    out: dict = {"engines": a.engines, "arena_blocks": a.arena_blocks, "weights_blocks": a.weights, "host_blocks": a.host,
                 "blocks": blocks, "workers": a.workers, "chunk_log2": store.CHUNK_LOG2,
                 "cpus": len(os.sched_getaffinity(0)), "cpu_count": os.cpu_count(), "nice": os.nice(0), "phases": []}

    def write() -> None:
        (a.out / "binding_bench.json").write_text(json.dumps(out, indent=1) + "\n")
        print(json.dumps(out), flush=True)

    key = hashlib.sha256(b"binding-bench key").digest()
    zero = bytes(64)
    ok = True
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "weights.sha512rows"
        path.write_bytes(hashlib.shake_256(b"binding-bench weights").digest(64 * a.weights))
        b = binding(a.weights, a.engines, a.arena_blocks, a.workers, path)
        lean = lv.Lean(lv.build(), lambda _msg: None)
        t0 = time.monotonic()
        bound = V.bind_lean(lean, b)
        out["lean_bind_s"] = round(time.monotonic() - t0, 3)
        out["lean_bind"] = {k: v for k, v in bound.items() if k != "pin"}
        write()

        def hello(ln: lv.Lean, bd: dict, bb: store.Binding, root: bytes) -> dict:
            """`verifier.hold_store` of a hello carrying `root`, timed: bound, refused or disagreement, with each
            verifier's time."""
            load, t, c, lc = os.getloadavg(), time.monotonic(), python_cpu(), lean_cpu(ln)
            try:
                V.hold_store(ln, bd, bb, NODE, blocks, root, key, None)
                decided = "bound"
            except store.Refused as e:
                decided = "disagreement" if str(e).startswith("disagreement") else "refused"
                if decided == "disagreement":
                    print(f"binding_bench: {e}", flush=True)
            return {"hello_s": round(time.monotonic() - t, 3), "decided": decided,
                    "python_s": round(bb.state.get("recompute_s", 0.0), 3), "lean_s": round(bb.state["lean_s"], 3),
                    "python_cpu_s": round(python_cpu() - c, 1), "lean_cpu_s": round(lean_cpu(ln) - lc, 1),
                    "expected_root": bb.state.get("expected_root"), "load_before": load, "load_after": os.getloadavg()}

        def sequential(root: bytes) -> dict:
            """The sequential decision of a hello carrying `root`: `Binding.hold`, then the Lean's `hold`, timed."""
            load, t, c, lc = os.getloadavg(), time.monotonic(), python_cpu(), lean_cpu(lean)
            try:
                regions = b.hold(NODE, blocks, root, key, None)
                mine = {"regions": [list(r) for r in regions], "expected_root": b.state["expected_root"]}
            except store.Refused as e:
                mine = {"refused": str(e), "expected_root": b.state.get("expected_root")}
            t1 = time.monotonic()
            got = lean.call({"op": "hold", "node": NODE.hex(), "blocks": blocks, "payload_root": root.hex(),
                             "key": key.hex(), "layout": None})
            t2 = time.monotonic()
            decided = "disagreement" if got != mine else "refused" if "refused" in got else "bound"
            return {"hello_s": round(t2 - t, 3), "decided": decided, "python_s": round(t1 - t, 3),
                    "lean_s": round(t2 - t1, 3), "python_cpu_s": round(python_cpu() - c, 1),
                    "lean_cpu_s": round(lean_cpu(lean) - lc, 1), "expected_root": mine["expected_root"],
                    "load_before": load, "load_after": os.getloadavg()}

        def split(root: bytes) -> dict:
            """`hello` with store.py's root in `--python-share` of the cores and a second Lean's on the rest."""
            python = max(1, min(a.workers - 1, round(a.python_share * a.workers)))
            b2 = binding(a.weights, a.engines, a.arena_blocks, a.workers - python, path)
            lean2 = lv.Lean(lv.build(), lambda _msg: None)
            try:
                bound2 = V.bind_lean(lean2, b2)
                b2.workers = python
                return hello(lean2, bound2, b2, root) | {"python_workers": python, "lean_workers": a.workers - python}
            finally:
                lean2.close()

        root = None
        for phase in phases:
            if phase in ("sequential", "concurrent"):
                r = sequential(zero) if phase == "sequential" else hello(lean, bound, b, zero)
                want, root = "refused", root or bytes.fromhex(r["expected_root"])
            else:
                r, want = (hello(lean, bound, b, root) if phase == "bound" else split(root)), "bound"
            r["ok"] = r["decided"] == want and r["expected_root"] == root.hex()
            ok = ok and r["ok"]
            out["phases"].append({"phase": phase} | r)
            write()
        lean.close()
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
