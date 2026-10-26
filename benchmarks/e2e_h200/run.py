"""The H200 walking skeleton (CMP-96): steps of the frozen MQA shape (or a small one) from the weights' store through
decode-to-staging, every op's output committed per step, drawn units replayed bit for bit, each leg run or named, and the
replay bundle and block record written to ``--out``.

    python benchmarks/e2e_h200/run.py --out OUT [--shape tiny|small|slice|h200] [--seed test|demo|TEXT] [--batch B] [--steps T]
        [--prompt-bytes P] [--device cpu|cuda] [--store generator|p2-encode|p2:PATH] [--salt HEX] [--stage-lib DIR]
        [--commit reference|host|device] [--draw K] [--tamper OP] [--run-id ID] [--seed S]

Exit 0 when every drawn unit replays to its committed word, 3 when one doesn't (``--tamper OP`` flips bit 0 of OP's committed
words at every step: replay must find it), 4 when a leg's fixture check fails.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from e2e_h200 import legs as L  # noqa: E402
from e2e_h200 import record as REC  # noqa: E402
from e2e_h200.commit import Committer, fixture_secrets  # noqa: E402
from e2e_h200.driver import Driver  # noqa: E402
from e2e_h200.shape import SHAPES  # noqa: E402
from e2e_h200.store import GeneratorStore, P2Store  # noqa: E402
from e2e_h200.units import Replayer  # noqa: E402
from e2e_h200.weights import Weights  # noqa: E402
from verity_torch_loop import weights as SW  # noqa: E402

#: the weights' registration digest under the test salt key is computed for streams up to this size (about 1 s a MiB)
DIGEST_BYTES = 1 << 26


def prompts_for(run_id: str, batch: int, p: int) -> np.ndarray:
    h = hashlib.sha512(b"e2e-h200/prompts-fixture\0" + run_id.encode()).digest()
    rng = np.random.Generator(np.random.PCG64(int.from_bytes(h[:16], "big")))
    return rng.integers(32, 127, size=(batch, p), dtype=np.int64)


def make_store(a, weights: Weights, device: str):
    salt = bytes.fromhex(a.salt) if a.salt else hashlib.sha512(b"e2e-h200/salt-fixture\0" + a.run_id.encode()).digest()[:24]
    if a.store == "generator":
        return GeneratorStore(weights)
    if a.store == "p2-encode":
        return P2Store.encode(weights, salt, workers=a.workers)
    if a.store.startswith("p2:"):
        return P2Store.load(weights, salt, a.store[3:], device if a.stage_lib else None, a.stage_lib)
    raise SystemExit(f"--store is generator, p2-encode or p2:PATH, not {a.store!r}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--shape", default="small", choices=sorted(SHAPES))
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--steps", type=int, default=4)
    ap.add_argument("--prompt-bytes", type=int, default=2)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--store", default="generator")
    ap.add_argument("--salt", default="")
    ap.add_argument("--stage-lib", default="")
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--commit", default="host", choices=("reference", "host", "device"))
    ap.add_argument("--draw", type=int, default=4, help="units drawn per family of the replayed step")
    ap.add_argument("--replay-step", type=int, default=-1)
    ap.add_argument("--pous-reads", type=int, default=16)
    ap.add_argument("--tamper", default="")
    ap.add_argument("--run-id", default="e2e-h200-fixture")
    ap.add_argument("--seed", default="test", help="test (ALLP-IFACE's TEST_SEED), demo (DEMO_SEED) or a text seed")
    a = ap.parse_args(argv)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    shape = SHAPES[a.shape]
    seed = {"test": SW.TEST_SEED, "demo": SW.DEMO_SEED}.get(a.seed, a.seed.encode())
    weights = Weights(shape, seed)
    t0 = time.perf_counter()
    store = make_store(a, weights, a.device)
    t_store = time.perf_counter() - t0
    ctx, key = fixture_secrets(a.run_id)
    committer = Committer(a.commit, ctx, key)
    prompts = prompts_for(a.run_id, a.batch, a.prompt_bytes)

    def drive():
        d = Driver(shape, store, a.batch, committer, a.device, tamper=a.tamper or None)
        return d, d.run(prompts, a.steps)

    driver, results = drive()
    for r in results:
        print(json.dumps({"step": r.step, "root": r.root.hex()[:32], "value_bytes": r.value_bytes,
                          "seconds": {k: round(v, 3) for k, v in r.seconds.items()}}), flush=True)
    step = a.replay_step % len(results)
    t0 = time.perf_counter()
    units = Replayer(shape, weights, driver.steps, [r.tokens_in for r in results]).draw(step, a.draw, a.run_id.encode())
    t_replay = time.perf_counter() - t0
    REC.write_bundle(out, units, results[step].root, step)
    replay = REC.replay_summary(units) | {"step": step, "seconds": t_replay,
                                          "draw": f"fixture coin, {a.draw} a family (PRF-170's plan replaces it)"}
    outputs = np.stack([r.tokens_out for r in results], 1)
    legs = {
        "integrity": L._entry("integrity", "fixture", replayed=replay["drawn"], mismatches=replay["mismatches"],
                              stands_in_for="C-Flock --zk proving the drawn units and Lean's ACCEPT"),
        "pous": L.pous_spot_reads(store, a.pous_reads, a.run_id.encode()),
        "bandwidth": L.not_run("bandwidth", "memory-accounting's audit reads run against the demo card's store"),
        "certifier": L.certifier(prompts, outputs),
        "wipe": L.wipe_reload(store, lambda: drive()[1][-1].root, results[-1].root),
        "pouw": L.not_run("pouw", "ALLP-H1LOOP's H1 tiles over the step's INT8 linears, with ALLP-SALT's live salt"),
    }
    doc = REC.write_record(
        out, run_id=a.run_id, shape=shape, batch=a.batch, device=a.device,
        weights={**weights.manifest(), "registration_digest_test_key":
                 weights.registration_digest() if weights.nbytes <= DIGEST_BYTES else None},
        store={**store.describe(), "seconds": t_store,
               "stands_in_for": None if store.kind == "p2-v4" else "MEM-69's P2-encoded store and its decode call"},
        commit={"backend": a.commit, "context": "fixture", "layout": "table-major: a table an op, a row a slot",
                "stands_in_for": "kernels' commit tree with one subtree per slot (CMP-95 item 3)"},
        results=results, prompts=prompts, replay=replay, legs=legs,
        extra={"tamper": a.tamper or None, "attention": "Definition numpy kernels on the CPU (KP2-ATTN pending)",
               "gemm": "torch._int_mm + torch epilogue" if a.device.startswith("cuda") else "int64 matmul + torch epilogue"})
    print(json.dumps({"replay": {k: replay[k] for k in ("drawn", "mismatches", "families")},
                      "legs": {k: v["status"] for k, v in legs.items()}}), flush=True)
    if replay["mismatches"]:
        return 3
    if legs["pous"].get("mismatched_blocks") or legs["wipe"].get("store_digest_equal") is False \
            or legs["wipe"].get("step_root_equal") is False:
        return 4
    return 0 if doc else 1


if __name__ == "__main__":
    sys.exit(main())
