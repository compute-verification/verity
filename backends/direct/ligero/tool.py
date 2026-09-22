"""Store declaration for the Ligero VU benchmark (research.store README §6.1).

    research run --tool bench_vu --project verity --scratch triton -- python -m backends.direct.ligero.run bench-vu --mode fiat-shamir --zk ...
    research run --tool bench_vu ... -- python backends/direct/ligero/bench_result.py bench-vu --zk --mode interactive ... --verifier /workspace/bin/ligero-verify

The second form is the wrapper that runs bench-vu, verifies every dumped proof with the independent Rust verifier and writes
$RESEARCH_RUN_DIR/result.json; its argv carries the same subcommand and flags, so the same parse applies.

Key params follow run.py's argparse for the `bench-vu` / `gate-vu` / `bench` / `gate` subcommands.  `--device`, `--out` and
`--run-id` never enter the derivation (the device is a condition, `hw.family`; the others name where the answer goes).  `--root`
(the bench-instances dataset) is a path-valued key param: the store replaces it by the content identity of the tree.
Nothing here imports torch or the backend: the declaration must load on the control machine.
"""
from __future__ import annotations

import argparse

from research.store.tool import Tool

SUBCOMMANDS = ("gate", "bench", "gate-vu", "bench-vu")
DEFAULT_ROOT = "fixtures/bench-instances/v1"


def parse(argv: list[str]) -> dict:
    idx = next((i for i, t in enumerate(argv) if t in SUBCOMMANDS), None)
    if idx is None:
        return {"_unparsed": argv}
    ap = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    ap.add_argument("--root", default=DEFAULT_ROOT)      # run.py's default (instances.DEFAULT_ROOT), repo-relative so the store can hash it
    ap.add_argument("--device", default=None)
    ap.add_argument("--target", type=float, default=-128.0)
    ap.add_argument("--rate-log2", type=int, default=2)
    ap.add_argument("--batch", type=int, default=4096)
    ap.add_argument("--out", default=None)
    ap.add_argument("--zk", action="store_true")
    ap.add_argument("--mode", default="interactive")
    ap.add_argument("--hints", default="torch")
    ap.add_argument("--vus", type=int, default=64)
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--run-id", default=None)
    ap.add_argument("--total-vus", type=int, default=None)
    ap.add_argument("--hvzk-label", action="store_true")
    ap.add_argument("--dump-proof", default=None)
    ap.add_argument("--dump-statement", default=None)
    ap.add_argument("--dump-dir", default=None)
    ap.add_argument("--auth", action="store_true")       # authentication = included (what is proved changes: roots + multiproofs)
    ap.add_argument("--auth-cache", default=None)        # where the trees are cached: not what is computed (the bindings pin the object)
    ap.add_argument("--auth-procs", type=int, default=None)
    # bench_result.py (the wrapper that adds the independent Rust verifier and writes result.json) accepts the same argv plus:
    ap.add_argument("--verifier", default=None)         # path of the ligero-verify binary: where the check runs, not what is computed
    ap.add_argument("--threads", type=int, default=None)
    ns, unknown = ap.parse_known_args(argv[idx + 1:])
    out = vars(ns)
    cmd = out["cmd"] = argv[idx]
    for flag, subs in (("vus", ("gate", "gate-vu")), ("dump_proof", ("gate-vu",)), ("dump_statement", ("gate-vu",)), ("reps", ("bench", "bench-vu")),
                       ("run_id", ("bench", "bench-vu")), ("total_vus", ("bench-vu",)), ("hvzk_label", ("bench-vu",)), ("dump_dir", ("bench-vu",)),
                       ("auth", ("bench-vu",)), ("auth_cache", ("bench-vu",)), ("auth_procs", ("bench-vu",))):
        if cmd not in subs:
            out.pop(flag)                       # run.py does not define it for this subcommand; a default would only be noise in the derivation
    if unknown:
        out["_unknown"] = unknown
    return out


BENCH_VU = Tool(
    name="bench_vu", version="1",
    command=["python", "-m", "backends.direct.ligero.run"],
    closure=["backends/direct/ligero/**", "packages/verity/src/verity/**"],
    key_params=["cmd", "mode", "zk", "batch", "total_vus", "vus", "reps", "rate_log2", "target", "hints", "hvzk_label", "root", "auth"],
    key_conditions=["hw.family"],
    param_artifacts=["root"],
    caches=["triton"],
    produces=["bench-result/v1", "run-files/v1"],
    parse=parse,
)


def parse_verify(argv: list[str]) -> dict:
    """`verify_result.py --dir DUMPS --jobs N [--threads T] [--reps R] --verifier BIN [--prover-result R.json]` (lane b-verify-par)."""
    ap = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    ap.add_argument("--dir", default=None)              # the proof dumps: path-valued key param, replaced by the content identity of the tree
    ap.add_argument("--jobs", type=int, default=None)
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--verifier", default=None)         # where the binary is, not what is computed
    ap.add_argument("--prover-result", default=None)    # provenance of prove_wall; the dumps carry the derivation
    ns, unknown = ap.parse_known_args(argv)
    out = vars(ns)
    if unknown:
        out["_unknown"] = unknown
    return out


# The independent verifier's batch timing on a bench-vu proof dump (CPU work: the key condition is the CPU, not the GPU).
LIGERO_VERIFY_BATCH = Tool(
    name="ligero_verify_batch", version="1",
    command=["python", "backends/direct/ligero/verify_result.py"],
    closure=["backends/direct/ligero/verify_result.py", "backends/ligero-verify/**"],
    key_params=["dir", "jobs", "threads", "reps"],
    key_conditions=["hw.cpu"],
    param_artifacts=["dir"],
    produces=["bench-result/v1", "run-files/v1"],
    parse=parse_verify,
)
