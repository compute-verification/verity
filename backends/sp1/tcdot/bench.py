"""The modified-SP1 (TC_DOT chip) bench-result: ``benchmarks/dot_product/vector_run.py --backend sp1-bare`` with this
variant's software identity.

vector_run.py writes stock SP1's toolchain into the fingerprint (Succinct's release ``sp1-gpu-server``, sp1-primitives
6.6.0).  This variant builds every SP1 crate and the GPU server from one fork checkout (``build_fork.sh``: v6.4.0 plus
``sp1-patches/``), so the wrapper asks the host for its fork pin, checks it against ``FORK_HEAD``, swaps those fields
for the fork's and adds ``software.modified``; everything else -- negatives, proving, the per-proof verify, phases,
rates, the contract check -- is vector_run's, unchanged.  Arguments are vector_run's (``--backend sp1-bare`` and a
default ``--backend-name`` are supplied)::

    python3.12 backends/sp1/tcdot/bench.py --host $CARGO_TARGET_DIR/release/verity-tcdot-host \\
        --instances /workspace/bi --reps 3 --out RUN_DIR
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
LABEL = "modified SP1 (TC_DOT chip)"
BACKEND_NAME = f"sp1 tc-dot relation-bare/v2 ({LABEL}: SP1 6.4.0 fork + TC_DOT_BF16 precompile, core STARK, CUDA prover)"
BASE = "succinctlabs/sp1 v6.4.0 (f66b4bff51d0ccff51d152e0f7f66b2ffedf3529)"


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _arg(argv: list[str], name: str) -> str | None:
    for i, a in enumerate(argv):
        if a == name and i + 1 < len(argv):
            return argv[i + 1]
        if a.startswith(name + "="):
            return a.split("=", 1)[1]
    return None


def fork_identity(host: str) -> dict:
    """The fork pin as the host binary reports it, checked against FORK_HEAD, with the patch series' digests."""
    p = subprocess.run([host, "info"], capture_output=True, text=True, env={**os.environ, "SP1_PROVER": "cpu", "RUST_LOG": "error"}, timeout=900)
    info = json.loads([line for line in p.stdout.splitlines() if line.startswith("{")][-1])
    pinned = (HERE / "FORK_HEAD").read_text().strip()
    if p.returncode != 0 or info.get("fork_head") != pinned:
        raise SystemExit(f"bench.py: host {host} reports fork_head {info.get('fork_head')!r}, FORK_HEAD pins {pinned} (rc {p.returncode})")
    server = Path(os.environ.get("HOME", "~")).expanduser() / ".sp1" / "bin" / "sp1-gpu-server"
    return {
        "label": LABEL,
        "base": BASE,
        "fork_head": pinned,
        "fork_tree": (HERE / "FORK_TREE").read_text().strip(),
        "patches": {p.name: _sha256(p) for p in sorted((HERE / "sp1-patches").glob("*.patch"))},
        "chips": "TC_DOT (fp8 E4M3 operands through the Ampere BF16 pipeline; ~/projects/sp1 tc-dot-precompile, unused here) and "
                 "TC_DOT_BF16 (one tc-ampere-bf16 m16n8k16 step per call: K = 16 BF16 products, groups [8, 8], 25-bit adder, floor -132, "
                 "truncation; operand contract = finite operands and accumulator, both GroupSum outputs zero or fp32-normal)",
        "guest_routing": "steps outside the chip's contract run tc::tc_dot in the guest; a VU with >= vu_software_threshold such steps runs "
                         "bare::check_one whole (the stock kernel)",
        "proof_modes": "core only: the fork's recursion vk map does not contain the TC_DOT chips",
        "prover_sharding": "patch 0006: the shape checker charges a deferred precompile's memory accesses to its precompile shard, not "
                           "the CPU shard; local_gpu_opts honours ELEMENT_THRESHOLD (prover scheduling only: no AIR or verifier change)",
        "prover_env": {k: os.environ[k] for k in ("ELEMENT_THRESHOLD", "VERITY_TCDOT_VU_SOFTWARE_THRESHOLD") if k in os.environ},
        "gpu_server": {"build": "cargo install --path sp1-gpu/crates/server from the fork (CUDA_ARCHS=80)", "path": str(server),
                       "sha256": _sha256(server) if server.is_file() else None},
        "host_info": {k: info.get(k) for k in ("elf_sha256", "vk_hash", "statement", "formats")},
        "claim": "a `modified` backend (veritor claim register CL10): a separately labelled research track, not stock SP1",
    }


def main(argv: list[str]) -> int:
    spec = importlib.util.spec_from_file_location("vector_run", REPO / "benchmarks" / "dot_product" / "vector_run.py")
    vector_run = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(vector_run)

    host = _arg(argv, "--host")
    if not host:
        raise SystemExit("bench.py: --host (the verity-tcdot-host binary) is required")
    modified = fork_identity(host)
    contract = vector_run.contract
    stock_fingerprint = contract.fingerprint

    def fingerprint(**kw):
        software = json.loads(json.dumps(kw["software"]))
        software["backend"]["version"] = f"6.4.0 fork @ {modified['fork_head'][:12]} ({LABEL})"
        fork = f"6.4.0 (fork @ {modified['fork_head'][:12]})"
        software["toolchain"].update({"sp1-sdk": fork, "sp1-zkvm": fork, "sp1-primitives": fork, "sp1-core-executor": fork,
                                      "sp1-gpu-server": f"built from the fork @ {modified['fork_head'][:12]} (CUDA_ARCHS=80)"})
        software["modified"] = modified
        security = dict(kw["security"])
        security["source"] = ("SP1's own accounting, unchanged by the TC_DOT patches: the fork's crates/primitives/src/fri_params.rs (v6.4.0) "
                              "SP1_TARGET_BITS_OF_SECURITY = 100 per STARK proof (CORE_LOG_BLOWUP 2, SP1_PROOF_OF_WORK_BITS 16; KoalaBear^4)")
        return stock_fingerprint(**{**kw, "software": software, "security": security, "label": LABEL})

    contract.fingerprint = fingerprint
    args = list(argv)
    if "--backend" not in args:
        args = ["--backend", "sp1-bare", *args]
    if _arg(args, "--backend-name") is None:
        args += ["--backend-name", BACKEND_NAME]
    if _arg(args, "--mode") not in (None, "core"):
        raise SystemExit("bench.py: core proofs only (the fork's recursion vk map lacks the TC_DOT chips)")
    return vector_run.main(args)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
