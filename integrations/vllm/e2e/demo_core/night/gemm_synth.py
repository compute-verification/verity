#!/usr/bin/env python3
"""[NIGHT 2026-09-12, E8/G4] SmolLM2-135M GEMM-coordinate statements (K = 576 and K = 1536) as SP1 proof batches for the
per-kind proving-seconds measurement -- SYNTHETIC activation rows, REAL weight rows.

Reuses lane PB's ``zk/bench/pb_gemm_batches.py`` (``Builder``, ``coordinate_program`` = IF's ``GemmCoordinate<K>`` lowered exactly
as ``veritor.interface.lowering.lower()`` lowers it, family ``veritor.tc-ampere-bf16@1`` -- approved in guest ``9fd00e60``) with one
substitution: the activation row ``x`` is drawn from a seeded RNG (bf16 words of small magnitude) instead of a captured fixture,
and the claimed output word is the EXACT evaluation of the kind program under the Python gate semantics (the same
``semantics.evaluate`` the transparent checker runs), so every batch is honest by construction and the checker accepts it before
it is written.  The weight matrix is the pinned checkpoint's (HF snapshot ``93efa2f0``): ``layers.0.self_attn.qkv_proj`` (N = 960,
K = 576) and ``layers.0.mlp.down_proj`` (N = 576, K = 1536).

Labelled differences from a record-backed statement: word-per-leaf trees (frame v3) for x / W / out instead of the record's
``interior-layout/chunks-4096/v2`` chunk leaves and the weights tree; the proving cost is dominated by the K-step tc_dot16 chain and
the per-word openings, which are the same shape either way.  n = 1 and n = 64 coordinates per statement (R1/R2's strategy at n = 64).

    PYTHONPATH=vllm-poc:src:zk/bench .venv/bin/python e2e/demo_core/night/gemm_synth.py --out /tmp/night-e8/batches/gemm --sizes 1,64
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

HERE = Path(__file__).resolve()
REPO = HERE.parents[4]
for p in (REPO / "vllm-poc", REPO / "src", REPO / "zk" / "bench"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import pb_gemm_batches as PB  # noqa: E402
from pb_common import write_batch, write_manifest  # noqa: E402
from veritor.protocol.proofs.statement import LOCAL  # noqa: E402

CHECKPOINT = Path.home() / ".cache/huggingface/hub/models--HuggingFaceTB--SmolLM2-135M/snapshots/93efa2f097d58c2a74874c7e644dbc9b0cee75a2"
CALLS = {"qkv_proj": ("layers.0.self_attn.qkv_proj.weight", 576), "down_proj": ("layers.0.mlp.down_proj.weight", 1536)}


def evaluate_program(gate_set, program, inputs: list[int]) -> int:
    """The kind program's outgoing word under the Python gate semantics (the transparent checker's evaluation loop)."""

    local: list[int] = []
    for gate in program.gates:
        semantics = gate_set[gate.op]
        args = tuple(local[v] if space == LOCAL else inputs[v] for space, v in gate.args)
        local.append(semantics.evaluate(args))
    (out,) = program.outputs
    return int(local[out])


def bf16_small(rng: np.random.Generator, n: int) -> np.ndarray:
    """bf16 words of N(0, 1) f32 values (an activation-like row; no NaN/Inf)."""

    f = rng.standard_normal(n).astype(np.float32)
    return (f.view(np.uint32) >> 16).astype("<u2")


class SynthInstance(PB.Instance):
    def __init__(self, family: str, weights: PB.Safetensors, seed: int) -> None:  # noqa: D107 -- PB.Instance.__init__ is replaced
        param, k = CALLS[family]
        self.family, self.K = family, k
        self.meta = {"family": family, "K": k, "phase": "synthetic", "layer": 0, "weight_param": param, "x_id": f"synthetic/x/{family}/seed={seed}",
                     "canonical_id": f"synthetic/out/{family}/seed={seed}"}
        self.W = weights.matrix(param)
        self.N = int(self.W.shape[0])
        assert self.W.shape[1] == k, (param, self.W.shape)
        rng = np.random.default_rng(seed)
        self.x = bf16_small(rng, k)
        self.j = 0
        self.out = np.zeros(self.N, dtype="<u2")
        self.trees = None
        self.dir = Path(f"synthetic/{family}")
        self._evaluated: set[int] = set()

    def evaluate_rows(self, gate_set, program, rows: list[int]) -> None:
        xs = [int(v) for v in self.x.tolist()]
        for j in rows:
            if j in self._evaluated:
                continue
            self.out[j] = evaluate_program(gate_set, program, xs + [int(v) for v in self.W[j].tolist()])
            self._evaluated.add(j)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--checkpoint", default=str(CHECKPOINT))
    ap.add_argument("--families", default="qkv_proj,down_proj")
    ap.add_argument("--sizes", default="1,64")
    ap.add_argument("--seed", type=int, default=20260912)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    weights = PB.Safetensors(Path(a.checkpoint))
    builder = PB.Builder()
    sizes = sorted({int(s) for s in a.sizes.split(",") if s.strip()})
    entries: list[dict[str, Any]] = []
    for family in [f for f in a.families.split(",") if f]:
        inst = SynthInstance(family, weights, a.seed)
        program = builder.program(inst.K, 1)
        print(f"{family}: K={inst.K} N={inst.N} program {program.kind.hex()[:12]} gates={len(program.gates)}", flush=True)
        # every row any size opens is evaluated BEFORE the first commit (the out tree is built once and cached by PB.Instance.commit)
        t0 = time.perf_counter()
        inst.evaluate_rows(builder.gate_set, program, sorted({j for n in sizes for j in PB.spread(inst.N, n, inst.j)}))
        eval_s = time.perf_counter() - t0
        for n in sizes:
            rows = PB.spread(inst.N, n, inst.j)
            statement, witness, check_s = builder.batch([(inst, [j]) for j in rows])
            rel = f"coordinate/{family}/K{inst.K}/n{len(rows)}"
            entry = write_batch(out / rel, statement, witness, {
                "workload": "smollm2-135m-gemm-coordinate-synthetic", "model": "HuggingFaceTB/SmolLM2-135M", "checkpoint_snapshot": "93efa2f0", "family": family,
                "kind": f"GemmCoordinate_v1{{K={inst.K}}}|batch", "K": inst.K, "N": inst.N, "rows": rows, "size": len(rows), "composition": "same_instance",
                "gate_set": f"{builder.gate_set.name}@{builder.gate_set.version}", "operands": "x SYNTHETIC (seeded N(0,1) bf16), W REAL (checkpoint row), out = exact Python-semantics evaluation",
                "commitment": "SYNTHETIC word-per-leaf trees (frame v3) for x / W / out (labelled difference from the record's chunks-4096 leaves)",
                "transparent_check_s": check_s, "evaluate_s": eval_s, "seed": a.seed})
            entries.append(entry)
            print(f"  {rel}: {len(statement.obligations)} obligations, {entry['positions']:,} positions, statement {entry['statement_bytes']:,} B, "
                  f"witness {entry['witness_bytes']:,} B (eval {eval_s:.1f}s, transparent check {check_s:.2f}s)", flush=True)
    write_manifest(out / "manifest.json", entries, {"generator": "e2e/demo_core/night/gemm_synth.py", "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})
    print(json.dumps({"batches": len(entries), "manifest": str(out / "manifest.json")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
