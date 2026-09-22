# Candidate A on the GPU (torch + Triton, BabyBear^6)

Track A's prover (GKR + LogUp + Ligero, `../PROTOCOL.md`) for the **full K=1536 VU** (96 chained units + epilogue,
PROTOCOL §12) at **batch 4096 on one H100**, in Python/torch with fused Triton kernels, over `F_{p^6}`,
`p = 2^31 - 2^27 + 1`.  Design: `DESIGN.md`.  Protocol and relation are those of the CPU Rust crate (`../src/`)
with the field changed and the sub-16-bit range check fixed (DESIGN §1); the proof is checked by the Python verifier
in `prover.py` (`verify`), a faithful port of `../src/` over the new field (the Rust verifier is Goldilocks-only, so
it cannot check these proofs directly; the Python verifier is the same code path that rejects the negatives).

## Runs (H100 80GB HBM3, B = 4096 VUs = 393 216 units + 4096 epilogues, `research run` ids in the ledger)

| version | commit | what changed | prover median of 3 | split (lookup / arith / open / commit) | verifier |
|---|---|---|---|---|---|
| v1 | 517dc4e | pure torch | 26.95 s | 10.82 / 8.91 / 6.71 / 0.47 | 10.0 s |
| v2a | 53ad266 | fused `ext_mul` (Montgomery SIMT) | 17.50 s | 5.48 / 5.74 / 5.74 / 0.45 | 9.4 s |
| v2b | 581c6e5 | fused fold + LogUp degree-3 round | 13.81 s | 1.53 / 5.39 / 6.38 / 0.45 | 9.8 s |
| v2c | 0d6437c | fused GKR phase-1 round; opening as tensor-core matmuls | 8.32 s | 1.49 / 1.85 / 4.46 / 0.45 | 8.9 s |
| v2d | 428db0b | lookup functional as a CSR scatter kernel | 5.41 s | 1.53 / 1.95 / 1.41 / 0.45 | 7.1 s |
| v2e | c4d472a | sparse gate evaluation; fused LogUp level combine | **4.43 s** | 1.40 / 1.07 / 1.44 / 0.45 | 7.1 s |

Every run: 4096/4096 `vu-k1536` accepted with the recorded words, proof 33.9 MB (6086 extension elements + Ligero
opening), `rounds.sequential_depth` 438 (circuit 197, LogUp R16 435), 1920 transcript slots, peak device memory
65.7 GB (v1-v2c) / 46.7 GB (v2d-e), 177.8 M committed base elements in 43 417 Ligero rows.  Reference: the CPU Rust prover, 12 threads,
297.6 s for the same batch (Goldilocks).  The verifier is Python (transcript replay + Merkle paths in `hashlib`),
not optimised.

Soundness (`security.accounting.account_candidate_a`, BabyBear^6, Ligero `k = 4096, n = 16384, t = 192`):
2^-127.7, dominated by the generic SHA-256 collision term; ledger security field `2^-127.7`, proof class
`NON_ZK_PROOF_DIAGNOSTIC`, authentication excluded, omitted: ZK masking, external authentication.

## Negatives

`vu-k1536-neg` (52 instances): 52/52 rejected — 8 at witness generation (`bf16.t`/`acc.t: 255 not in [0, 255)`,
a hint outside its table range, i.e. the prover cannot even build an in-domain witness) and 44 by the verifier
(`assertions phase-2a: round 0 sum mismatch`: the assertion layer's claimed zero is not zero).  An accepted negative
would be a soundness bug; there are none.  Re-run with `negatives` below.

## How to run

On the pod (`/workspace/venv312/bin/python` = torch cu124 + Triton; `cupy-cuda12x` for `hash_gpu`):

~~~
cd backends/gkr
# operands (bench-instances/v1, tier vu-k1536 and vu-k1536-neg) then the BabyBear export
python -m verity_numerical.bench.instances build
PYTHONPATH=. python -m gpu.bb_export export --root bench-instances/v1 --tier vu-k1536 --hi 4096 --out /workspace/bb/pos4096 --procs 16
PYTHONPATH=. python -m gpu.bb_export export --root bench-instances/v1 --tier vu-k1536-neg --out /workspace/bb/neg --procs 16
# prove + verify (median of --reps), JSON record with the split
PYTHONPATH=. python -m gpu.run prove /workspace/bb/pos4096 --device cuda --reps 3 --json out.json
PYTHONPATH=. python -m gpu.run prove /workspace/bb/pos4096 --vus 64 --device cuda          # the 64-VU gate
PYTHONPATH=. python -m gpu.run negatives /workspace/bb/neg --device cuda --json neg.json   # all must be rejected
~~~

Laptop (CPU, no Triton, < 30 s): a synthetic 2-VU, 3-step batch through the whole prover and verifier, PCS included:

~~~
PYTHONPATH=backends/gkr python -m gpu.bb_export synth --out /tmp/bbsynth --vus 2 --steps 3
PYTHONPATH=backends/gkr python -m gpu.run prove /tmp/bbsynth --device cpu
~~~

`VERITY_GPU_TORCH_ONLY=1` disables every Triton kernel (the torch path is the bit-exact reference the kernels are
tested against; `--clear` skips the commitment and opening for a clear-mode diagnostic).

## Where the time goes (v2e, profiler with per-call syncs, B = 4096; the wall split is 1.40 / 1.07 / 1.44 / 0.45)

Ligero opening 1.2 s (0.84 s in the `Mᵀ (aᵀ X) M` fp64 matmuls, the rest Merkle paths + column extraction on the
host), LogUp rounds 0.6 s (1201 sequential rounds: each is one kernel, a block-partial reduction and a
device-to-host sync for the transcript), LogUp folds 0.6 s (6562 small launches, ~90 µs each of Python + launch
overhead), phase-1 rounds 0.4 s (119), commit 0.4 s (NTT butterflies in torch + SHA-256 Merkle via `hash_gpu`),
eq tables 0.3 s (152), the dense layer matrices for phase 2 0.24 s.  The prover is now dominated by the
sequential round structure (1920 transcript slots, each a sync) and launch overhead rather than by arithmetic:
next steps are CUDA graphs / one launch per round (fold fused into the round kernel, all five LogUp tables folded
in one kernel), a batched NTT for the commitment, INT8 tensor-core limbs for the opening matmuls, and a compiled
verifier (the Python verifier at 7 s is now slower than the prover).

## Kernels (`kernels.py`, all bit-exact against the torch path incl. `p-1` edge values)

* `ext_mul`: elementwise `F_{p^6}` product, 32-bit Montgomery (`mul.lo + umulhi + cond. add` per base product,
  `X^6 = 31` via pre-scaled `31 R`), canonical in/out.  2^24 products: 13.2 ms (torch) -> 1.31 ms.
* `fold`: `v[i] + (v[i+half] - v[i]) r` for base or extension tables, 2^22 elements 2.0 -> 0.19 ms.
* `logup_round`: the LogUp degree-3 round at 4 points in one pass (`p0 q1 + p1 q0 + λ q0 q1`, `eq`-weighted,
  block partials reduced on the host): 5.6 -> 0.39 ms at 2^18.
* `phase1`: the GKR copy-round `h(t) = Σ_c e_t(c) Σ_g w_g a_t(c,g) b_t(c,g)` with 2D tiles over copies x gates:
  591 -> 6 ms at 2^18 x 64 gates (extension), 12.8 -> 0.8 ms (base).
* `scatter_terms`: the lookup functional `v[u, idx] += Σ e[u, j] c_{j,idx}` over the sparse (query, column) terms
  (CSR by column), replacing a `(units, 3546) x (3546, 2712)` fp64 matmul per table.
* Ligero opening (`ligero.open_w_qc`): `w = rᵀ X`, `G = Mᵀ (aᵀ X) M` with `M` the INTT matrix, `qc[m] = Σ_{u+v=m} G[u, v]`
  (all `mm_mod` = exact 16-bit-limb float64 matmuls on tensor cores); `open_w_qc_reference` is the v1
  encode-multiply-INTT path kept for the equality test.
