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
| v5 | 9fb4180 | a-gpu2: packed k=3 GKR phase 1, CUDA-graphed LogUp (a-fusion), INT8 tensor-core opening, SIMT commit NTT | **1.857 s** | 0.43 / 0.66 / 0.57 / 0.10 | 6.5 s |
| v6 | 5ad05fb | a-gpu2: vectorised Ligero verifier; fused rank-1 input-claim accumulate | **1.790 s** | 0.43 / 0.66 / 0.53 / 0.10 | 1.34 s |

v5/v6 are **not** transcript-identical to v2e (correction, lane a-protocol-diff, 2026-09-22): the LogUp per-level
sumcheck binds the low bit of the level index first and the next point is `rho = [mu] + x*` (`logup.py` docstring,
`../PROTOCOL.md` 4.2 "Variable order"), where v2e bound the top bit first with `rho = x* + [mu]`.  Same protocol
under bit-reversal of the level index, same message *count* and order, different message *values* from level 2 on --
the independent Rust verifier (`../verifier/`, written against v2e) rejected every v5/v6 proof at `LogUp POW level
2: final check` until it followed the new schedule (`note:r20-proof/a-protocol-diff/20260922T1047Z-report-a-protocol-diff`).  The GKR
schedule, the eq weights and everything else are v2e's.  Peak device memory 47.9 GB; `--warmup 1` excludes the
one-off ~60 s LogUp graph compile/capture per process.  Negatives: 52/52 rejected on the v6 path (r20260922-094606-1813).

Every run: 4096/4096 `vu-k1536` accepted with the recorded words, proof 33.9 MB (6086 extension elements + Ligero
opening), `rounds.sequential_depth` 438 (circuit 197, LogUp R16 435), 1920 transcript slots, peak device memory
65.7 GB (v1-v2c) / 46.7 GB (v2d-e), 177.8 M committed base elements in 43 417 Ligero rows.  Reference: the CPU Rust prover, 12 threads,
297.6 s for the same batch (Goldilocks).  The verifier is Python (transcript replay + Merkle paths in `hashlib`),
not optimised.

## Checker v2 (`lane/a-gpu-v2`; A100-SXM4-80GB, same prover, `gpu/v2/` + `--variant`-free: the manifest selects v2)

PROTOCOL §14's circuit (264 columns / 290 wires / 26 products / 86 lookups per unit into 10 tables, depth 1, the
(s, e, M, z) state chained across the 96 units, 583 linkage constraints per VU) exported over BabyBear by
`gpu.v2.export` and proved by the same prover: implicit tables materialised on device, `R<w>` range tables from the
queries, 12-column strided linkage read, table union over segments (DESIGN §5b).  Same box, same day, median of 3:

| circuit | epilogue | run id | B | prover | split (lookup / arith / open / commit) | verifier | proof | depth | committed | peak mem | label |
|---|---|---|---|---|---|---|---|---|---|---|---|
| v1 (a-gpu v2e) | limb (v1) | r20260922-091544-4e02 | 4096 | 7.93 s = 1.94 ms/VU | 2.22 / 1.75 / 3.07 / 0.77 | 10.1 s | 33.9 MB | 438 | 177.8 M | 46.6 GB | `NON_ZK_PROOF_DIAGNOSTIC` |
| v2 | word (a-v2) | r20260922-090930-3bc5 | 4096 | 5.62 s = 1.37 ms/VU | 2.78 / 0.33 / 2.03 / 0.43 | 8.1 s | 21.1 MB | 328 | 109.1 M | 24.4 GB | `INCOMPLETE_CHECKER_DIAGNOSTIC` (epilogue word arithmetic SAT over 31-bit fields) |
| v2 | **limb** (b-air-v2 gadget) | r20260922-095013-db80 | 4096 | **5.73 s = 1.40 ms/VU** | 2.85 / 0.36 / 2.05 / 0.43 | 8.1 s | 21.1 MB | 328 | 109.1 M | 24.4 GB | **`NON_ZK_PROOF_DIAGNOSTIC`** |
| v2 | limb | r20260922-095013-db80 | 64 | 2.95 s | 1.97 / 0.16 / 0.76 / 0.06 | 5.5 s | 1.87 MB | 279 | 6.87 M | 9.9 GB | `NON_ZK_PROOF_DIAGNOSTIC` |

v2/v1 same box: **1.38x** end to end (arith 4.9x, commit 1.8x, open 1.5x, lookup 0.78x — the lookup bucket is
round/launch-bound: 13 shallow trees instead of 5 deep ones, 3524 transcript slots vs 1920; `lane/a-fusion`'s
graphed LogUp is the fix, not the circuit).  Against a-gpu v1 on the H100 (4.43 s) v2 is 0.77x raw; with the
measured same-circuit A100/H100 ratio of 0.56x (inside anchor_a100's HBM-bound band 0.52–0.61x) that normalises to
~1.4x.  Not a breakthrough (< 2x).  Acceptance: 4096/4096 with the recorded words; negatives 52/52 `vu-k1536-neg`
(8 at witness, 44 at the verifier), a-v2's 30 unit + 4 chain cases 34/34, 15/15 epilogue base columns +1
(`gpu.run epilogue-negatives`).  Field: BabyBear^6 (the kernels' field); the epilogue's word-form Pack is the only
v2 gadget SAT over 31-bit fields and is replaced by `lane/b-air-v2`'s limb form (z3 direct UNSAT over BabyBear,
same function over Z); checker-min's remaining v2 queries re-run with the BabyBear modulus are UNSAT
(`gpu/v2/results/z3_babybear_v2.json`).  Results: `gpu/v2/results/` (`limbs/` for the honest-label set).

Soundness (`security.accounting.account_candidate_a`, BabyBear^6, Ligero `k = 4096, n = 16384, t = 192`):
**2^-130.19** since lane agkr-table (2026-09-24), computed live on the proved instance by `backends/gkr/soundness.py`
and written into every result's `security.achieved_log2`.  Before, 2^-127.7: the generic collision term of the
SHA-256 Merkle tree, `q^2 / 2^256` at `q = 2^64` hash queries, is 2^-128 by itself, so no statistical parameter could
bring the total under 2^-128.  The Merkle tree is now SHA-512 (`sha512_cuda.py` on the device, `hashlib` on the host,
`verifier/src/sha512.rs` in the Rust verifier with the FIPS 180-4 vectors; proof.bin v2 carries 64-byte digests),
which moves that term to 2^-384.  The remaining terms, by component: Ligero IRS query `(1 - e/n)^t = 0.625^192`
2^-130.19 (dominant), lookups 2^-158.6, batching 2^-164.3 (includes the chain linkage `(N - 1)/|F|` over the ~2.4 M
constraints folded by powers of one challenge), sumcheck 2^-176.6.  The accountant now books the Ligero proximity test
as the prover runs it: rows combined by powers `1, r, ..., r^(m-1)` of one challenge (`row_coeffs`), BCIKS20's
curve bound `((m-1) n + 1) / |F|` at the prover's `m = 26644` rows (2^-156.7) instead of the uniform-combination
`(n+1)/|F|`.  Model: the interactive protocol (PROTOCOL.md: the verifier's coins are live; B-Ligero's interactive mode
books the same way, `fiat_shamir_queries_log2: null`).  Proof files derive the coins as SHA-256 of the transcript so
they can be re-checked offline; no Fiat-Shamir soundness is claimed for them (under the campaign's `q x eps` rule every
statistical term would take a 2^60 factor).  Proof class `NON_ZK_PROOF_DIAGNOSTIC`, authentication excluded, omitted:
ZK masking.

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

## Proof file (`proof.bin`, `--proof-out` / `negatives --proof-dir`; `Proof.to_bytes`)

Little-endian throughout; an extension element is six `u32` coordinates `c0..c5` (24 bytes), a base element one
`u32`, a digest 32 bytes.

~~~text
magic  "VGKRBB6\0" (8)      version u32 = 1        root (32)
n_msgs u64                  msgs: n_msgs x 24      (every sumcheck/GKR/LogUp message in transcript order)
has_opening u64 (0 | 1)
k u64        w:  k x 24     (Ligero proximity row)
n_qc u64     qc: n_qc x 24  (the 2k-1 coefficients of q)
t u64  rows u64  depth u64
per opened column (t of them, in `open_set` order): rows x u32 values, then depth x 32-byte Merkle siblings
~~~

The counts make the file self-delimiting; a verifier takes `k, n, t`, `rows = ceil(l / k)`, `depth = log2 n`, the
message count and the column indices from the circuit/config and its own transcript and **rejects** a file whose
counts differ -- nothing read from the proof sizes anything (`note:r20-proof/red-team-vu/20260922T1042Z-report-redteam-vu` F1-F3).  The independent Rust
verifier for this format is `../verifier/` (`verity-gkr-verify verify --dir D --proof proof.bin [--vus N]`); it
reads `circuit.txt`, `epilogue.txt`, `manifest.json` (steps, chain columns) and `public.bin` only -- never the
witness files.

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
