# verity-gkr-verify: the independent verifier for Candidate A's GPU proofs

Lane `a-verifier`.  A dependency-free Rust verifier for the proofs of `backends/gkr/gpu` (the a-gpu prover: GKR +
LogUp + Ligero over BabyBear^6 = F_p[X]/(X^6 - 31), p = 2^31 - 2^27 + 1, SHA-256 Fiat-Shamir).  It shares no code with
the Python prover/verifier or with the Goldilocks Rust crate in `../src`: field, SHA-256, transcript, NTT, circuit
parser, `proof.bin` parser and every check are written from `../gpu/README.md`, `../PROTOCOL.md` and the a-gpu source.
Before it, no implementation other than the a-gpu lane's own Python port had accepted a GPU A proof.

~~~text
cargo build --release
verity-gkr-verify verify    --dir D --proof proof.bin [--vus N] [--threads T] [--json out.json]
verity-gkr-verify negatives --dir D --proofs DIR  [--threads T] [--json out.json]     # every DIR/<id>.bin (D/neg/<id>) must be rejected
verity-gkr-verify mutate    --dir D --proof proof.bin [--vus N] [--threads T] [--parallel P] [--sample K] [--json out.json]
python -m verifier.e2e --prover run.json --verifier out.json --out envelope.json         # A-GPU e2e with this verifier
~~~

`D` is a `gpu.bb_export` directory.  The verifier reads **only** `circuit.txt`, `epilogue.txt`, `manifest.json`
(`steps`) and `public.bin` (the words; `--vus N` = the first N).  It never opens `units.bin` / `epilogue.bin`.

## What it checks, and what it takes from where

Statement binding (`redteam_vu.md` F1-F3): every size comes from the statement -- `k = 4096, n = 16384, t = 192`
(the a-gpu config), rows = ceil(l / k) from the circuits' column counts x units, the message count from the
circuits' layer structure and the tables' hypercube dimensions, the Merkle depth from `n`, the column indices from its
own transcript.  `proof.bin`'s counts are *checked* against these, never *used*; a proof whose counts disagree is
rejected at parse (`proof.bin: message count N exceeds the statement's bound`, ...).  Non-canonical field elements are
rejected at parse.

1. Transcript: `H(domain)`, the circuits' text digest (unit, epilogue), the tables, `steps`, the public words, the
   Merkle root; then in order the per-table `(z, beta)` and per-segment `(z0, gamma)` challenges.
2. LogUp per table (`gpu/logup.py`): the fractional-sum tree from the root (`lambda q_root = p0 q1 + p1 q0 + lambda
   q0 q1`) down `n` levels of sumcheck (degree-3 messages, `h(0) + h(1) = running`), each level's final check, the
   leaf claim `(rho, lambda, value)`.
3. GKR per segment (`gpu/gkr.py`): every layer's phase-1 / phase-2a / phase-2b sumcheck rounds, the gate identity at
   the final point with the layer's linear forms (incl. the assertion layer's zero), `eq(z*, x*)`, the copy
   constraints, the depth; the input layer's two claims `(l*, v_l)`, `(r*, v_r)` at the copy point `c*`.
4. The functional `<a, X> = b`: `a` = rho-weighted input claims (eq tables), lookup claims (the queries' linear forms
   scattered over each unit's columns with `eq(rho_t, leaf)`, the table rows), the chain linkage (`x.c` of step
   `i+1` = `y` of step `i`, epilogue `y32` = the last `y`, `y16` = the public word, geometric weights); `b` from the
   claimed values, the lookup constants (`konst_t`), the padding leaves.
5. Ligero (`gpu/ligero.py`): `r` (the proximity row's coefficients), `w`, `q` absorbed; `q_0 + q_k` scaled by `k` = `b`;
   the `t` opened columns' Merkle paths against the root (SHA-256 of the column's `u32` LE bytes, then pairs); the
   proximity test `sum_i r^i C[i, j] = Enc(w)[j]`; the linear test `q(eta_j) = sum_i a_i(eta_j) C[i, j]` with every
   `a_i` built row by row from the statement and evaluated at the opened points (an inverse NTT of size `k` and one
   twisted forward NTT per needed coset, not the full size-`n` encode).

Everything that is a loop over independent rows / columns / eq blocks runs on `--threads` OS threads (`std::thread
::scope`; no rayon).  Peak RSS is ~100 MB at B = 4096 (the proof + one chunk of `a` per thread); `a` is never held
whole.

## Results (2026-09-22)

* **v1 (the a-gpu circuit), B = 4096, 12 threads of vy-cpu2: 1.00 s** (1T 10.3 s; B = 64: 0.077 / 0.222 s); the
  a-verifier lane's 3.02 s before the a-verifier-2 work (batched AVX-512 NTTs, output-pruned forward transforms,
  planar functional fill).  44/44 negatives, 16485/16485 mutations rejected.  Run r20260922-105647-c947; per-stage
  numbers and what changed in `backends/numerical/reports/a_verifier_2.md`; ledger `ledger/a-verifier-2.jsonl`.
* **F1 settled**: the kernel path's proofs are byte-identical to the torch path's on a clean 4090 (`gpu/F1_TRIAGE.md`);
  `main`'s prover is Rust-verified at B <= 512 from a committed tree.
* **Checker v2** (`"variant": "v2"` manifests: `chain.txt` links, `itable` implicit tables, table union across the
  segments): `lane/a-gpu-v2`'s pos64 and synth proofs accepted, 4/4 chain + 44/44 negatives + 786/786 sampled
  mutations rejected (r20260922-103210-9cba).

Env switches (all default on): `VERITY_GKR_NO_AVX512=1` scalar butterflies, `VERITY_GKR_ROWWISE=1` one row per NTT
(the a-verifier path), `VERITY_GKR_PRUNE=<stages>` (default 6) the pruned forward transform, `VERITY_GKR_PROFILE=1`
per-phase fill times on stderr.

## Layout

~~~text
src/field.rs       BabyBear, Montgomery multiply, Ext (X^6 - 31), inverse, roots of unity, interpolation
src/sha256.rs      SHA-256
src/transcript.rs  gpu/transcript.py: state <- H(state || msg); challenge words by rejection sampling on H(state || "chal" || ctr)
src/circuit.rs     circuit.txt / epilogue.txt parser (columns, linear forms, gates, queries, tables, R<w> and v2 `itable` implicit tables), GKR layers
src/ntt.rs         radix-2 NTT (Montgomery twiddles), 48-lane batched DIF/pruned-DIT, RowEval (8 rows at the opened positions)
src/lanes.rs       AVX-512 lane kernels (butterflies, 6-lane accumulate), runtime-detected, scalar fallback
src/proof.rs       proof.bin parser (statement-bounded)
src/verify.rs      the verifier: transcript replay, LogUp, GKR, the functional, Ligero
src/main.rs        verify / negatives / mutate
e2e.py             the A-GPU envelope with this verifier for benchmarks/latency_compare.py
~~~
