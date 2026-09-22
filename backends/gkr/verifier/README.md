# verity-gkr-verify: the independent verifier for Candidate A's GPU proofs

Lane `a-verifier`.  A dependency-free Rust verifier for the proofs of `backends/gkr/gpu` (the a-gpu prover: GKR +
LogUp + Ligero over BabyBear^6 = F_p[X]/(X^6 - 22) (since a-gpu2; a-gpu used X^6 - 31, whose proofs this verifier no longer accepts), p = 2^31 - 2^27 + 1, SHA-256 Fiat-Shamir).  It shares no code with
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
   leaf claim `(rho, lambda, value)`.  Variable order: **low bit first** (`PROTOCOL.md` 4.2 "Variable order"; the
   a-gpu2 schedule, `rho = [mu] ++ x*`), the leaf `rho` reversed to `eq_table`'s top-bit-first convention.  The
   verifier's first version followed a-gpu's top-bit-first order (`rho = x* ++ [mu]`) and rejected every a-gpu2
   proof at `LogUp POW level 2: final check` (lane a-protocol-diff).  `verify --trace-logup` prints every level's
   `rho, x*, lambda, p0, p1, q0, q1, eq, eq*g, running, mu` and the value the other pairing would give.
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

## Results (2026-09-22; proofs from the a-gpu prover on vy-g5, `VERITY_GPU_TORCH_ONLY=1` -- see the discrepancies report
for why the kernel path's proofs are rejected by both verifiers today)

See `backends/numerical/reports/a_verifier_discrepancies.md` and the ledger `backends/numerical/reports/ledger/
a-verifier.jsonl` for the numbers (verify seconds 1 / 12 threads at B = 64 and 4096, negatives 44/44, mutations
16485/16485 at B = 64, the A-GPU end-to-end rows).

## Layout

~~~text
src/field.rs       BabyBear, Montgomery multiply, Ext (X^6 - 22), inverse, roots of unity, interpolation
src/sha256.rs      SHA-256
src/transcript.rs  gpu/transcript.py: state <- H(state || msg); challenge words by rejection sampling on H(state || "chal" || ctr)
src/circuit.rs     circuit.txt / epilogue.txt parser (columns, linear forms, gates, queries, tables, implicit R16), GKR layers
src/ntt.rs         radix-2 NTT (Montgomery twiddles), RowEval (evaluate a row at the opened positions)
src/proof.rs       proof.bin parser (statement-bounded)
src/verify.rs      the verifier: transcript replay, LogUp, GKR, the functional, Ligero
src/main.rs        verify / negatives / mutate
e2e.py             the A-GPU envelope with this verifier for benchmarks/latency_compare.py
~~~
