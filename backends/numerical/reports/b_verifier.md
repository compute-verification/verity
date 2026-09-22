# b-verifier: an independent Rust verifier for Candidate B's B-Ligero proofs

Lane `b-verifier` (track B), 2026-09-22, branch `lane/b-verifier`.  Crate `backends/ligero-verify/`
(`verity-ligero-verify`, zero dependencies: its own BabyBear, NTT and coset code, BLAKE3, BLAKE2b-256, SHAKE-256,
parsers for the `ligero-{system,statement,proof}/v1` files, and the proximity / linear / quadratic / chain tests), written
from `backends/direct/ligero/PROTOCOL.md` + `HANDOFF.md` + `redteam_vu.md` F1-F3; the Python sources were read as the
reference of record only where the document is silent, and each such place is an entry in
`backends/ligero-verify/DISCREPANCIES.md`.  It shares no code, constants file or transcript helper with the Python
prover / verifier.

What this lane verifies: **soundness of the transcript** -- statement digest, both SHAKE-256 challenge derivations,
the step-0 coin commitments and their openings, the Merkle paths, the proximity / linear / quadratic / chain tests on
the opened columns, the D-fold, and the union bound over the sub-batches recomputed from the file's own parameters
(refuse below `--soundness-bits`, default 128).  What it does not verify: the zero-knowledge property.
`COMPLETE_ZK_BACKEND` (ZK), `COMPLETE_HVZK_BACKEND` (HVZK) and `NON_ZK_PROOF_DIAGNOSTIC` (non-ZK) are the b-zk lane's
labels for these proofs; this report carries them as *their* claims.

## 1. Agreement matrix (Rust verifier vs the Python verifier on the same files)

Proofs from the torch prover on `vy-g3` (L40S), run `r20260922-100145-374a` (honest / negative / transcript
mutations) and `r20260922-101053-0dc9` (witness mutations); sha256 of every `.proof` / `.stmt` in
`backends/ligero-verify/fixtures/manifests/*.manifest.json`, Rust verdict JSONs (per-proof reason, per-stage timing,
recomputed soundness) in `fixtures/rust_verdicts/`.  "Honest 4096" = 4096 VUs of K = 1536 as 25 sub-batches x 170
VUs, `l = 16384`, `n = 65536`, `D = 6`, `t = 197` (ZK, HVZK; `k = l + 256`) / `196` (non-ZK), 122.3 MB (ZK, HVZK) /
110.8 MB (non-ZK) per set.  "Small" = B = 1 (VU 0) and B = 64 proofs.

| set | mode | n | Rust accept | Rust reject | Python accept | agree | note |
|---|---|---|---|---|---|---|---|
| honest 4096 (25 sub-batches) | ZK | 25 | 25 | 0 | 25 | 25/25 | union 2^-128.05 recomputed |
| honest 4096 | HVZK | 25 | 25 | 0 | 25 | 25/25 | union 2^-128.05 |
| honest 4096 | non-ZK | 25 | 25 | 0 | 25 | 25/25 | union 2^-128.25 |
| small B=1, B=64 | ZK | 2 | 2 | 0 | 2 | 2/2 | per proof 2^-128.16 / 2^-128.46 |
| small B=1, B=64 | HVZK | 2 | 2 | 0 | 2 | 2/2 | |
| small B=1, B=64 | non-ZK | 2 | 1 | 1 | 2 | 1/2 | **B=1 refused: its parameters give 2^-127.73 (D1)**; 2/2 at `--soundness-bits 127.5` |
| 52 `vu-k1536-neg` negatives | ZK | 52 | 0 | 52 | 0 | 52/52 | 44 chain sum / 4 column challenge / 4 off-domain operand |
| 52 negatives | HVZK | 52 | 0 | 52 | 0 | 52/52 | same split |
| 52 negatives | non-ZK | 52 | 0 | 52 | 0 | 52/52 | at 2^-128 all 52 refused on parameters (l = 256); at 2^-127.5 the same 44/4/4 as Python |
| transcript mutations | ZK | 35 | 0 | 35 | 0 | 35/35 | see table below |
| transcript mutations | HVZK | 35 | 4 | 31 | 4 | 35/35 | the 4 = coin commitment / opening mutations: HVZK coins are bound to nothing (D3) |
| transcript mutations | non-ZK | 32 | 0 | 32 | 0 | 32/32 | (no mask row, no `v`); named reasons at 2^-127.5 match Python 32/32 |
| witness mutations (before commitment) | ZK | 8 | 0 | 8 | 0 | 8/8 | 7 linear / 1 quadratic -- rejected *inside the tests* |
| witness mutations | non-ZK | 8 | 0 | 8 | 0 | 8/8 | same |
| **total** | | **355** | **89** | **266** | **90** | **354/355** | the one disagreement is D1, by design |

The reason classes agree in every case where both reject (Rust's message is more specific: it names the slot, the
path index, the header field or the element index).  Per-mutation reasons for the ZK set:

| mutation | Rust | Python |
|---|---|---|
| opened value +1 / -1, mask row +1 | `merkle path 0 invalid` / `merkle path 188 invalid` | same |
| opened value = p (non-canonical), w = p | `proof file: non-canonical field element 2013265921 >= p at index 0` | `transcript element out of the field` |
| Merkle sibling flipped | `merkle path 0 invalid` | same |
| Merkle paths truncated / extended | `malformed proof: header depth = 10 / 12, the statement fixes 11` | `malformed Merkle paths` |
| w, h, h last coefficient, q, q sum-preserving, v, v in ker H, root | `column challenge mismatch` | same |
| t-1 columns; statement t-1 | `malformed proof: header t = 188 / 189, the statement fixes 189 / 188` | `malformed proof` |
| duplicated column index | `bad column set: repeated column index` | `bad column set` |
| column index out of range | `proof file: column index 2048 >= n` | `bad column set` |
| columns swapped (same set, other order) | `column challenge mismatch` | same |
| extra / dropped opened row | `malformed proof: header M = 3547 / 3545, the statement fixes 3546` | `malformed proof` |
| coin commitment c1 flipped; r1 / s2 opening mismatch | `coin slot 1 / 2: opening does not match its commitment` | `coin opening does not match its commitment` |
| coins stripped | `coin commitments / openings missing (the verifier's mode commits the coins)` | `malformed coin commitments / openings` |
| statement operand a / b, y16, coin-mode flipped; sub-batch proof under the other sub-batch's statement (both directions) | `column challenge mismatch` | same |
| statement zk-mode flipped | `linear-test message presence disagrees with the zk mode` | `malformed proof` |
| statement operand exponent = 255 | `public inputs: operand exponent field off the finite range` | `public inputs: ... a.t off the finite range` |

Witness mutations (`serialize.py gen-witness-mutations`, the prover's `mutate=` hook before commitment, so root,
messages and openings are consistent with a wrong witness): one column of one row of each row class (pin, hint,
prod, bit, sel), a broken chain link, `c_0 != 0` on the second VU, a whole bit column flipped -> 7 x `linear
constraints failed` + 1 x `quadratic constraints failed` (the hint row), identical in both verifiers and both modes.
These are the only negatives that reach the tests: every transcript mutation is caught earlier by the Fiat-Shamir
binding or the Merkle path (DISCREPANCIES.md D6).

Gate: **honest set accepted (25 x 3 + B = 1/64 x 3 at each file's own bound), every negative and every mutation
rejected with a named reason**, except the four HVZK coin mutations that are not forgeries (both verifiers accept,
nothing is bound to those bytes).  The Rust integration tests (`cargo test`) replay the honest B = 1 ZK fixture and a
byte-flip sweep over every region of it (header, root, messages, columns, openings, paths, coins) plus a flipped
statement word.

## 2. Discrepancies found (full text and structured entries: `backends/ligero-verify/DISCREPANCIES.md`)

* **D1 (open, b-zk).** `config_for` chooses `t` with `k = l`; `soundness()` reports with `k = l + 1`; `verify()`
  checks neither against the target.  At `l = 16384` the two agree (2^-128.25 / 2^-128.05 union); at `l = 256` non-ZK
  the chosen `t = 189` gives 2^-127.73 by the reported formula and Python accepts a proof whose statement claims
  2^-128.  The Rust verifier recomputes the bound and refuses.
* **D2 (open, b-zk).** The statement digest's byte layout is undocumented and absorbs Python's `repr(float)` of the
  target; the Rust verifier reproduces it and refuses exponent-form targets rather than guess.
* **D3 (open, b-zk).** HVZK proofs carry 192 bytes of coin commitments / openings bound to nothing (the prover emits
  them in every mode; `_coin_prefix` / `_coin_open` are empty when the commitment is off).  Harmless for soundness;
  a mutation of them is accepted by both verifiers.
* **D4 (resolved, scope).** Python's `coins are not this verifier's step-0 coins` is an interactive check; a file
  verifier checks the Fiat-Shamir chain (`c_i = H(tag || r_i || s_i)`, both derivations).  Soundness only.
* **D5 (open, b-zk).** Code-only facts a verifier needs: challenge-1 block order (`r | rho_lin | rho_quad | 7 chain
  families x l`), the column-sampling loop (`SHAKE(seed || ctr_u32)`, `2t` words per call, `u mod n`, ordered set),
  leaf = `BLAKE3(column as M u32 LE)` with no leaf/node domain separation, the `|v=` tag, message lengths, the coset
  `31 . <w_n>`, the `<u2` public-word encodings and the off-domain refusal.  Now written down in the crate README.
* **D6 (resolved).** Transcript mutations never reach the tests; witness mutations were added to the gate.
* Checked and consistent: `redteam_3.md` O1 (the torch transcript draws independent coefficients, as section 4
  says -- a geometric-power prover would be rejected), `redteam_vu.md` F1-F3 (every size from the verifier's
  statement; the statement mutations confirm the binding), O5 (`n_proofs` is in the statement file and the bound is
  the union over it).

## 3. Timing

Rust verifier on the 4096-VU ZK set (25 sub-batches; the sum over them is the whole batch's verification), best of 3
reps per sub-batch.  `vy-cpu3` is an AMD EPYC 7713 container with 16 vCPUs, shared with the b-bind lane's ~7-core job,
on a host with load average ~420 over 256 cores; the numbers below moved by 2x between two runs 15 minutes apart, so
they are upper bounds on this verifier, not a measurement of it.  Idle-laptop numbers (Apple M-series) are given as the
low-noise reference on different hardware.

| where | threads | sum over 25 sub-batches | per sub-batch | parse | statement | challenge (SHAKE/BLAKE3) | Merkle | message NTT / encode | D-fold linear | quadratic | chain | run |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| vy-cpu3 (pre-Barrett mul, quieter) | 1 | 8.29 s | 0.33 s (+-1%) | 0.006 | 0.012 | 0.020 | 0.011 | 0.215 | 0.033 | 0.020 | 0.012 | r20260922-101533-4fe3 |
| vy-cpu3 (pre-Barrett) | 12 | 5.85 s | 0.23-0.30 s | | | | | 0.12-0.20 | 0.034 | 0.017 | 0.011 | same |
| vy-cpu3 (Barrett, loaded) | 1 | 15.95 s | 0.64 s (0.39-1.37) | 0.25 | 0.52 | 0.90 | 0.50 | 9.14 | 2.90 | 1.14 | 0.48 | r20260922-102215-ad56 (stage columns are sums) |
| vy-cpu3 (Barrett, loaded, cold 1 rep) | 12 | 6.72 s | 0.27 s (0.22-0.34) | 0.07 | 0.20 | 0.47 | 0.17 | 3.50 | 1.55 | 0.37 | 0.34 | same (sums) |
| laptop, idle | 1 | 3.37 s | 0.134 s | 0.001 | 0.003 | 0.006 | 0.003 | 0.086 | 0.021 | 0.012 | 0.003 | local |
| laptop, idle | 8 | 1.28 s | 0.051 s | | | | | | | | | local |
| Python verifier (torch, L40S GPU, b-zk's host) | GPU | 1.03 s | 0.041 s | | | | | | | | | manifest `python_verdict` |

Where the time goes: encoding the D = 6 rows of each message (`w`, `v`, the mask rows' contributions, the chain
message) onto the `n = 65536` coset -- about 30 NTTs of 2^16 per sub-batch -- is 60-65% of the single-thread time;
the D-fold of the 347 linear constraints and the 7 chain families is 15-25%; hashing (statement BLAKE2b, two BLAKE3 +
SHAKE-256 expansions of ~6 x 22k words, 197 leaf hashes of 3546 words and 197 x 16 node hashes) is under 10%.  The
`%`-by-u64 BabyBear multiply was the EPYC hot spot (a 20-40 cycle divide); it is now a Barrett reduction (two
multiplies), which did not change the laptop time (fast divider) and could not be measured cleanly on the pod.  Threads
parallelise the encodes over rows and the column-wise tests only; a further 2-3x is available from a
radix-4 / cache-blocked NTT and from parallelising the challenge expansion, none of it done.

**Latency line.** `latency.md`'s B-Ligero row carries the Python verifier's 0.40-1.0 s on GPU hardware; the Rust
verifier's 5.9-16 s (EPYC, contended) / 1.3-3.4 s (laptop) is on *other* hardware and is added there as a note, not a
replacement: no same-hardware pair exists (the Python verifier needs torch + the L40S).

## 4. Deliverables

* `backends/ligero-verify/` -- the crate (`src/{field,hash,format,verify,main}.rs`, `tests/fixture.rs`, 11 tests),
  `README.md` (file formats, checks in order, CLI), `DISCREPANCIES.md`, `fixtures/` (B = 1 ZK proof + statement +
  `system.bin`, 2.7 MB; manifests with the sha256 of every large proof; Rust verdict JSONs).  The 110-122 MB sets are
  not committed: runs `r20260922-100145-374a` (honest, negatives, mutations), `r20260922-101053-0dc9` (witness
  mutations) on `vy-g3`; `r20260922-101533-4fe3`, `r20260922-102215-ad56` (timing) on `vy-cpu3`.
* `backends/direct/ligero/serialize.py` -- the `ligero-*/v1` formats (writer, reader, round-trip), `dump-system`,
  `gen-honest` / `gen-small` / `gen-negatives` / `gen-mutations` / `gen-witness-mutations`, and `verify` (the Python
  verifier driven from the files, used for the agreement column).  The torch prover gained the dump path through it
  (`--dump-proof` / `--dump-statement` semantics: every generator writes `.stmt` + `.proof` + `manifest.json`).
* Ledger `backends/numerical/reports/ledger/b-verifier.jsonl`: `b-verifier.accept_rate`, `.reject_rate`,
  `.seconds_1T`, `.seconds_12T` (scope `component`, `implementation_status` and the label caveat in `note`).

## 5. Open items

1. **D1** needs a decision from b-zk: make the claim at `k = l + 1` and choose `t` there (one more query at
   `l = 16384`, +0.5% proof size), or document `k = l` and make `soundness()` report it.  Until then the B = 1 non-ZK
   diagnostic proof does not meet its own statement's target by the reported formula.
2. **D2 / D5**: fold the byte layouts into PROTOCOL.md (statement digest, challenge block order, sampling loop, leaf
   hashing, message tags); replace `repr(float)` by an integer encoding of the target.
3. **D3**: drop the coins from the HVZK transcript or absorb them.
4. Timing on an unloaded CPU: the pod numbers are contended; a quiet EPYC run (or `--exclusive`) would give the
   real 1T / 12T figures.  Verifier optimisation headroom: radix-4 NTT, parallel challenge expansion.
5. The verifier's `system.bin` is a verification key pinned by two digests (Python `system_id` + this crate's own
   BLAKE3 of the exported tables); the compilation of the REAL constraint system itself (`compile_unit`) is not
   re-derived here -- a wrong-but-consistent system would be accepted with `--allow-any-system` only.
6. Leaf / node domain separation in the Merkle tree (D5): safe today because `4M != 64`, worth a tag.
