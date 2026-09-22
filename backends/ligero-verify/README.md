# ligero-verify: the independent verifier for Candidate B's B-Ligero proofs

Lane `b-verifier`. A dependency-free Rust verifier for the chain proofs of `backends/direct/ligero` (BabyBear Ligero
with statement-bound SHAKE-256 challenges, BLAKE3 Merkle, BLAKE2b statement digest and HM96 coin commitments;
PROTOCOL.md sections 3, 4, 6, 7b, 8b, 8c). It shares no code with the Python prover/verifier: BLAKE3, BLAKE2b,
SHAKE-256 (Keccak), the BabyBear NTT, the coset code, the file parsers and every check are written from
`PROTOCOL.md`, `HANDOFF.md` and the red-team transcript rules (`redteam_vu.md` F1-F3), with the Python sources read
as the *reference of record* where the document is silent (each such place is a finding in `DISCREPANCIES.md`).

Before it, every B-Ligero proof had been verified only by the code that produced it (shared `field.py`, shared
transcript code, shared constants). This verifier accepts the honest 4096-VU proof sets in all three modes and
rejects every negative and every mutation; the agreement matrix against the Python verifier is in
`backends/numerical/reports/b_verifier.md`.

~~~text
cargo build --release
ligero-verify verify --system system.bin --statement X.stmt --proof X.proof [--threads T] [--soundness-bits 128]
                     [--allow-any-system] [--reps R] [--json out.json]
ligero-verify batch  --system system.bin --dir DIR [--threads T] [--soundness-bits 128] [--reps R] [--json out.json]
                     # every DIR/*.proof with its .stmt; agreement with DIR/manifest.json's python_verdict when present
ligero-verify system-digest --system system.bin
cargo test --release                                 # hash vectors (hashlib / BLAKE3 spec), NTT / coset code
~~~

## Files (`ligero-*/v1`, defined in `backends/direct/ligero/serialize.py`)

| file | what | who writes it |
|---|---|---|
| `system.bin` (`ligero-system/v1`) | the compiled REAL chain constraint system: pins, 347 linear + 3387 quadratic constraints as sparse affine forms, the chain rows, and the Python `system_id` | `serialize.py dump-system` / every generator |
| `X.stmt` (`ligero-statement/v1`) | the verifier's configuration (`l, n, D, t, target, zk, t_pad, coin_commitment, n_proofs`) and the public words (`a, b` per unit, `y16` per column; chain layout `n_vus, steps`) | the prover side (`gen-set`, `gen-small`, `gen-negatives`) |
| `X.proof` (`ligero-proof/v1`) | the transcript: coin commitments + openings, root, `w, h, q [, v]`, the `t` column indices, the opened `M x t` columns, the Merkle paths | the prover side |

The verifier reads the statement first, derives every size from it and the system (`k = l + t_pad`, `h_len = 2k - 1
- l`, `q_len = l + k - 1`, `M = m + 5D` under ZK, depth `log2 n`), and parses the proof against those bounds: a
header that disagrees is `malformed proof` before the body is read; every field element is checked `< p` at parse.
The system file is a verification key: its Python `system_id` **and** this crate's own BLAKE3 digest of the exported
tables are pinned in `verify.rs` (`PINNED_SYS_ID`, `PINNED_TABLE_DIGEST`); a different system is refused unless
`--allow-any-system`.

## What it checks, in order (reject-first, no panics on malformed input)

1. Parameters: Appendix C recomputed from the statement (`k = l + t_pad` under ZK, `l + 1` otherwise as the
   accounting does; `e = floor((n - k)/2)`; challenge field BabyBear^D; the step-0 coin-hiding term `2 * 2^64 / 2^256`
   when the commitment is on) and union-bounded over `n_proofs` sub-batches; **refused when above `2^-soundness_bits`**
   (default 128). Also `n > 2k` (quadratic test), `k > l + t` under ZK (Lemma 4.15), power-of-two `l, n`, the layout.
2. Step 0: `c_i = BLAKE2b-256("ligero-b/coin|v1|" || r_i || s_i)` for both slots (when the mode commits the coins).
3. Public inputs: BF16 decodes of every operand (`s, m, e` pins; an all-ones exponent field is rejected), `y16 <
   2^16` by the format.
4. Statement digest `BLAKE2b-256("ligero-b/statement|v2" | p | hash | l | n | D | t | target | zk | k | sys_id | mode |
   layout | a | b | y16)`; challenge 1 = SHAKE-256(BLAKE3("ligero-b/c1|" stmt coins root r1)) -> `r (D x M)`,
   `rho_lin (D x L)`, `rho_quad (D x Q)`, `rc (D x 7 x l)`; challenge 2 = the `t` distinct columns by rejection from
   SHAKE-256(BLAKE3("ligero-b/c2|" stmt coins root w h q [|v= v] r2) || ctr) -- must equal the proof's column list.
5. Merkle: leaf = BLAKE3(column as `M` u32 LE), node = BLAKE3(left || right), `t` paths against the root.
6. Proximity: `w_d(eta_c) = sum_i r_{d,i} U_i[c]` (+ the prox mask row with coefficient 1 under ZK; `w` are
   coefficients under ZK, evaluations on H otherwise).
7. Linear: `alpha = sum_q rho_q a_q`, `beta = sum_q rho_q (pub_q - c_q)`; ZK: `v|_H = beta` (fold mod `x^l - 1`,
   NTT_l) and `v(eta_c) = alpha . U[c] + Z_H(eta_c) s_lin[c]`; non-ZK: `Enc(beta)[c] = alpha . U[c]`.
8. Quadratic: `sum_q rho_q (A_q B_q - C_q)(U[c])` (+ `Z_H (s_qa + eta^k s_qb)`) `= h(eta_c) Z_H(eta_c)`, all 3387
   constraints evaluated from the CSR tables at each opened column.
9. Chain (the general linear test): the coefficient rows `R_i` on H from `rc` and the layout masks, `l sum_{l | j} q_j =
   r^T b`, and `q(eta_c) = sum_i R_i(eta_c) U_i[c]` (+ `Z_H s_chain`).

Every NTT (D messages, D x R chain rows) and every column loop (Merkle, quadratic) runs on `--threads` OS threads
(`std::thread::scope`, no rayon). Stage timings (`parse, statement_hash, challenge_hash, merkle, ntt, dfold,
quadratic, chain`) are in the JSON.

## What it does NOT do

* Unit-mode proofs (`compile_unit(chain=False)`, public accumulator per unit): not implemented (the decision workload
  is the chain).
* It does not recompute the constraint system from the checker: the tables come from `system.bin` (pinned digests).
  An error in the *compiler* (a constraint missing from the system) is outside this verifier's reach; `redteam_vu`
  / `redteam_battery` cover that.
* It verifies **soundness only**: acceptance says the transcript proves the statement under the stated parameters.
  Nothing here checks zero knowledge; the `COMPLETE_ZK_BACKEND` label is lane b-zk's claim.

## Layout

~~~text
src/hash.rs     BLAKE3 (chunks, parents, root), BLAKE2b-256, SHAKE-256 -- tests against hashlib / spec vectors
src/field.rs    BabyBear, radix-2 NTT (natural in/out), coset points, Z_H, encode / eval_on_coset -- tests vs naive
src/format.rs   ligero-system / -statement / -proof v1 parsers (statement-bounded, canonical field elements)
src/verify.rs   soundness accounting, statement digest, challenges, Merkle, the four tests, the chain test
src/main.rs     verify / batch / system-digest; agreement with manifest.json's Python verdicts
~~~
