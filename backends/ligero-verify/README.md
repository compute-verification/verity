# ligero-verify: the independent verifier for Candidate B's B-Ligero proofs

Lanes `b-verifier` / `b-verifier-2` (the latter brought it to the b-zk-fix protocol, `main` a52441c). A dependency-free Rust verifier for the chain proofs of `backends/direct/ligero` (BabyBear Ligero
with statement-bound SHAKE-256 challenges, BLAKE3 Merkle, BLAKE2b statement digest and HM96 coin commitments;
PROTOCOL.md sections 3, 4, 6, 7b, 8b, 8c). It shares no code with the Python prover/verifier: BLAKE3, BLAKE2b,
SHAKE-256 (Keccak), the BabyBear NTT, the coset code, the file parsers and every check are written from
`PROTOCOL.md`, `HANDOFF.md` and the red-team transcript rules (`note:r20-proof/red-team-vu/20260922T1042Z-report-redteam-vu` F1-F3), with the Python sources read
as the *reference of record* where the document is silent (each such place is a finding in `DISCREPANCIES.md`).

Before it, every B-Ligero proof had been verified only by the code that produced it (shared `field.py`, shared
transcript code, shared constants). This verifier accepts the honest 4096-VU proof sets in all four modes (ZK /
non-ZK x `--mode interactive` / `--mode fiat-shamir`) and rejects every negative and every mutation; the agreement
matrix against the Python verifier is in `note:r20-proof/b-verifier/20260922T1132Z-report-b-verifier`.

**Read this first if you verify an `--mode interactive` file.**  In that mode the challenges are functions of the
verifier's step-0 coins alone (PROTOCOL.md 8c after b-zk-fix), so a proof *file* verifies against the coins it
carries -- and a prover that chose them can forge any statement (DISCREPANCIES.md D7: 49/49 forged files accepted).
Without `--coins` the verdict reads `accept (interactive: coins replayed from the transcript; sound only if they
were the verifier's step-0 coins)`; pass `--coins FILE` (your own `r1 || s1 || r2 || s2`, 128 bytes raw or hex) to
check a transcript from your own session.  Fiat-Shamir files (`--mode fiat-shamir`) are transferable and get a plain
`accept`.

~~~text
cargo build --release
ligero-verify verify --system system.bin --statement X.stmt --proof X.proof [--threads T] [--soundness-bits 128]
                     [--allow-any-system] [--reps R] [--json out.json] [--coins FILE]   # --coins: interactive mode only
ligero-verify batch  --system system.bin --dir DIR [--jobs N] [--threads T] [--target-bits 128] [--reps R] [--json out.json]
                     # every DIR/*.proof with its .stmt (+ its .coins when present, = `verify --coins` for that proof), the
                     # sub-batches verified concurrently on N jobs (default: available parallelism); batch ACCEPT iff every
                     # sub-batch accepts AND the union of the verifier's own per-proof bounds over the N presented is
                     # <= 2^-target AND every proof's n_proofs is N (PROTOCOL.md 8c "Batch verification"); agreement
                     # with DIR/manifest.json's python_verdict when present
ligero-verify system-digest --system system.bin
cargo test --release                                 # hash vectors (hashlib / BLAKE3 spec), NTT / coset code, Barrett mul,
                                                     # and tests/fixture.rs: the B=1 ZK interactive fixture accepted, byte flips
                                                     # in every region rejected, a flipped statement word rejected, --coins checked,
                                                     # batch verdict + parsed statements identical at --jobs 1 / 8, one corrupted
                                                     # sub-batch rejects the batch with the same error at any --jobs, the batch
                                                     # union bound and n_proofs = N enforced with the Python reason strings
~~~

Exit status: `verify` 0 accept, 1 reject (the reason on stdout), 2 usage / unreadable operator input.  `batch`: 0 every
sub-batch accepted and the batch rule met, 1 a sub-batch rejected (`batch_reason` names the first one in name order) or
the batch rule failed (`batch: n_proofs mismatch (...)`, `batch: union bound 2^-x over N sub-batches exceeds the target
2^-t`), 2 a sub-batch file unreadable, 3 a verdict disagrees with `manifest.json`'s Python verdict (checked first: the
agreement gate).  Malformed *proof or statement bytes* never panic (every read is bounded and every element checked);
an unreadable path given on the command line is reported and exits 2.

**Batch mode (lane b-verify-par; the batch rule: lane b-batch-bound).**  A 4096-VU batch is `N_subbatches` (25)
independently committed proofs, each with its own statement (its VUs' `a, b, y16`; `n_proofs` = 25, the count the
prover sized `t` for) -- there is no batch-level statement (PROTOCOL.md 8c; `bench-vu` dumps them as
`repK/sub_XX.{stmt,proof[,coins]}`).  `batch` verifies them on `--jobs` OS threads pulling from one queue, each
sub-batch exactly as `verify` would (its own statement, proof, challenges, checks, `--threads` inside) except for the
per-proof soundness gate, and accepts the batch iff every sub-batch accepts AND `verify::batch_bound` does: the union
`sum_i 2^-b_i` of this verifier's OWN per-proof bounds (`soundness()` from each statement's `n, k, l, t, D, mode`; never
a prover field) over the `N` proofs presented is `<= 2^-target` (`--target-bits`, default 128; `--soundness-bits` is
its alias here), and every proof's `n_proofs` equals `N`.  Before this rule `N` proofs each sized for `n_proofs = 1`
(2^-128.16 each) were accepted as a batch that is only 2^-(128.16 - log2 N); DISCREPANCIES D10.  The summary reports
`target_bits`, `batch_log2`, `batch_bits` (= `-log2 sum`), `per_proof_bits[]`, `n_proofs_claimed[]`; the Python file
verifier (`python -m backends.direct.ligero.serialize verify-batch --dir DIR [--target-bits 128]`) applies the same
rule with the same strings and agrees on the bits to 1e-6 (`protocol.batch_bound`).
Shared, read-only, computed once: the parsed `system.bin` with its pinned-digest verdict, and the domain tables per
`(l, n)` (NTT twiddles for `H` and the coset, `g^i`, `eta_c`; built by the first sub-batch that reaches them).  Results
are reported in name order and the batch reason is the first rejected sub-batch in that order, so the verdict and the
JSON (minus timings) do not depend on scheduling.  Interactive-mode files are still live-verifier-only (D7): with a
`sub_XX.coins` beside the proof the transcript is checked against those coins (`accept (interactive: coins are this
verifier's step-0 coins)`); without one the accept stays conditional (`coins replayed from the transcript`); nothing
about what is checked changes with `--jobs`.  Per-item JSON carries the parsed statement (`l, n, D, t, zk, mode,
n_proofs, n_vus, steps` and the BLAKE2b statement digest) and the summary `batch_accepted`, `batch_reason`, `jobs`,
`shared_seconds` (system parse + pin check), `wall_seconds` (the batch's in-process wall) beside `verify_seconds_sum`
(the sum of per-sub-batch totals, the number the sequential measurements reported).

Producing the files (on a machine with torch): `python -m backends.direct.ligero.run gate-vu --zk --vus 2
--dump-proof X.proof --dump-statement X.stmt` for the honest sub-batches of the gate, or the generators in
`backends/direct/ligero/serialize.py` (`dump-system`, `gen-set`, `gen-small`, `gen-negatives`, `gen-mutations`,
`gen-witness-mutations`, `gen-forged-interactive`; all take `--zk` and `--mode interactive|fiat-shamir`; each writes
`.stmt` + `.proof` + a `manifest.json` with sha256 and the Python verdict).  The Python file verifier
(`serialize.py verify`) replays the transcript's coins in interactive mode -- the same caveat as above.

## Files (`ligero-system/v1`, `ligero-statement/v2`, `ligero-proof/v1`; defined in `backends/direct/ligero/serialize.py`)

| file | what | who writes it |
|---|---|---|
| `system.bin` (`ligero-system/v1`) | the compiled REAL chain constraint system: pins, 347 linear + 3387 quadratic constraints as sparse affine forms, the chain rows, and the Python `system_id` | `serialize.py dump-system` / every generator |
| `X.stmt` (`ligero-statement/v2`) | the verifier's configuration (`l, n, D, t, target, zk, t_pad, mode` byte `0 = interactive / 1 = fiat-shamir`, `n_proofs`) and the public words (`a, b` per unit, `y16` per column; chain layout `n_vus, steps`) | the prover side (`run.py gate-vu --dump-statement`, `serialize.py gen-*`) |
| `X.proof` (`ligero-proof/v1`) | the transcript: coin commitments + openings (interactive mode only; a Fiat-Shamir transcript that carries them is refused), root, `w, h, q [, v]`, the `t` column indices, the opened `M x t` columns, the Merkle paths | the prover side |

The verifier reads the statement first, derives every size from it and the system (`k = l + t_pad`, `h_len = 2k - 1
- l`, `q_len = l + k - 1`, `M = m + 6D` under ZK (`prox, lin, qa, qb, chain_a, chain_b`), depth `log2 n`), and parses the proof against those bounds: a
header that disagrees is `malformed proof` before the body is read; every field element is checked `< p` at parse.
The system file is a verification key: its Python `system_id` **and** this crate's own BLAKE3 digest of the exported
tables are pinned in `verify.rs` (`PINNED_SYS_ID`, `PINNED_TABLE_DIGEST`); a different system is refused unless
`--allow-any-system`.

## What it checks, in order (reject-first, no panics on malformed input)

1. Parameters: Appendix C recomputed from the statement (`k = l + t_pad` under ZK, `l + 1` otherwise as the
   accounting does; `e = floor((n - k)/2)`; challenge field BabyBear^D) per mode -- interactive: plus the step-0
   coin-hiding term `2 * 2^64 / 2^256`; Fiat-Shamir: every statistical term `x 2^60` (the campaign's `q x eps` rule,
   which is what forces `t = 288 / 285`, `D = 7`) -- and union-bounded over `n_proofs` sub-batches; **refused when
   above `2^-soundness_bits`** (default 128). Also `n > 2k` (quadratic test), `k > l + t` under ZK (Lemma 4.15),
   power-of-two `l, n`, the layout.
2. Step 0 (interactive mode): `c_i = BLAKE2b-256("ligero-b/coin|v1|" || r_i || s_i)` for both slots; with `--coins`
   the openings must equal the verifier's own.  Fiat-Shamir mode: the transcript must carry no coins.
3. Public inputs: BF16 decodes of every operand (`s, m, e` pins; an all-ones exponent field is rejected), `y16 <
   2^16` by the format.
4. Statement digest `BLAKE2b-256("ligero-b/statement|v2" | p | hash | l | n | D | t | target | zk | k | mode | sys_id |
   chain layout | a | b | y16)`; challenge 1 -> `r (D x M)`, `rho_lin (D x L)`, `rho_quad (D x Q)`, `rc (D x 7 x l)`
   from SHAKE-256 of a BLAKE3 seed; challenge 2 = the `t` distinct columns by rejection from SHAKE-256(seed || ctr),
   which must equal the proof's column list.  Seeds -- interactive: `BLAKE3("ligero-b/c1|v2|" stmt |coins= c1 c2 |r= r1)`
   and `BLAKE3("ligero-b/c2|v2|" stmt |coins= c1 c2 |r= r2)` (no prover message enters); Fiat-Shamir:
   `BLAKE3("ligero-b/c1|" stmt root)` and `BLAKE3("ligero-b/c2|" stmt root w h q [|v= v])`.
5. Merkle: leaf = BLAKE3(column as `M` u32 LE), node = BLAKE3(left || right), `t` paths against the root.
6. Proximity: `w_d(eta_c) = sum_i r_{d,i} U_i[c]` (+ the prox mask row with coefficient 1 under ZK; `w` are
   coefficients under ZK, evaluations on H otherwise).
7. Linear: `alpha = sum_q rho_q a_q`, `beta = sum_q rho_q (pub_q - c_q)`; ZK: `v|_H = beta` (fold mod `x^l - 1`,
   NTT_l) and `v(eta_c) = alpha . U[c] + Z_H(eta_c) s_lin[c]`; non-ZK: `Enc(beta)[c] = alpha . U[c]`.
8. Quadratic: `sum_q rho_q (A_q B_q - C_q)(U[c])` (+ `Z_H (s_qa + eta^k s_qb)`) `= h(eta_c) Z_H(eta_c)`, all 3387
   constraints evaluated from the CSR tables at each opened column.
9. Chain (the general linear test): the coefficient rows `R_i` on H from `rc` and the layout masks, `l sum_{l | j} q_j =
   r^T b`, and `q(eta_c) = sum_i R_i(eta_c) U_i[c]` (+ `u_ca(eta_c) + eta_c^k u_cb(eta_c)` from the two opened chain
   mask rows; the pre-fix `Z_H s_chain` vanished on H and leaked `q|_H`, red team F1).

Every NTT (D messages, D x R chain rows) and every column loop (Merkle, quadratic) runs on `--threads` OS threads
(`std::thread::scope`, no rayon); `batch` runs whole sub-batches on `--jobs` threads (total concurrency `jobs x
threads`). Stage timings (`shared, parse, statement_hash, challenge_hash, merkle, ntt, dfold, quadratic, chain`) are
in the JSON.

## Byte-level facts the document does not state (DISCREPANCIES.md D5)

* Challenge-1 word order inside each of the `D` slices: `r (M) | rho_lin (L) | rho_quad (Q) | 7 chain families x l`;
  SHAKE-256 output taken as u32 LE words, each `u mod p`.
* Challenge-2 columns: `SHAKE-256(seed || ctr as u32 LE)` read `2t` words at a time, `u mod n`, repeats skipped,
  `ctr += 1` until `t` distinct -- the *ordered* list is what the proof carries and what is compared.
* Merkle leaf `BLAKE3(column as M x u32 LE)`, node `BLAKE3(left || right)`, no leaf/node domain separator (safe
  because `4M != 64` for every system in use).
* Fiat-Shamir seeds: `w`, `h`, `q` are absorbed as raw `D x len` u32 LE arrays; `v` with a `|v=` tag and only under
  ZK.  Interactive seeds: the coins with `|coins=` / `|r=` prefixes after `stmt`, nothing else; tags carry `|v2|`.
* Mask-row order after the `m` witness rows: `prox, lin, qa, qb, chain_a, chain_b`, `D` rows each.
* Statement digest text fields: `|target=` is Python's `repr(float)` (D2); `a`, `b`, `y16` are `<u2` words.
* The evaluation domain is the coset `31 . <w_n>`; the encoder is INTT_l, zero-pad to n, scale coefficient `i` by
  `31^i`, NTT_n.

## Performance notes

Single-thread time on the 4096-VU ZK sub-batch (`l = 16384`, `n = 65536`, `t = 197`, `D = 6`, 5.7 MB, interactive)
is 0.13 s on an idle laptop (0.17 s for the Fiat-Shamir sub-batch: `t = 288`, `D = 7`, 7.4 MB) and 0.33-0.64 s on
the shared `vy-cpu3` EPYC; 60-65% is the message encodes (~30 NTTs of 2^16), 15-25%
the D-fold of the linear constraints. BabyBear multiplication is a Barrett reduction (`field::reduce62`); the
earlier `% p` on a u64 was a 20-40 cycle divide on Zen 3 and dominated there. `--threads` parallelises the encodes over
rows and the column loops; the hashing is serial.  The 25 sub-batch proofs of a 4096-VU batch were 2.6 s (non-ZK
interactive) / 3.3 s (ZK Fiat-Shamir) as a sequential sum at 16 threads each on the h100-bestvsbest H100 pod
(`note:r21-bestvsbest/h100-bestvsbest/20260922T1833Z-report-h100-bestvsbest`); `batch --jobs` measurements are in lane
b-verify-par's note.

## What it does NOT do

* Unit-mode proofs (`compile_unit(chain=False)`, public accumulator per unit): not implemented (the decision workload
  is the chain).
* It does not recompute the constraint system from the checker: the tables come from `system.bin` (pinned digests).
  An error in the *compiler* (a constraint missing from the system) is outside this verifier's reach; `redteam_vu`
  / `redteam_battery` cover that.
* It verifies **soundness only**: acceptance says the transcript proves the statement under the stated parameters --
  unconditionally for Fiat-Shamir files, and *given the coins* for interactive files (D7; use `--coins`).  Nothing
  here checks zero knowledge; the `COMPLETE_ZK_BACKEND` label is lane b-zk / b-zk-fix's claim about the interactive
  session.

## Layout

~~~text
src/hash.rs     BLAKE3 (chunks, parents, root), BLAKE2b-256, SHAKE-256 -- tests against hashlib / spec vectors
src/field.rs    BabyBear, radix-2 NTT (natural in/out), coset points, Z_H, encode / eval_on_coset -- tests vs naive
src/format.rs   ligero-system/v1, -statement/v2, -proof/v1 parsers (statement-bounded, canonical field elements)
src/verify.rs   soundness accounting, statement digest, challenges, Merkle, the four tests, the chain test
src/main.rs     verify / batch / system-digest; agreement with manifest.json's Python verdicts
~~~
