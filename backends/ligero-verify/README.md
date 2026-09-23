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
ligero-verify batch  --system system.bin --dir DIR [--jobs N] [--threads T] [--soundness-bits 128] [--reps R] [--json out.json]
                     # every DIR/*.proof with its .stmt (+ its .coins when present, = `verify --coins` for that proof), the
                     # sub-batches verified concurrently on N jobs (default: available parallelism); batch ACCEPT iff every
                     # sub-batch accepts; agreement with DIR/manifest.json's python_verdict when present
ligero-verify system-digest --system system.bin     # sys_id, table digest and the relation they are pinned for (or null)
cargo test --release                                 # hash vectors (hashlib / BLAKE3 spec), NTT / coset code, Barrett mul,
                                                     # the operand decodes; tests/fixture.rs: the B=1 ZK interactive fixture
                                                     # accepted, byte flips in every region rejected, a flipped statement word
                                                     # rejected, --coins checked, batch verdict + parsed statements identical at
                                                     # --jobs 1 / 8, one corrupted sub-batch rejects the batch with the same error
                                                     # at any --jobs; tests/relations.rs: the FP8 Ada v4 fixture accepted against
                                                     # its pinned system, every system file named by system-digest, cross-relation
                                                     # pairs refused by the pin, --allow-any-system flagged, v4 negatives rejected
~~~

**Relations and pins.**  The statement names the relation it is about: `ligero-statement/v4` carries a `relation`
string (`fp8-ada`, `fp8-hopper`, `bf16-hopper`), a v2 / v3 file is the Ampere BF16 unit (`bf16-ampere`).  For each
relation `src/relation.rs` pins the compiled constraint system by two digests -- the Python `system_id` (the last
32 bytes of `system.bin`) and this crate's own BLAKE3 digest of the exported tables -- and mirrors what the Python
side fixes outside `system.bin` (`protocol.RelationHooks`): the digest tag, the byte widths the words are absorbed
with, the range of the chain-end word, the operand decode.  A `system.bin` whose digests are not the statement's
relation's pair is refused (`system file is the pinned bf16-ampere system, the statement is a fp8-ada statement`;
`... is not the pinned fp8-ada system (sys_id ..., table digest ...)`).  `--allow-any-system` lifts the pin, and then
EVERY verdict carries `"system_pinned":false` (and an accept reason ending in `[system NOT pinned: --allow-any-system;
sys_id ...]`): such an accept says the proof is valid for the system file, not that the file is the relation's system
-- the BF16 Ampere and Hopper relations, for instance, share tag and word widths and are told apart by the pin alone.

| relation | statement | words (`a, b` / `y16`) | operands | `sys_id` (prefix) | table digest (prefix) | source of the digests |
|---|---|---|---|---|---|---|
| `bf16-ampere` | v2 / v3 (or v4 named `bf16` / `bf16-ampere`) | u16 / u16, `y16 < 2^16` | BF16, K = 16 | `44cb05b9e698…` | `c147cc4cece7…` | `compile_unit(REAL, chain=True)`, main aecc298 = `fixtures/system.bin` = every A100 / 4090 dump |
| `fp8-ada` | v4 (tag `\|rel=fp8-ada\|v1`) | u8 / u32, `y16 < 2^22` | E4M3, K = 32 | `6b570eef0c7f…` | `443e4fc74c0a…` | `compile_fp8_unit(ADA, chain=True)`, main aecc298 and lane b-hopper a39de82 (byte-identical) = `proofs/system.bin` of the ada-fp8-dumps pod runs (sha256 `cba9a09f…`) |
| `bf16-hopper` | v4 (tag empty) | u16 / u16, `y16 < 2^16` | BF16, K = 16 | `9dcc7cdc1377…` | `74ce68a2801e…` | `relations.relation("bf16-hopper").compile(params, True)`, lane b-hopper a39de82 (`wide.py` adder; 3292 rows); no pod `system.bin` in the store yet |
| `fp8-hopper` | v4 (tag `\|rel=fp8-hopper\|v1`) | u8 / u32, `y16 < 2^22` | E4M3, K = 32 | `c7cbebe365cb…` | `d5b77463fee9…` | `relations.relation("fp8-hopper").compile(params, True)`, lane b-hopper a39de82 (3396 rows); no pod `system.bin` in the store yet |

The compilations are torch-free and were run twice each on the laptop (byte-identical files); `system-digest` prints
both digests of any `system.bin` and names the relation they are pinned for.  The full values are in `src/relation.rs`.

Exit status: `verify` 0 accept, 1 reject (the reason on stdout), 2 usage / unreadable operator input.  `batch`: 0 every
sub-batch accepted, 1 a sub-batch rejected (`batch_reason` names the first one in name order), 2 a sub-batch file
unreadable, 3 a verdict disagrees with `manifest.json`'s Python verdict (checked first: the agreement gate).  Malformed
*proof or statement bytes* never panic (every read is bounded and every element checked); an unreadable path given on
the command line is reported and exits 2.

**Batch mode (lane b-verify-par).**  A 4096-VU batch is `N_subbatches` (25) independently committed proofs, each with
its own statement (its VUs' `a, b, y16`; `n_proofs` = 25 for the union bound) -- there is no batch-level statement and
no check that spans sub-batches (PROTOCOL.md 8c; `bench-vu` dumps them as `repK/sub_XX.{stmt,proof[,coins]}`).  `batch`
therefore verifies them on `--jobs` OS threads pulling from one queue, each sub-batch exactly as `verify` would
(its own statement, proof, challenges, checks, `--threads` inside), and accepts the batch iff every sub-batch accepts.
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

## Files (`ligero-system/v1`, `ligero-statement/v2..v4`, `ligero-proof/v1..v2`; defined in `backends/direct/ligero/serialize.py`)

| file | what | who writes it |
|---|---|---|
| `system.bin` (`ligero-system/v1`) | a relation's compiled chain constraint system: pins, linear + quadratic constraints as sparse affine forms (347 + 3387 for the Ampere BF16 unit, 589 + 3532 for FP8 Ada), the chain rows, and the Python `system_id` | `serialize.py dump-system` / every generator / `bench-vu --dump-dir` |
| `X.stmt` (`ligero-statement/v2`) | the verifier's configuration (`l, n, D, t, target, zk, t_pad, mode` byte `0 = interactive / 1 = fiat-shamir`, `n_proofs`) and the public words (`a, b` per unit, `y16` per column; chain layout `n_vus, steps`) as u16 -- the Ampere BF16 unit | the prover side (`run.py gate-vu --dump-statement`, `serialize.py gen-*`) |
| `X.stmt` (`ligero-statement/v3`) | v2 + the authentication block (`authentication = included`, D9) | `run.py ... --auth` |
| `X.stmt` (`ligero-statement/v4`) | the v2 header through `K`, then `u8 word_bytes \| u8 y_bytes \| str relation \| a \| b \| y16` at those widths `\| u8 has_auth [\| the v3 block]`: the relation-named statement (`fp8-ada`: 1-byte operands, 4-byte slots for the 22-bit chain-end words) | `run.py --relation fp8-ada bench-vu --dump-dir` |
| `X.proof` (`ligero-proof/v1`) | the transcript: coin commitments + openings (interactive mode only; a Fiat-Shamir transcript that carries them is refused), root, `w, h, q [, v]`, the `t` column indices, the opened `M x t` columns, the Merkle paths | the prover side |
| `X.proof` (`ligero-proof/v2`) | v1 + the three Merkle multiproofs of an authenticated statement (D9) | the prover side |

The verifier reads the statement first, derives every size from it and the system (`k = l + t_pad`, `h_len = 2k - 1
- l`, `q_len = l + k - 1`, `M = m + 6D` under ZK (`prox, lin, qa, qb, chain_a, chain_b`), depth `log2 n`), and parses the proof against those bounds: a
header that disagrees is `malformed proof` before the body is read; every field element is checked `< p` at parse.
The system file is a verification key: its Python `system_id` **and** this crate's own BLAKE3 digest of the exported
tables must be the pair `src/relation.rs` pins for the statement's relation (table above); a different system is
refused unless `--allow-any-system`, which flags every verdict `system_pinned = false`.  A v4 statement whose word
widths are not its relation's, or that carries an authentication block over words that are not 16-bit (the leaf
schema is `u16`; Python defines no other), is refused at parse.

## What it checks, in order (reject-first, no panics on malformed input)

1. Parameters: Appendix C recomputed from the statement (`k = l + t_pad` under ZK, `l + 1` otherwise as the
   accounting does; `e = floor((n - k)/2)`; challenge field BabyBear^D) per mode -- interactive: plus the step-0
   coin-hiding term `2 * 2^64 / 2^256`; Fiat-Shamir: every statistical term `x 2^60` (the campaign's `q x eps` rule,
   which is what forces `t = 288 / 285`, `D = 7`) -- and union-bounded over `n_proofs` sub-batches; **refused when
   above `2^-soundness_bits`** (default 128). Also `n > 2k` (quadratic test), `k > l + t` under ZK (Lemma 4.15),
   power-of-two `l, n`, the layout.
2. Step 0 (interactive mode): `c_i = BLAKE2b-256("ligero-b/coin|v1|" || r_i || s_i)` for both slots; with `--coins`
   the openings must equal the verifier's own.  Fiat-Shamir mode: the transcript must carry no coins.
3. Public inputs: the relation's decode of every operand into the `s, m, e` pins (BF16: an all-ones exponent field
   is rejected; E4M3: the NaN codes `0x7F / 0xFF` are rejected, the all-ones exponent field is finite), the
   statement's `K` equal to the operands the system pins, `y16 < 2^y_bits` (16 for BF16, 22 for the packed FP32
   claim of the FP8 relations).
4. Statement digest `BLAKE2b-256("ligero-b/statement|v2" | relation tag | p | hash | l | n | D | t | target | zk | k |
   mode | sys_id | chain layout | a | b | y16)`, the words at the relation's widths (`<u2` for BF16; `<u1` operands
   and `<u4` chain-end words for FP8, exactly as `protocol._u_bytes` writes them; the tag is empty for the BF16
   relations, `|rel=fp8-ada|v1` / `|rel=fp8-hopper|v1` for FP8);
   challenge 1 -> `r (D x M)`, `rho_lin (D x L)`, `rho_quad (D x Q)`, `rc (D x 7 x l)`
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
* Statement digest text fields: `|target=` is Python's `repr(float)` (D2); `a`, `b`, `y16` are `<u2` words for the
  BF16 relations and `<u1` / `<u1` / `<u4` for the FP8 relations (`RelationHooks.word_dtype / y16_dtype`, D10); the
  relation tag (`RelationHooks.tag`) follows the statement tag immediately and is not in any file.
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
* It does not recompute the constraint system from the checker: the tables come from `system.bin` (pinned digests,
  one pair per relation).  An error in the *compiler* (a constraint missing from the system) is outside this
  verifier's reach; `redteam_vu` / `redteam_battery` cover that.  Relations this build does not pin (a v4 statement
  naming anything but `bf16[-ampere]`, `fp8-ada`, `bf16-hopper`, `fp8-hopper`) are refused: their digest tag and
  word widths are not in the file, so nothing could be checked.
* It verifies **soundness only**: acceptance says the transcript proves the statement under the stated parameters --
  unconditionally for Fiat-Shamir files, and *given the coins* for interactive files (D7; use `--coins`).  Nothing
  here checks zero knowledge; the `COMPLETE_ZK_BACKEND` label is lane b-zk / b-zk-fix's claim about the interactive
  session.

## Layout

~~~text
src/hash.rs     BLAKE3 (chunks, parents, root), BLAKE2b-256, SHAKE-256 -- tests against hashlib / spec vectors
src/field.rs    BabyBear, radix-2 NTT (natural in/out), coset points, Z_H, encode / eval_on_coset -- tests vs naive
src/format.rs   ligero-system/v1, -statement/v2..v4, -proof/v1..v2 parsers (statement-bounded, canonical field elements)
src/relation.rs the pinned relations: sys_id + table digest, digest tag, word widths, y range, operand decode (BF16 / E4M3)
src/auth.rs     authentication = included: the SHA-256 merkle frame, leaves, multiproof folds (D9)
src/verify.rs   soundness accounting, statement digest, challenges, Merkle, the four tests, the chain test, the pin gate
src/main.rs     verify / batch / system-digest; agreement with manifest.json's Python verdicts (DIR or DIR/..)
~~~
