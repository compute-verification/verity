# b-verifier: where PROTOCOL.md, the Python B-Ligero code and the independent Rust verifier disagree

Lane `b-verifier` (track B), 2026-09-22.  The verifier is `backends/ligero-verify/` (`verity-ligero-verify`, no
dependencies: its own BabyBear, NTT / coset code, BLAKE3, BLAKE2b-256, SHAKE-256, `ligero-*/v1` parsers, and the
proximity / linear / quadratic / chain tests), written from `backends/direct/ligero/PROTOCOL.md` + `HANDOFF.md` +
`redteam_vu.md` F1-F3, with the Python sources read as the reference of record wherever the document is silent.
Every such place is an entry below.  Statuses follow `fixtures/discrepancy_log.json` (`open` / `resolved` /
`excluded`); each entry also carries a structured block (`id`, `issue`, `document`, `code`, `verifier`, `status`,
`owner`, `evidence`) so the coordinator can lift it into the campaign log.  None of the entries changes a verdict on
the decision workload (4096 VUs, `l = 16384`): the Rust and Python verifiers agree on every one of the 25 x 3 + 2 x 3
honest proofs, 3 x 52 negatives, 3 x 32-35 transcript mutations and 2 x 8 witness mutations
(`backends/numerical/reports/b_verifier.md`).

## D1. The Python verifier never checks that its parameters meet the target; at small `l` non-ZK they do not

PROTOCOL.md section 6 says the accounting uses `k = l` and that `k = l + 1` "stands in, conservative".  The code does
two different things with that: `config_for` picks `t` from `ligero_queries_for(..., k = l, ...)` (no stand-in), while
`soundness()` -- the number reported in every ledger row -- evaluates the same parameters at `k = l + 1`.  At the
decision workload (`l = 16384`, `t = 196`) the two agree to the second decimal (`2^-128.25` union).  At `l = 256`
(B = 1, non-ZK) the `t = 189` that `config_for` chose gives `(1 - e/n)^t` with `e = (1024 - 257) / 2 = 383`:
`2^-127.73`, below the `2^-128` the statement claims.  `verify()` in `protocol.py` never recomputes this: the
target enters only through the statement digest (`|target=-128.0`) and the choice of `t`.

The Rust verifier recomputes Appendix C from the file's `(n, k, l, t, D, coin_commitment, n_proofs)` exactly as
`soundness()` does (`k = l + t_pad` ZK, `k = l + 1` non-ZK, the coin-hiding term when the commitment is on, union
over `n_proofs`) and refuses when the bound is above `2^-soundness_bits` (default 128).  So on `small_plain/b1` the
verifiers disagree by design: Rust `parameters give 2^-127.73 ...; requested 2^-128`, Python `accept`.  With
`--soundness-bits 127.5` Rust accepts it too (`fixtures/rust_verdicts/small_plain_127.5.json`).  Every negative and
mutation of the `l = 256` non-ZK set is likewise refused on parameters first; the named-reason agreement for those
sets was taken at `--soundness-bits 127.5` (`neg_plain_127.5.json`, `mut_plain_127.5.json`: 84/84 agree).

~~~json
{"id": "BV-D1", "issue": "config_for chooses t at k = l; soundness() reports at k = l + 1; verify() checks neither against the target. At l = 256 non-ZK the chosen t = 189 gives 2^-127.73 by the reported formula.",
 "document": "PROTOCOL.md section 6: 'k = l (no ZK padding; validate() needs l < k, so k = l + 1 stands in, conservative)'",
 "code": "protocol.py config_for (k = l) vs soundness() (k = l + 1); verify() has no parameter check",
 "verifier": "recomputes the k = l + 1 bound from the statement and refuses below --soundness-bits (default 128)",
 "status": "open", "owner": "b-zk (decide which k the claim is made at; make the prover choose t at that k)",
 "evidence": ["fixtures/rust_verdicts/small_plain.json", "fixtures/rust_verdicts/small_plain_127.5.json", "run r20260922-100145-374a"]}
~~~

## D2. The statement digest binds a Python `repr(float)`

`statement_digest` absorbs `f"...|target={cfg.target_log2!r}"`: the target enters the challenges as Python's
shortest-round-trip float repr (`-128.0`, `-80.0`).  PROTOCOL.md section 8c gives the digest only as `stmt` and does not
fix its byte layout at all -- the tag `ligero-b/statement|v2`, the `|p=|hash=blake3|l=|n=|D=|t=|target=` text, `|zk=|k=`,
`|sys=` + the 32-byte `system_id`, `|mode=chain|l=|n_vus=|steps=`, `|a=|b=|y16=` as `<u2` are all from the code.  A
language-neutral verifier has to reproduce a language-specific number format.  The Rust verifier uses `{:?}` (which
agrees with Python for every target a statement can carry, e.g. `-128.0`, `-133.02`) and *refuses* a target whose
repr would take exponent form rather than guess Python's `1e-05` / `1e+16` spelling.  Recommendation: absorb the
target as an integer of milli-bits (or the IEEE bytes), and write the digest layout into PROTOCOL.md.

~~~json
{"id": "BV-D2", "issue": "statement_digest byte layout is undocumented and includes repr(float) of the target",
 "document": "PROTOCOL.md 8c names `stmt` only", "code": "protocol.py statement_digest / STATEMENT_TAG",
 "verifier": "reproduces the layout from the code; refuses exponent-form targets", "status": "open",
 "owner": "b-zk (document; consider an integer encoding)", "evidence": ["src/verify.rs statement_digest, py_float_repr"]}
~~~

## D3. HVZK-mode proofs carry coin commitments and openings that nothing binds

`_COMMIT_COINS` is a module global (`protocol.py`, "red-team control ONLY").  When it is `False` (the campaign's
HVZK mode, `--no-coin-commitment`) the prover still draws `r_i, s_i`, computes `c_i`, and the transcript still carries
all six 32-byte values; `_coin_prefix` / `_coin_open` return `b""` so **none of them enters any challenge**, and the
Python verifier ignores them.  The document (8c) presents the commitment as the ZK-mode step and says nothing about
the HVZK transcript.  Consequence: 192 bytes of the HVZK proof are free for the prover -- harmless for soundness, but a
verifier that read the header's "coins present" flag as "coins are committed" would be wrong, and a mutation of them is
accepted by both verifiers (`mut_hvzk`: `coin_*` cases, 4 of 35 accepted by both -- the agreement holds, the mutation
is not a forgery).  The Rust verifier takes the *mode* from the verifier's statement (`coin_commitment`), requires the
coins when it is on, and parses-but-ignores them when it is off, exactly like Python; the proof-file header flag is
checked for consistency only.

~~~json
{"id": "BV-D3", "issue": "HVZK proofs contain coin commitments/openings bound to nothing",
 "document": "PROTOCOL.md 8c: commitment is the ZK step; HVZK transcript unspecified",
 "code": "protocol.py _COMMIT_COINS global; prover emits coins in every mode; verifier ignores them when off",
 "verifier": "mode from the statement; coins required when on, parsed and ignored when off", "status": "open",
 "owner": "b-zk (either drop the coins from the HVZK transcript or absorb them)", "evidence": ["fixtures/rust_verdicts/mut_hvzk.json"]}
~~~

## D4. The Python verifier is interactive; a file verifier checks a Fiat-Shamir transcript

`verify()` refuses `coins are not this verifier's step-0 coins`: it compares the proof's `(r_i, s_i)` with the ones it
drew itself.  A verifier reading a proof from disk has no coins of its own, so the coin *values* are prover-supplied
data as far as it is concerned.  What it can and does check is what the document's soundness argument actually uses:
`c_i = BLAKE2b(tag || r_i || s_i)`, the `c_i` absorbed before the root and the `r_i` absorbed after the message of
their round, both derived challenges recomputed.  With prover-chosen coins the transcript is a plain Fiat-Shamir
transcript over `stmt || c1 c2 || root || r1` and `... || w h q [v] || r2`, and its soundness is the FS soundness of
the same hash chain.  This is a scoping statement, not a bug: the malicious-verifier ZK claim (b-zk's
`COMPLETE_ZK_BACKEND`) is about the *interactive* session and is not something this lane verifies -- it verifies
soundness only.

~~~json
{"id": "BV-D4", "issue": "the transferable-proof verifier cannot check 'coins are this verifier's coins'; it checks the FS chain",
 "document": "PROTOCOL.md 8c (interactive, V draws r_i)", "code": "protocol.py verify(): coins compared to its own",
 "verifier": "checks c_i = H(tag||r_i||s_i) and both challenge derivations; treats the file as a Fiat-Shamir transcript",
 "status": "resolved", "owner": "b-verifier", "resolution": "documented scope: soundness of the transcript, not the ZK property",
 "evidence": ["README.md 'What it does NOT do'"]}
~~~

## D5. Things the document does not say that a verifier needs (taken from the code, now written down)

Each of these had to be read from `protocol.py` / `chain.py` / `field.py` / `merkle.py`; none contradicts the document,
but an implementer working from PROTOCOL.md alone cannot produce interoperable challenges without them.

* Challenge 1 is `SHAKE-256(BLAKE3("ligero-b/c1|" || stmt || coins || root || r1))` expanded to `D x (M + n_lin +
  n_quad + 7 l)` words, each reduced `u mod p` (section 8c only names the inputs); the order of the coefficient blocks
  inside a `D`-slice is `r (M rows) | rho_lin (n_lin) | rho_quad (n_quad) | 7 chain families x l`.
* Challenge 2's column set: `SHAKE-256(seed || ctr_u32_le)` streams of `2t` u32 words, `u mod n`, repeats rejected,
  `ctr` incremented per call until `t` distinct indices -- the *ordered* set is the transcript's `cols` and must be
  reproduced exactly (a permutation of the same columns is rejected: `columns_swapped`).
* The Merkle leaf is `BLAKE3(column as M x u32 LE)` over the *committed* rows `M = m + 5D` (ZK) / `m` (non-ZK); the
  node is `BLAKE3(left || right)`; the tree is over `n` leaves with no domain separation between leaf and node hashes
  (a second-preimage-style leaf/node confusion is prevented only by the leaf length `4M != 64` for every system in use --
  worth a domain separator).
* The ZK message `v` is absorbed with a `|v=` tag; `w`, `h`, `q` are absorbed raw (`tobytes()` of `D x len` u32 LE
  arrays); a non-ZK transcript has no `|v=` at all.  Message lengths: `w` is `D x k` (ZK) / `D x l`; `h` is `D x h_len`,
  `q` is `D x q_len` (both from the configuration, see `serialize.py` for the formulas the file carries).
* The evaluation domain is the coset `31 . <w_n>` (`g = 31`, BabyBear's generator; the encoder is INTT_l, zero-pad,
  scale coefficient `i` by `31^i`, NTT_n), and the chain's public
  `y16` is bound in the digest as `<u2` words, the operands `a, b` as `<u2` BF16 words with the decode
  `sign / 7-bit mantissa / 8-bit exponent`, exponent `255` (inf/NaN) refused as "off the finite range" before any hash.
* The `k = l + 1` stand-in (D1) also fixes `e = floor((n - k)/2)`, so the non-ZK proximity bound is evaluated one
  message symbol more conservatively than the code's own encoder uses.

~~~json
{"id": "BV-D5", "issue": "challenge expansion order, column-sampling loop, leaf/node hashing, message tags and lengths, coset shift and public-word encodings are code-only",
 "document": "PROTOCOL.md sections 4, 8c name the inputs, not the byte layouts", "code": "protocol.py _challenge1/_challenge2, merkle.py, field.py, chain.py",
 "verifier": "implemented from the code; written down here and in README.md", "status": "open",
 "owner": "b-zk (fold into PROTOCOL.md)", "evidence": ["backends/ligero-verify/README.md", "src/verify.rs"]}
~~~

## D6. A transcript mutation never reaches the tests

Every mutation of a *message* (`w`, `h`, `q`, `v`), the root, the statement, or the column set is rejected at
"column challenge mismatch" -- the Fiat-Shamir binding catches it before the proximity / linear / quadratic / chain
tests run; the opened values fail at the Merkle path.  That is the correct behaviour, but it means a mutation battery
alone exercises the tests only through the honest proofs' acceptance and the 52 negatives' `chain constraints failed
(sum over H)`.  This lane therefore added `serialize.py gen-witness-mutations` (the prover's `mutate=` hook, before
commitment: one row per row class, a broken chain link, a nonzero `c_0`, a flipped bit column) so that both verifiers
are seen rejecting *inside* the tests: 8/8 `linear constraints failed` / `quadratic constraints failed` in both
modes, same reason in both verifiers (`fixtures/rust_verdicts/wmut_*.json`).  Not a discrepancy; recorded so the
next reader does not mistake the mutation table for test coverage.

~~~json
{"id": "BV-D6", "issue": "transcript mutations are caught by the FS binding, not the tests; test coverage needs witness mutations",
 "document": "n/a", "code": "n/a", "verifier": "gen-witness-mutations added; 16/16 rejected inside the tests by both",
 "status": "resolved", "owner": "b-verifier", "resolution": "witness-mutation set is part of the gate", "evidence": ["fixtures/rust_verdicts/wmut_zk.json", "fixtures/rust_verdicts/wmut_plain.json", "run r20260922-101053-0dc9"]}
~~~

## Not findings (checked and consistent)

* `redteam_3.md` O1 (`gpu/ligero.py::row_coeffs` geometric powers): the transcript path this lane verifies
  (`protocol.py`, torch prover) draws every combination coefficient independently from SHAKE, as section 4 says
  ("`D` *independent* BabyBear combinations"); the Rust verifier would reject a prover that used geometric powers
  (the coefficient rows it recomputes would differ).  If the GPU kernel path ever produces transcripts, this verifier
  is the check.
* `redteam_vu.md` F1-F3: every size (`M, l, n, t, D, k, depth, n_vus, steps`) comes from the verifier's statement;
  the proof header's copies are compared and a disagreement is a named rejection; the statement digest covers
  `a, b, y16`, the configuration and the `system_id` -- confirmed by the statement mutations (`statement_*`: 7/7
  rejected in every mode).
* `redteam_3.md` O5 (`N_subbatches` not in the ledger schema): the statement file carries `n_proofs` and the
  verifier's bound is the union over it; the ledger rows below spell out `25 sub-batches` in `note`.
