# b-verifier: where PROTOCOL.md, the Python B-Ligero code and the independent Rust verifier disagree

Lane `b-verifier` (track B), 2026-09-22.  The verifier is `backends/ligero-verify/` (`verity-ligero-verify`, no
dependencies: its own BabyBear, NTT / coset code, BLAKE3, BLAKE2b-256, SHAKE-256, `ligero-*/v1` parsers, and the
proximity / linear / quadratic / chain tests), written from `backends/direct/ligero/PROTOCOL.md` + `HANDOFF.md` +
`redteam_vu.md` F1-F3, with the Python sources read as the reference of record wherever the document is silent.
Every such place is an entry below.  Statuses follow `fixtures/discrepancy_log.json` (`open` / `resolved` /
`excluded`); each entry also carries a structured block (`id`, `issue`, `document`, `code`, `verifier`, `status`,
`owner`, `evidence`) so the coordinator can lift it into the campaign log.

**Post b-zk-fix (lane `b-verifier-2`, `main` a52441c).**  The protocol changed under the verifier (PROTOCOL.md 8b/8c:
masked `q`, coin-only challenges in `--mode interactive`, `t = 288 / t_pad = 512 / D = 7` in `--mode fiat-shamir`);
the verifier was re-derived from the new text and re-gated on fresh proof sets in all four modes (ZK / non-ZK x
interactive / Fiat-Shamir).  D1 re-checked (still open, both modes), D3 resolved by the fix, D4 superseded by **D7**
(an interactive-mode transcript is not transferable evidence: 49/49 forged files for false claims accepted by both
file verifiers), D5 extended with the new byte facts, D8 (named-reason differences) added.  On the decision workload
(4096 VUs, `l = 16384`, 25 sub-batches) the two verifiers agree on every file in every mode
(`backends/numerical/reports/b_verifier.md`, "post b-zk-fix").

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
`--soundness-bits 127.5` Rust accepts it too (`fixtures/rust_verdicts/small_*_plain_127.5.json`).  Every negative and
mutation of the `l = 256` non-ZK set is likewise refused on parameters first; the named-reason agreement for those
sets was taken at `--soundness-bits 127.5` (`mut_*_plain_127.5.json`).

**Post b-zk-fix: NOT resolved.**  `config_for` now walks `D` up until `_field_terms_ok` holds (that is what gives
`D = 7` in Fiat-Shamir mode) but still sizes `t` at `k = l` for non-ZK while `soundness()` reports at `k = l + 1`.
Re-evaluated with the new rules (`/tmp` replica of `config_for` + `soundness`, and the Rust verifier on the fresh
files): interactive non-ZK `l = 256`: `t = 189`, `2^-127.73` (MISS); Fiat-Shamir non-ZK `l = 256`: `t = 278`,
`2^-127.88` (MISS, `small_fiat-shamir_plain/b1`); every ZK configuration and every `l >= 8192` configuration meets
the target (`2^-128.05` interactive ZK, `2^-128.04` FS ZK, `2^-128.25` / `2^-128.59` non-ZK at 25 x 170).  The gap
is the one-symbol `k` mismatch and only bites where `(1 - e/n)^t` is the dominant term at small `n`.

~~~json
{"id": "BV-D1", "issue": "config_for chooses t at k = l; soundness() reports at k = l + 1; verify() checks neither against the target. At l = 256 non-ZK the chosen t gives 2^-127.73 (interactive, t = 189) / 2^-127.88 (Fiat-Shamir, t = 278) by the reported formula. Unchanged by b-zk-fix.",
 "document": "PROTOCOL.md section 6: 'k = l (no ZK padding; validate() needs l < k, so k = l + 1 stands in, conservative)'",
 "code": "protocol.py config_for (k = l) vs soundness() (k = l + 1); verify() has no parameter check",
 "verifier": "recomputes the k = l + 1 bound from the statement and refuses below --soundness-bits (default 128)",
 "status": "open", "owner": "b-zk (decide which k the claim is made at; make the prover choose t at that k)",
 "evidence": ["fixtures/rust_verdicts/small_interactive_plain.json", "fixtures/rust_verdicts/small_fiat-shamir_plain.json", "fixtures/rust_verdicts/small_*_plain_127.5.json", "runs r20260922-100145-374a, r20260922-111453-609b, r20260922-111531-e2d3"]}
~~~

## D2. The statement digest binds a Python `repr(float)`

`statement_digest` absorbs `f"...|target={cfg.target_log2!r}"`: the target enters the challenges as Python's
shortest-round-trip float repr (`-128.0`, `-80.0`).  PROTOCOL.md section 8c gives the digest only as `stmt` and does not
fix its byte layout at all -- the tag `ligero-b/statement|v2`, the `|p=|hash=blake3|l=|n=|D=|t=|target=` text, `|zk=|k=|mode=`
(the `|mode=interactive` / `|mode=fiat-shamir` suffix is new with b-zk-fix; the document says "absorbs the mode", not how),
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

## D3. HVZK-mode proofs carry coin commitments and openings that nothing binds -- RESOLVED by b-zk-fix

**Post b-zk-fix.**  `_COMMIT_COINS` is gone; the mode is a `Config` field bound into the statement digest, `prove`
refuses coins in Fiat-Shamir mode and `verify` rejects a Fiat-Shamir transcript that carries them ("fiat-shamir
transcript must not carry coins").  The Rust verifier does the same from the statement's mode byte
(`mut_fiat-shamir_*/coins_planted_in_fs_transcript`, `statement_mode_flipped`: both verifiers reject, same reason).
The text below is the pre-fix record.

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
 "verifier": "mode from the statement byte; coins required in interactive mode, refused in fiat-shamir mode", "status": "resolved",
 "owner": "b-zk-fix", "resolution": "Config.mode bound into the digest; the two modes are mutually exclusive in prove/verify (main a52441c)",
 "evidence": ["fixtures/rust_verdicts/mut_fiat-shamir_zk.json (coins_planted_in_fs_transcript, statement_mode_flipped)"]}
~~~

## D4. The Python verifier is interactive; a file verifier checks a Fiat-Shamir transcript -- SUPERSEDED by D7

**Post b-zk-fix.**  The reading below ("with prover-chosen coins the transcript is a plain Fiat-Shamir transcript
over `stmt || c1 c2 || root || r1` ...") was true of the pre-fix derivation, in which `root` and `w, h, q, v` entered
the hashes.  It is FALSE after F2: the interactive-mode challenges are functions of `(stmt, c1, c2, r_i)` alone, so a
transcript whose coins the prover chose has no soundness at all as a file -- see D7.  Fiat-Shamir mode (`--mode
fiat-shamir`) is now the transferable protocol and the one this entry's argument applies to.

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
 "status": "excluded", "owner": "b-verifier-2", "resolution": "superseded by BV-D7 after b-zk-fix F2: the FS-chain reading no longer holds for interactive-mode files; it holds for --mode fiat-shamir files",
 "evidence": ["README.md 'What it does NOT do'", "BV-D7"]}
~~~

## D5. Things the document does not say that a verifier needs (taken from the code, now written down)

Each of these had to be read from `protocol.py` / `chain.py` / `field.py` / `merkle.py`; none contradicts the document,
but an implementer working from PROTOCOL.md alone cannot produce interoperable challenges without them.

* Challenge 1 is `SHAKE-256(BLAKE3(seed1))` expanded to `D x (M + n_lin + n_quad + 7 l)` words, each reduced
  `u mod p` (section 8c only names the inputs); the order of the coefficient blocks inside a `D`-slice is
  `r (M rows) | rho_lin (n_lin) | rho_quad (n_quad) | 7 chain families x l`.  **Post b-zk-fix** the seed preimages
  are (from `protocol.py seed1_preimage / seed2_preimage`, not the document, which writes `H("ligero-b/c1|v2" ||
  stmt || c1 c2 || r1)` without the separators): interactive `b"ligero-b/c1|v2|" || stmt || b"|coins=" || c1 || c2 ||
  b"|r=" || r1` and `b"ligero-b/c2|v2|" || stmt || b"|coins=" || c1 || c2 || b"|r=" || r2`; Fiat-Shamir
  `b"ligero-b/c1|" || stmt || root` and `b"ligero-b/c2|" || stmt || root || w || h || q [|| b"|v=" || v]` -- the
  pre-8c tags without a version bump, and no coin bytes at all.  The mode is not in the seed except through `stmt`.
* **Post b-zk-fix, mask rows.** Six kinds in chain mode, in this order after the `m` witness rows, `D` rows each:
  `prox, lin, qa, qb, chain_a, chain_b` (`M = m + 6D`); 8b lists the kinds but not the order.  The chain check is
  `q(eta_c) = sum_i R_i(eta_c) U_i[c] + U_ca[c] + eta_c^k U_cb[c]` with `k = l + t_pad` -- the same `eta^k` the
  quadratic test uses for `x^k s_qb`; 8b writes it that way, the code agrees.  `u_ca` has degree `< k`, `u_cb`
  degree `< l - 1`, so `u = u_ca + x^k u_cb` has degree `< l + k - 1 = q_len` and needs no new shape check; the
  zero-sum condition is enforced by the existing `sum_H q = rhs` check (the verifier never sees `sum_H u`).
* **Post b-zk-fix, Fiat-Shamir parameters.** `t_pad = 512` (ZK) makes `k = l + 512 > 2l` at `l = 256`: the fold
  `v mod (x^l - 1)` wraps a coefficient more than once (`x^j -> x^(j mod l)`), which the pre-fix verifier (written
  for `t_pad = 256 <= l`) got wrong -- fixed here; the document does not mention `l < t_pad` configurations.
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
{"id": "BV-D5", "issue": "challenge expansion order, column-sampling loop, leaf/node hashing, message tags and lengths, coset shift, public-word encodings, and (post b-zk-fix) the seed separators, mask-row order and the |mode= digest field are code-only",
 "document": "PROTOCOL.md sections 4, 8b, 8c name the inputs, not the byte layouts", "code": "protocol.py seed1_preimage/seed2_preimage, _MASK_KINDS, statement_digest, merkle.py, field.py, chain.py",
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

**Post b-zk-fix.**  Still true in both modes, with a twist in interactive mode: `root` no longer enters any
challenge, so `root_flipped` now reaches the Merkle check ("merkle path 0 invalid") instead of "column challenge
mismatch", and a mutated message (`w`, `h`, `q`, `v`) reaches ITS test (`proximity test failed`, `quadratic
constraints failed`, `chain constraints failed`, `linear constraints failed`) -- the interactive transcript is not
hash-bound, only coin-bound (D7).  A consistent re-opening of one coin slot (`challenge{1,2}_coin_reopened_consistently`:
fresh `r_i, s_i`, `c_i := H(r_i || s_i)`) changes `c1 c2`, which sits in BOTH seeds, so it is caught at the column
challenge in every case; no mutation can change challenge 1 alone.  The mask rows are covered by
`opened_mask_row_{prox,chain_a,chain_b}_plus1` (Merkle) and by the witness mutations (tests).

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

## D7. An interactive-mode transcript is not transferable evidence (post b-zk-fix; the file verifier's verdict is conditional)

PROTOCOL.md 8c after b-zk-fix: "the challenges are functions of the verifier's coins alone ... there is no
non-interactive reading of this mode ... `verify(coins=own)` is the only verifier".  A verifier reading a proof from
disk has no coins of its own.  The only thing it can do is **replay the transcript's openings** as if they were its
coins -- which is what `serialize.verify_files` (Python) and `ligero-verify` without `--coins` (Rust) do -- and then
the coins are prover-supplied: the prover knew `r, rho, rc` and `cols` before committing `root`.  With `cols` known
in advance, a tableau that passes every column test at exactly those `t` columns exists for ANY public statement;
b-zk-fix's own simulator (`redteam/simulator.py`, written to show zero knowledge) is precisely that construction.

Demonstrated (`serialize.py gen-forged-interactive`, run r20260922-112419-ec4f): the simulator run as a cheating
prover with `Coins.sample()`d coins, on honest VU operands with the claimed word flipped and on the 52 `vu-k1536-neg`
claims (48 emitted; 4 have off-domain operands the simulator itself refuses at `public_vectors`).  **49/49 forged
files are accepted by the Python file verifier and by the Rust verifier without `--coins`**, with the real BLAKE3 /
BLAKE2b / SHAKE (no oracle programmed), in 13 ms each; the same 49 are rejected by `verify(coins=Coins.sample())`
and by `ligero-verify --coins <other>` ("coins are not this verifier's step-0 coins").  Fiat-Shamir mode has no such
object: its challenges depend on `root` and the messages, and the 25 + 2 + 52 + 36 + 8 FS files behave as before.

Consequences and what this lane did about it:

* The Rust verifier prints `accept` for Fiat-Shamir files and **`accept (interactive: coins replayed from the
  transcript; sound only if they were the verifier's step-0 coins)`** for interactive files; with `--coins FILE`
  (the verifier's own `r1 || s1 || r2 || s2`, 128 bytes) it checks the transcript against them and prints
  `accept (interactive: coins are this verifier's step-0 coins)`.  The accept-rate rows in the ledger say which.
* The agreement matrix on interactive-mode files is an agreement about *transcript consistency with the carried
  coins*; the honest-proof acceptances in that mode are evidence that the verifier reproduces the protocol, not that
  the files prove anything to a third party.  Every soundness claim that reaches the morning report from a *file* must
  come from `--mode fiat-shamir` (t = 288 / 285, D = 7, `2^-128.04` / `2^-128.59` at 25 x 170) or from a live session.
* Not a bug in b-zk-fix: the document says exactly this.  It is a finding about how B-Ligero proofs can be *used*
  (the ledger's `COMPLETE_ZK_BACKEND` row is an interactive session; a stored transcript of it certifies nothing),
  and a trap for any archive / audit pipeline that stores `bench-vu --mode interactive` transcripts as proofs.

~~~json
{"id": "BV-D7", "issue": "interactive-mode transcripts verify against their own carried coins; a prover that chose the coins knows r and cols before committing, and the b-zk-fix simulator run as a cheating prover produces accepted files for false statements (49/49, real hash)",
 "document": "PROTOCOL.md 8c: 'no non-interactive reading of this mode; verify(coins=own) is the only verifier'",
 "code": "protocol.py verify(coins=...) requires the caller's coins; serialize.verify_files replays the transcript's",
 "verifier": "accept wording is conditional without --coins; --coins FILE checks the verifier's own step-0 randomness; the report and ledger rows say which mode each accept-rate is about",
 "status": "open", "owner": "coordinator / b-zk-fix (label stored interactive transcripts as non-transferable; archive Fiat-Shamir files or the session's coins with the transcript)",
 "evidence": ["fixtures/rust_verdicts/forged_interactive_zk.json", "fixtures/manifests/forged_interactive_zk.manifest.json", "run r20260922-112419-ec4f", "tests/fixture.rs interactive_mode_own_coins_are_checked"]}
~~~

## D8. Named-reason differences that are not verdict differences (post b-zk-fix)

Verdicts agree on every file; three families of *reasons* differ and are recorded so nobody reads them as a
disagreement later.  (a) Coin openings: Python (with the transcript's coins replayed as its own) says "coins are not
this verifier's step-0 coins" for a flipped `r1` / `s2` / `c1`, because its first check compares `H(r_i || s_i)`
against `coins.commitments()`; Rust says "coin slot i: opening does not match its commitment".  Same check, named
from the other side.  (b) Malformed shapes: Python says "malformed proof" / "malformed Merkle paths" / "bad column
set"; Rust names the header field and the statement's value (`header t = 188, the statement fixes 189`, `column
index 2048 >= n`, `repeated column index`).  (c) Off-domain operands: Python "public inputs rejected: a.t off the
finite range", Rust "operand exponent field off the finite range".  The comparison in `ligero-verify batch` is on
the verdict; the reasons are listed side by side in `fixtures/rust_verdicts/*.json`.

~~~json
{"id": "BV-D8", "issue": "reason strings differ for coin-opening, shape and operand-domain rejections", "document": "n/a",
 "code": "protocol.py verify reasons", "verifier": "src/verify.rs / src/format.rs reasons", "status": "excluded", "owner": "b-verifier-2",
 "resolution": "verdict agreement is the gate; reasons recorded side by side", "evidence": ["fixtures/rust_verdicts/mut_*.json"]}
~~~
