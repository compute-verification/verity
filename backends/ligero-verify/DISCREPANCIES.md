# b-verifier: where PROTOCOL.md, the Python B-Ligero code and the independent Rust verifier disagree

Lane `b-verifier` (track B), 2026-09-22.  The verifier is `backends/ligero-verify/` (`verity-ligero-verify`, no
dependencies: its own BabyBear, NTT / coset code, BLAKE3, BLAKE2b-256, SHAKE-256, `ligero-*/v1` parsers, and the
proximity / linear / quadratic / chain tests), written from `backends/direct/ligero/PROTOCOL.md` + `HANDOFF.md` +
`note:r20-proof/red-team-vu/20260922T1042Z-report-redteam-vu` F1-F3, with the Python sources read as the reference of record wherever the document is silent.
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
(`note:r20-proof/b-verifier/20260922T1132Z-report-b-verifier`, "post b-zk-fix").

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

* `note:r20-proof/red-team-3/20260922T0921Z-report-redteam-3` O1 (`gpu/ligero.py::row_coeffs` geometric powers): the transcript path this lane verifies
  (`protocol.py`, torch prover) draws every combination coefficient independently from SHAKE, as section 4 says
  ("`D` *independent* BabyBear combinations"); the Rust verifier would reject a prover that used geometric powers
  (the coefficient rows it recomputes would differ).  If the GPU kernel path ever produces transcripts, this verifier
  is the check.
* `note:r20-proof/red-team-vu/20260922T1042Z-report-redteam-vu` F1-F3: every size (`M, l, n, t, D, k, depth, n_vus, steps`) comes from the verifier's statement;
  the proof header's copies are compared and a disagreement is a named rejection; the statement digest covers
  `a, b, y16`, the configuration and the `system_id` -- confirmed by the statement mutations (`statement_*`: 7/7
  rejected in every mode).
* `note:r20-proof/red-team-3/20260922T0921Z-report-redteam-3` O5 (`N_subbatches` not in the ledger schema): the statement file carries `n_proofs` and the
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

**The live session (lane `live-verifier`, `backends/direct/ligero/live.py`).**  The verifier the document names is now a
process on another machine: `python -m backends.direct.ligero.live serve` plays 8c against `bench-vu --verifier
tcp://host:port`.  Per sub-batch it samples the coins itself (`Coins.sample()`, `os.urandom`) and sends `c1, c2`
BEFORE the prover's first message; it receives the statement (STMT, sent at step 0) and the root, opens slot 1,
receives `(w, h, q, v)`, opens slot 2, receives the `ligero-proof/v1` bytes on a second connection; it refuses -- before
any cryptography -- a transcript whose coin commitments / openings are not the ones it generated for that sub-batch,
whose root is not the ROOT it received before opening slot 1, or whose test message is not the TESTS it received before
opening slot 2; then **this** Rust verifier decides: `ligero-verify verify --coins <own>` per sub-batch on the files it
wrote (`<session>/sub_NN.{stmt,proof,coins}`) and `ligero-verify batch --target-bits` on the session directory at END
(the union bound, D10).  The transcript is byte-identical to the local-coins path with the same coins (the coin
*source* moved, not the protocol; `live_test.py`), so the session directory is an ordinary dump that `ligero-verify
batch` re-verifies -- and now the `.coins` files beside the proofs are the verifier's, so "accept (interactive: coins
are this verifier's step-0 coins)" is what it prints, and the verdict is transferable *as this verifier's statement*
(`verification-verdict/v1`, `independent = true`, `verifier = live-verifier@<sha>`, `mode = interactive`), which is what
D7 asks for.  The prover's own result still only reports the verdicts as it received them; the `verified=` label comes
from the verifier's record (`live.py record`).  Measured (RTX 4090 in EUR-IS-2 -> CPU pod in EU-RO-1, 70 ms RTT,
fp8-ada int-ZK, 4096 VUs as 49 sub-batches): the prover's own time is unchanged (`t.total` ~1.0-1.7 s), the session adds
the two openings per sub-batch -- 98 round trips at the path's RTT, ~8 s of `net.wait_seconds`, `t.total_live` ~9.6 s --
and the verifier spends ~0.09 s wall / CPU per sub-batch (one `ligero-verify` process each, 6 in parallel) plus ~1.3 s
wall for the session's `batch`.  Until the sockets were configured before the handshake the same session cost ~12 s:
under the pods' default bbr the kernel paces every connection for life, and the protocol's bursty, application-limited
flows (one 0.5 MB test message, one 3.7 MB proof per sub-batch) never grow a window past one burst, so each opening
cost ~2 RTTs (`live.py`, `_TCP_CC`).

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

## D9. `authentication = included` (lane `auth-included`): a third hash, two new file versions, and the check order

The included result binds every published boundary word to three SHA-256 indexed Merkle trees of
`verity.commitments` (`backends/direct/ligero/auth.py`: a-words, b-words, output words of the frozen instance set,
one `u16` leaf per word, `RangeIndexedDomain(0, count)`, owner `-1 / -2 / -1`, binding = `identity_digest` of the
dataset manifest + word range).  Facts a verifier needs that PROTOCOL.md does not state (the code is the reference):

* **Hash.**  The commitments layer is SHA-256 under the `veritor/protocol/merkle/frame/v3` framing (`_hash`: frame
  tag, `u32be(len tag) || tag`, then `u64be(len part) || part` per part) and `veritor/tagged-sha256/v1` for the domain
  identity (canonical JSON `{"start":0,"step":1,"stop":N}`).  Neither BLAKE3 nor BLAKE2b appears in it, so the Rust
  verifier now carries a third primitive (`src/hash.rs Sha256`, FIPS 180-4, pinned against NIST vectors) and
  `src/auth.rs` mirrors the frame, the `domain / leaf / pad / node` frames and the minimal-big-endian integer
  encoding.  The tree's `domain_id` includes the *range identity digest as 32 raw bytes* (`bytes.fromhex`), not the
  hex text.  The B-Ligero column tree is unchanged (BLAKE3, D5).
* **Files.**  `ligero-statement/v3` (`LIGSTM03`) = the v2 body + `str version | str schema | u32 n_vus | n_vus x u32
  vu_index | 3 x (32 B binding | u8 owner + 2 | u64 count | 32 B root)`; `ligero-proof/v2` (`LIGPRF02`) = the v1 body +
  `3 x (u32 n_siblings | n_siblings x 32 B)`.  A statement without the block is still written as v2 and a proof
  without multiproofs as v1 (byte-identical to before); the Rust parser accepts both pairs and refuses a v2 proof
  against a v2 statement and a v1 proof against a v3 statement ("transcript carries multiproofs but the statement
  has no authentication block" / "... carries no multiproofs").  The multiproof lengths are bounded before reading
  by `min(opened leaves, 2^depth) x depth` and `VerificationLimits.max_proof_bytes` (64 MiB); the block's `count` by
  `max_positions` (10^7) and the openings by `max_openings` (10^7), the same defaults `auth.py` enforces.
* **Digest.**  `statement_digest` absorbs `auth.digest_bytes` after `|y16=`: `|auth=ligero-b/auth/v1|schema=u16|vus=`
  + the VU indices as `<u4`, then per tree `|<a|b|y>:binding=<32 B>|owner=<decimal>|count=<decimal>|root=<32 B>`
  (340 bytes for two VUs; pinned in both test suites).  So the Fiat-Shamir challenges depend on the roots and the
  indices, and a proof for the same words under another root or at another index is a different statement
  ("column challenge mismatch" in FS mode; in interactive mode the challenges are coin-only (D7) and the change is
  caught by the multiproof check instead).
* **Check order.**  Both verifiers run the Ligero tests first and the authentication check last, so an `auth:`
  rejection means "a valid B-Ligero proof over words that are not the committed ones" -- that is the negative the
  lane's prover-side battery produces (a swapped operand word, honest words at a shifted `vu_index`, honest words
  under a flipped root).  Leaf ranks: A/B word `o` of column `j` (step `i = j mod steps` of chain `v = j div steps`)
  is leaf `vu_index[v] * K + 16 i + o`; the chain's output word (`y16` at the chain-end column) is leaf `vu_index[v]`
  of the y tree.  The pairing of the fold is the library's (`multiproof._fold`: sorted known nodes, an even node
  hashed with a known right neighbour, otherwise with the next proof sibling; the proof must be consumed exactly).
* **Reasons (D8 family).**  Python: `auth: <tree>: multiproof rejected`; Rust: `auth: <tree>: multiproof rejected
  (root mismatch)` / `(multiproof too short)` / `(multiproof too long)`.  Shape rejections of the block are named in
  Rust at parse ("authentication block claims leaves beyond the committed count") and in Python inside `verify_auth`
  ("auth: a: leaf index out of range").  Verdicts agree; the gate is the verdict.
* **Timing.**  `timing.auth` (Rust JSON) is the leaf recomputation + the three folds; it is `0` for an excluded
  statement.  The Python verifier reports the same as `auth_timing["seconds"]` (`protocol.verify(auth_timing=)`).

~~~json
{"id": "BV-D9", "issue": "authentication = included needs SHA-256 + the veritor merkle frame (a third hash in the verifier), the v3 statement / v2 proof layouts, the digest_bytes layout and the leaf-rank convention; none is in PROTOCOL.md",
 "document": "PROTOCOL.md 12-14 (statement binding) say nothing about committed roots", "code": "backends/direct/ligero/auth.py, serialize.py (LIGSTM03 / LIGPRF02), protocol.py statement_digest(auth=)",
 "verifier": "src/hash.rs Sha256; src/auth.rs frame / fold / check; format.rs v3 / v2 parsers with statement-bounded lengths; check runs after the Ligero tests",
 "status": "open", "owner": "auth-included (fold the layouts into PROTOCOL.md section 12)",
 "evidence": ["src/auth.rs tests (pinned against verity.commitments)", "tests/test_ligero_auth.py (same constants)", "lane note auth-included"]}
~~~

## D10. No verifier enforced the union bound across sub-batches -- RESOLVED by b-batch-bound

Found by lane `b-verify-par` (`ligero-verify batch --jobs`), fixed by lane `b-batch-bound` (`main` 5a89d9d).  A
4096-VU batch is `N_subbatches` = 25 independently committed proofs.  Each proof's soundness is a per-commitment
number (`soundness()`'s `per_proof_total_log2`, 2^-132.9 at `t = 196`); the claim "the batch is sound to 2^-128"
rests on the union bound `sum_i 2^-b_i` over the 25, which `config_for(..., n_proofs = 25)` sizes `t` for and
`bench-vu` *reports* (`security.achieved_log2` = `union_over_proofs`, 2^-128.26 -- a report, not a check).  Neither
verifier checked it.  `ligero-verify batch` accepted iff every sub-batch accepted, each gated on ITS OWN
`union_log2 = per_proof_log2 + log2(n_proofs)` -- with `n_proofs` read from the proof's statement header, a
prover-supplied field that is not in the statement digest (only `l, n, D, t, target, zk, k, mode` and the words are).
The Python paths (`bench-vu`, `serialize verify`) had no batch-level check at all.  Consequences: 25 proofs each
claiming `n_proofs = 1` at 2^-132.9 passed while the batch was only 2^-128.26 (the number that saved it is a
coincidence of the same `t`); 25 proofs sized for a 25-batch presented as a 26-batch would have passed at 2^-128.2
claimed as 2^-128; and the `l = 256` fixture (`t = 189`, 2^-128.16, sized for one proof) copied six times was the
"honest batch" of the b-verify-par determinism tests, a batch that is 2^-125.57.

**Fix (PROTOCOL.md 8c, "Batch verification").**  A batch verifier with target `t` bits accepts iff (1) every sub-batch
proof verifies against its own statement with the shared system, (2) `sum_i 2^-b_i <= 2^-t` with `b_i` the
verifier's own recomputation of proof `i`'s bound from that proof's `(n, k, l, t, D, mode, zk)`, (3) every proof's
`n_proofs` equals the number `N` presented.  Implemented as `protocol.batch_bound` (Python; `ChainRunner.batch_bound`
/ `verify_batch` in `bench-vu`, `serialize verify-batch --dir DIR --target-bits T` on files) and
`verify::batch_bound` (Rust; `ligero-verify batch --target-bits T`, default 128, `--soundness-bits` its alias; the
per-proof gate of `verify` is off in batch mode because the batch gate is the right one).  Same reason strings, same
precedence (unreadable input, a rejected sub-batch, `batch: n_proofs mismatch (<name>: n_proofs = k, N sub-batches
presented)`, `batch: union bound 2^-x over N sub-batches exceeds the target 2^-t`); the accept reason and the JSON
carry the batch bits (`batch_bits = -log2 sum`; `per_proof_bits[]`, `n_proofs_claimed[]`).  `bench-vu`'s
`security.achieved_log2` is now the verifier's `batch_log2` (asserted equal to `soundness()`'s union to 1e-6), with
`achieved_bits` and `verifier_target_bits` beside it; `N_subbatches` and `B_proof` were already in the fingerprint.
Rule (2) needs nothing prover-supplied: `b_i` is a function of digest-bound parameters, so relabelling `n_proofs`
cannot move the bound -- it can only fail rule (3).  The check is arithmetic on `N` numbers inside the batch wall
(re-timed at `--jobs 25` on the B = 4096 batch before/after: lane note `b-batch-bound`).  Both verifiers agree on the
batch bits to 1e-6 and on every verdict and reason of the test matrix (honest 4-batch accepted at 128; the same
proofs rejected at a target above the achieved bound; four `n_proofs = 1` proofs presented together rejected with
the mismatch string in both).  D1 is unchanged: a single `protocol.verify` still checks the transcript only.

~~~json
{"id": "BV-D10", "issue": "the batch verdict was 'every sub-batch accepts'; the union bound over the sub-batches presented and n_proofs = N were never checked by any verifier (n_proofs is a prover-supplied header field outside the statement digest)",
 "document": "PROTOCOL.md 8c (before b-batch-bound): the union bound stated for the prover's sizing, no batch verification rule",
 "code": "vu.py bench_vu asserted each sub-batch and REPORTED the union; serialize.py had per-file verify only",
 "verifier": "main.rs batch: accept iff every sub-batch accepts, each gated on its own n_proofs",
 "status": "resolved", "owner": "b-batch-bound",
 "resolution": "PROTOCOL.md 8c 'Batch verification'; protocol.batch_bound / verify::batch_bound, --target-bits (default 128), same reason strings; tests/fixture.rs the_batch_union_bound_and_n_proofs_are_enforced, tests/test_ligero_batch_bound.py",
 "evidence": ["b-verify-par verifier results labelled finding=batch-union-bound-unenforced-before-<commit>", "lane note b-batch-bound (test matrix, batch bits, timing before/after)"]}
~~~

## D11. Relations beyond the Ampere BF16 unit (lane `ligero-verify-relations`): what a statement does not say, and the per-relation pin

`protocol.verify` takes a `RelationHooks` (`word_dtype`, `y16_dtype`, `y16_bits`, `tag`, `public_vectors`) and the
FP8 relations (`fp8/`, `relations.py` on lane b-hopper) supply their own; the statement file carries the words and,
from `ligero-statement/v4` on, their byte widths and the relation's name -- but not the tag, not the range of the
chain-end word, not the operand format, and nothing that says which compiled system the words are about.  Facts a
verifier needs (the code is the reference; PROTOCOL.md still describes the BF16 unit only):

* **Digest.**  `statement_digest` absorbs `rel.tag` immediately after `ligero-b/statement|v2` (`b""` for `bf16` and
  `bf16-hopper`, `b"|rel=fp8-ada|v1"`, `b"|rel=fp8-hopper|v1"`), then the words as `numpy.astype(dtype).tobytes()`
  with the RELATION's dtypes (`<u2 / <u2` BF16; `<u1 / <u4` FP8).  The widths in a v4 file are the writer's and are
  not re-checked by the Python reader: a word wider than the relation's dtype would be silently truncated in the
  digest (never produced by an honest writer).  The Rust verifier pins the widths per relation and refuses a file
  whose widths differ ("word widths 2/4 bytes are not the fp8-ada relation's 1/4"), so what it absorbs is always what
  it parsed.
* **Which system.**  Only the `sys_id` inside the digest ties a statement to a compiled system, and the BF16 Ampere
  and Hopper relations have the same tag and widths: a v4 statement named `bf16-hopper` over Ampere words verifies
  against the Ampere `system.bin` byte for byte.  The pin is therefore per relation (`src/relation.rs`:
  `relation -> (sys_id, table digest)`, chosen by the statement's name; v2 / v3 = `bf16-ampere`) and a mismatch is a
  rejection that names both sides ("system file is the pinned bf16-ampere system, the statement is a bf16-hopper
  statement").  `--allow-any-system` lifts it and every verdict then carries `system_pinned = false` -- an unpinned
  accept is a statement about the system file, not about the relation.  The digests: laptop compilations from the
  checked-in Python (main aecc298 for `bf16-ampere` / `fp8-ada`; the lane b-hopper tree at a39de82 for `bf16-hopper` /
  `fp8-hopper`, whose `relations.py` / `wide.py` are not on main), twice each, byte-identical; `fp8-ada` also equal to
  the pod `system.bin` of every ada-fp8-dumps run.  The Hopper pair awaits its first pod dump (`h100-dumps-v1`).
* **Public decode.**  E4M3 (`public_vectors_fp8`): `t = (w >> 3) & 15`, `f = w & 7`, `m = f + [t != 0] 8`,
  `e = t - [t != 0] - 6`, `s = w >> 7`; the NaN codes `0x7F / 0xFF` are rejected, the all-ones exponent field is
  finite (448 = `0x7E`) -- unlike BF16, where `t = 255` is rejected.  The chain-end word is the FP32 accumulator with
  its ten always-zero low bits dropped (`pack_public`: `s 2^21 + t 2^13 + f >> 10`, 22 bits), checked `< 2^22`
  (`y16_bits`).  `K` (operands per unit: 16 / 32) is read from the system's `a[i]` pins and the statement's `K` must
  equal it (`public_vectors*` rejects the shape mismatch).
* **Files.**  `LIGSTM04` = the v2 header through `K`, `u8 word_bytes | u8 y_bytes | str relation`, the words at those
  widths, `u8 has_auth`, an optional v3 block.  The authentication leaf schema is `u16`, so the Rust parser refuses a
  v4 block over words that are not 16-bit ("authentication block (leaf schema u16) is defined for 16-bit words
  only"); Python defines nothing else (`fp8/chain.py` always reports `authentication = excluded`).  Lane b-hopper's
  earlier files wrote this body under the `LIGSTM03` magic (before v3 was taken by the authentication block): the FP8
  Ada dumps of attempts r20260922-233507-7fb9 / -233617-4316 / -233711-6b9f (code 7bb0ffa1, manifest `statement:
  ligero-statement/v3`, `relation: fp8-ada`) are such files.  They are refused by the v3 parser ("statement file:
  truncated file": 2-byte words read over 1-byte ones, then no block) and are NOT disambiguated by a heuristic; the
  re-dumps at a4b0ac7 (r20260922-235542-295b, -235643-98f7, -235746-0929) are v4 and verify.
* **Reasons (D8 family).**  Python: `public inputs rejected: E4M3 NaN operand`; Rust: `public inputs: E4M3 NaN
  operand`.  Python rejects a v4 statement naming an unknown relation in `_runner_for` ("no runner for relation");
  Rust: "statement names relation ...; this build pins bf16-ampere, fp8-ada, bf16-hopper, fp8-hopper".  Verdicts agree
  on every FP8 Ada dump in the store (441 / 441 sub-batch proofs, three modes).

~~~json
{"id": "BV-D11", "issue": "a statement does not name its compiled system, its digest tag, its word ranges or its operand format; the BF16 Ampere and Hopper relations are indistinguishable except by sys_id; the v4 widths are trusted by the Python reader",
 "document": "PROTOCOL.md describes the BF16 unit only; serialize.py's docstring defines LIGSTM04", "code": "backends/direct/ligero/protocol.py RelationHooks / statement_digest, serialize.py read_statement, fp8/relation.py public_vectors_fp8, lane b-hopper relations.py",
 "verifier": "src/relation.rs (per-relation tag, widths, y_bits, operand decode, pinned sys_id + table digest); format.rs v4 parser (widths 1/2/4, has_auth 0/1, u16-only auth block); verify.rs gate per statement, system_pinned in every verdict",
 "status": "open", "owner": "b-merge-h100 (name the relation in PROTOCOL.md; confirm the bf16-hopper / fp8-hopper digests against a pod system.bin when h100-dumps-v1 lands)",
 "evidence": ["tests/relations.rs", "fixtures/fp8-ada (sub-batch 0 of art:c5dff3c6..., pinned accept)", "lane note ligero-verify-relations"]}
~~~

## D12. The v2 public-selection relations (lane `relmin-lookup`): the verifier computes part of the step, and the pins are no longer the operand words

`bf16-hopper-v2`, `bf16-ampere-v2`, `fp8-hopper-v2`, `fp8-ada-v2` (`backends/direct/ligero/pubsel/relation.py`,
registered in `relations.py` beside the v1 names, v1 untouched) claim exactly what their v1 counterparts claim -- the
same pinned instruction semantics (`verity.ml.tc.models`), the same statement widths, the same chain-end word
(`y16`, one public word per VU; `ligero-system/v1` files) -- with a different compiled unit: 292 / 469 / 195 / 357 rows
against 3292 / 3516 / 3396 / 3769.  What changes for a verifier:

* **The pins are derived values, not word components.**  A v1 system pins `a[i].s|m|e` / `b[i].s|m|e` and the
  witness re-does the whole step; a v2 system pins, per product group `g` of the step, the group maximum `g{g}.G` of
  the products' exponents (a zero product counts as `Params.floor`; `G >= floor`) and, for every extra alignment
  shift `j in [0, prod_bits)`, the signed sum `g{g}.Q[j]` of the products each aligned to `G + j` -- `AlignProduct`'s
  truncation term by term, then summed (`public_pins_int`; `PubSel::unit_pins`).  The BF16 Hopper sum exceeds p and
  is pinned as two limbs `g0.Q[j].lo` (the low 16 bits) and `g0.Q[j].hi` (the arithmetic shift; negative for a
  negative sum), reduced mod p like every pin.  The verifier therefore performs the public half of the tensor-core
  step itself (products, alignment, group maximum) -- O(K prod_bits) integer operations per unit -- and the prover's
  witness selects `Q[E - G]` with a lookup one-hot.  A wrong pin value anywhere fails the linear test with
  overwhelming probability, as before; a wrong *rule* in the verifier's public computation is a soundness bug of the
  verifier, not of the proof -- the Rust decode is pinned against `public_pins_int` by hand-computed vectors
  (`pubsel_unit_pins_match_public_pins_int`) and, on the Python side, by the v1-v2 differential
  (`pubsel/relation_test.py`: both units accept exactly the model's word on random units and reject every negative
  family).
* **K comes from the relation, not from the pins.**  There are no `a[i]` pins to read `K` off; `K` is the sum of
  `Params.groups` (16 / 8+8 / 32 / 16+16), fixed per relation in `relation.rs` (`PubSel::k`), and a statement whose
  `K` differs is refused ("statement has K = 32 operands per unit, the relation's step takes 16").  Every word of every
  column is decoded, pad columns included (all-zero operands are on the domain), so an off-domain word anywhere is
  refused with the decode's reason (Python: `public inputs rejected: ...`; Rust: `public inputs: ... (unit j)`).
* **Tags and pins.**  `bf16-hopper-v2` and `bf16-ampere-v2` absorb the empty tag like the other BF16 relations (the
  statement's name and the `sys_id` in the digest distinguish them, D11); the FP8 pair absorb `|rel=fp8-hopper-v2|v1`
  / `|rel=fp8-ada-v2|v1`.  Pinned digests (`sys_id` / table digest): `066c6696... / e9b1b7b4...`,
  `2a5f4378... / 2ce5db09...`, `2cb40f21... / c214a87c...`, `11a7194b... / 215e1beb...` -- laptop compilations
  from the lane tree, torch-free, twice each, byte-identical, the `sys_id`s equal to `protocol.system_id` under
  torch; fixtures under `fixtures/systems/`.  The units select the pinned sum, the accumulator's aligned term and the
  normalised significand without product rows (a hint row and one row-free quadratic `sel_i (X - value_i) = 0` per
  candidate), and the two shift hints are ranged by their lookups' one-hots alone (`LigeroCtx.lookup(range_key=False)`).
  The last group of a unit commits no fraction bits: its normalised significand, hidden bit and the BF16 epilogue's
  fields are picked from the sum's bit rows by the normalisation one-hot, with the bits above the window forced to zero.
  A zero sum (the terms cancel, or every term is zero: `z = 1`) is the zero word whatever normalisation shift the one-hot
  picked, so the subnormal-clamp and overflow proofs read the exponent masked by `z` (`e1m = acc_e_min` when `z = 1`) --
  the first cut of the unit demanded `t = tc(E)` for `z = 1` too and had no witness for a cancelling sum whose `E` put
  `tc(E)` outside the shift range (found by the 1e5 differential on fp8-ada-v2; `test_zero_sums_accepted_in_unit_and_chain_mode`).
* **`bf16-ampere-v2` is a new claim on the generic runner**, not a re-encoding of `vu.py`'s `--relation bf16` run:
  it proves the Ampere BF16 step over synthetic instances drawn like the Hopper BF16 set (`instances_dataset`
  `bench-instances-bf16-ampere/v1`), where `bf16-ampere` proves the frozen `bench-instances/v1` set through `vu.py`.
  The unit compiled is `compile_unit(Params.from_model(AMPERE_BF16_M16N8K16))`'s claim; the statement digest and
  the auth block differ from `vu.py`'s files.

~~~json
{"id": "BV-D12", "issue": "the v2 relations pin values the verifier computes from the operand words (group maximum, truncated product sums per shift); K is fixed by the relation; the verifier's public computation is part of the soundness surface",
 "document": "PROTOCOL.md describes the BF16 unit only; pubsel/relation.py's module comment is the reference",
 "code": "backends/direct/ligero/pubsel/relation.py (compile_v2_unit, public_pins_int, public_vectors_v2), relations.py (the four names)",
 "verifier": "src/relation.rs pubsel::PubSel (unit_pins, pin_index, the four parameter sets), Decode::PubSel; verify.rs public_pins_pubsel",
 "status": "open", "owner": "relmin-lookup (pin the digests against a pod system.bin when the fp8-ada-v2 dumps land; PROTOCOL.md section for the public-selection unit)",
 "evidence": ["src/relation.rs tests", "tests/relations.rs system_digest_names_every_pinned_relation", "backends/direct/ligero/pubsel/relation_test.py", "lane note relmin-lookup"]}

## D13. `authentication = included-hash` (lane `hash-relation`): the operands leave the statement, Poseidon2 enters the relation

PROTOCOL.md 8d describes the construction; these are the facts a verifier needs that only the code fixes, and where the two
verifiers' behaviour differs from the `included` mode of D9.

* **The statement carries no operand words.**  A v5 statement (`LIGSTM05`) has a `(l, 17)` public table -- `hash.is_end` (1 at
  the last unit of a VU, 0 elsewhere: derived from the layout, never read from the file) and 16 digest lanes (canonical field
  elements, refused if `>= p`) -- plus the `l` y words of v4 and the block `str "ligero-b/auth/v2h" | str "poseidon2-babybear-w24/row/v2h"
  | 32 B Poseidon2 constants digest | u32 n_vus | n_vus x (u32 vu_index, x_index, w_index) | 3 tree refs (D9 layout) | 3 multiproofs`.
  The header's K is `steps x operands per unit` (words per committed row), where v2..v4 carried operands per unit; the parser
  divides.  `word_bytes` is 4 for the digest lanes; the relation's own word width (`y_bytes`) still gates the y words.  The
  relation name is the same as the bare relation's ("fp8-ada"), so **only the pin tells a hashed statement from a bare one**:
  `relation.rs` carries `hashed_sys_id / hashed_table_digest` beside `sys_id / table_digest`; a v5 statement against the bare
  system and a v4 statement against the hashed system are refused ("system file is the pinned fp8-ada system, the statement
  is a hashed fp8-ada statement", and the converse).  bf16-ampere and fp4-nvf4 have no hashed system ("has no pinned hashed
  system").
* **Digest.**  `statement_digest` absorbs, through the relation tag, `hashauth.digest_bytes` (version, schema, the constants
  digest, the (vu, x, w) index rows as `<u4`, the three tree refs) -- the tag becomes `<relation tag>|hashed|<block bytes>`
  -- and, in the body, the 17 public columns as 4-byte words and the y words.  Consequences: a changed digest lane, index, root
  or binding is a different statement (in either mode the transcript's challenges no longer match: "column challenge
  mismatch"), while the multiproof **siblings are auxiliary** and a wrong path is caught only by the path check ("auth: a:
  multiproof rejected (root mismatch)").  D9's order stands: Ligero tests first, the path check last; the Poseidon2 constants
  digest and the leaf ranges are checked at parse.
* **What the path check hashes.**  Leaves `a` / `b` are the published digests re-encoded as 32 big-endian bytes (8 lanes x 4)
  under schema `poseidon2-babybear-w24/row/v2h`; the y leaves are D9's word leaves at the relation's width.  Ranks: VU v's x row
  is leaf `x_index[v]` of tree `a`, its W column leaf `w_index[v]` of tree `b`, its output word leaf `vu_index[v]` of tree `y`.
  Several VUs may claim one row (a GEMM tile): the check folds each distinct rank once and refuses two VUs publishing
  different digests for one rank ("VUs claiming leaf r published different digests").  The verifier never runs Poseidon2 --
  the relation did -- so the Rust verifier gains no fourth primitive; `auth.rs` reuses D9's SHA-256 frame.
* **The system file is `ligero-system/v3`** (`LIGSYS03`): the chain block names `n_links` linked rows (19 here) instead of the
  fixed three; the coefficients of links 3.. are the monomials `u_(e mod 6) * u_6^(e div 6 + 1)` of the seven existing families
  (`chain.py extra_coef`; `verify.rs` mirrors it).  Both verifiers accept v1 / v2 / v3.
* **Cost asymmetry.**  The Python verifier (GPU) pays +45-75 % on a hashed proof; the Rust verifier pays 4x CPU (22 s vs 5.5 s per
  4096 fp8-ada VUs at 13 jobs), because its chain test encodes one coefficient row per *involved* row and the capacity links'
  Y expressions involve ~80 rows instead of ~10 (`ntt` 1.0 s of 1.49 s single-threaded).  One hot spot is fixed in this lane
  (resolving the Y terms per column, 1.05 s -> 0.24 s); materialising the sponge outputs as rows would halve the rest.
* **Reasons (D8 family).**  Python: `auth: <tree>: multiproof rejected`, `auth: <tree>: leaf index out of range`, `authentication
  block does not match the statement's indices / layout`, `public table: is_end / digests off the chain ends`; Rust: `auth: a|b|y:
  multiproof rejected (root mismatch)`, `Poseidon2 parameters differ from the verifier's`, `hash-authentication block claims
  leaves beyond the committed count`, `v5 statement names an authentication other than included-hash`.  Verdicts agree on all
  13 pod dumps (python agreement 13/13, 25/25); the gate is the verdict.

~~~json
{"id": "BV-D13", "issue": "included-hash: operands private, digests public; the statement digest binds the block but not the siblings; the bare and hashed relations share a name and are told apart by the pin only; v3 system files with n_links != 3; the Rust chain test scales with the involved rows",
 "document": "PROTOCOL.md 8d (this lane)", "code": "backends/direct/ligero/hashchain.py, hashauth.py, serialize.py (LIGSTM05 / LIGSYS03), chain.py extra_coef, relchain.HashedRelationRunner",
 "verifier": "src/format.rs v5 / v3 parsers; src/relation.rs hashed pins; src/verify.rs hashed digest, pins, n-link chain test; src/auth.rs check_hashed",
 "status": "open", "owner": "hash-relation (materialise cap_out rows; store n_vus y words in the statement; compose bf16-ampere / fp4-nvf4)",
 "evidence": ["tests/relations.rs fp8_ada_hashed_*", "src/auth.rs tests::hashed_fixture_block_folds_and_every_mutation_is_refused", "fixtures/fp8-ada-hash", "lane note hash-relation (13 pod dumps, Rust batch on each)"]}
~~~
