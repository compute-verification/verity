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

**Resolved (lane `ligero-hygiene`, 2026-09-25).**  The claim is made at the stand-in: `config_for` now sizes `t` and
`D` at `k = l + 1` for non-ZK, the `k` that `soundness()` and ligero-verify evaluate.  Over 595 configurations
(both modes, ZK and not, targets -80 / -100 / -128, 1 to 50 sub-batches, `l = 2^6 .. 2^15`) `soundness()` now meets the
per-proof target everywhere (it missed 108 before).  ZK configurations and every `l >= 8192` configuration keep their
`(t, D, n)`; small-`l` non-ZK ones gain a column or two (`l = 256` interactive at 2^-128: `t = 189 -> 190`).  The
first visible casualty was the 2x4 shared-pair live test at 2^-100 (G at `l = 512`, `t = 149`: `2^-99.86` union).

~~~json
{"id": "BV-D1", "issue": "config_for chooses t at k = l; soundness() reports at k = l + 1; verify() checks neither against the target. At l = 256 non-ZK the chosen t gives 2^-127.73 (interactive, t = 189) / 2^-127.88 (Fiat-Shamir, t = 278) by the reported formula. Unchanged by b-zk-fix.",
 "document": "PROTOCOL.md section 6: 'k = l (no ZK padding; validate() needs l < k, so k = l + 1 stands in, conservative)'",
 "code": "protocol.py config_for (k = l) vs soundness() (k = l + 1); verify() has no parameter check",
 "verifier": "recomputes the k = l + 1 bound from the statement and refuses below --soundness-bits (default 128)",
 "status": "resolved", "owner": "b-zk (decide which k the claim is made at; make the prover choose t at that k); resolved by ligero-hygiene: config_for sizes t and D at k = l + 1",
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

Resolved: the mode is a `Config` field bound into the statement digest; `prove` refuses coins in Fiat-Shamir mode and both
verifiers reject a Fiat-Shamir transcript that carries them ("fiat-shamir transcript must not carry coins";
`statement_mode_flipped`).  Full history and evidence: `note:r20-proof/coordinator/20260923T0929Z-finding-ligero-verify-resolved-discrepancies`.

## D4. The Python verifier is interactive; a file verifier checks a Fiat-Shamir transcript -- SUPERSEDED by D7

Superseded by D7 after b-zk-fix F2 (`BV-D4`: status excluded, owner b-verifier-2).  Full text and block: this file at `cc885b2`.

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

Resolved (b-verifier): transcript mutations are caught by the transcript binding, not by the tests, so the witness-mutation set
(`gen-witness-mutations`, 16/16 rejected inside the tests by both verifiers) is part of the gate (`fixtures/rust_verdicts/wmut_*.json`).
Post b-zk-fix, in interactive mode `root` enters no challenge: a mutated message reaches ITS test and `root_flipped` the Merkle
check; a consistent coin re-opening changes `c1 c2` in both seeds (caught at the column challenge).  Full text:
`note:r20-proof/coordinator/20260923T0929Z-finding-ligero-verify-resolved-discrepancies` (appendix 2).

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
`verity.commitments` (`backends/direct/ligero/auth.py`: a-words, b-words, output words of the frozen input set,
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

Resolved: both verifiers check `union_log2 = per_proof_log2 + log2(n_proofs)`, with `n_proofs` from the statement header equal
to the `N` sub-batches presented (`protocol.batch_bound`; Rust `verify::batch_bound`).  Full text and evidence:
`note:r20-proof/coordinator/20260923T0929Z-finding-ligero-verify-resolved-discrepancies`.

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
  `fp8-hopper`, whose `relations.py` / `wide.py` have since merged to main), twice each, byte-identical; `fp8-ada` also equal to
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

`bf16-hopper-v2`, `bf16-ampere-v2`, `fp8-hopper-v2`, `fp8-ada-v2` (`backends/direct/ligero/pubsel/relation.py`) claim what
their v1 counterparts claim with 282 / 449 / 185 / 337 rows per unit (2ef2409) (v1: 3292 / 3516 / 3396 / 3769) by moving the public
half of the tensor-core step to the verifier: the pins are per-group maxima `g{g}.G` and truncated product sums `g{g}.Q[j]`
that the verifier computes from the operand words (`public_pins_int`; Rust `relation::pubsel::PubSel::unit_pins`), so the
verifier does O(K prod_bits) integer work per unit and the relations are unusable with private operands (not a Table 2
headline; a finding).  Full text (K, decode, tags, digests, the zero-sum fix, `bf16-ampere-v2` as a new claim):
`note:r20-proof/coordinator/20260923T0929Z-finding-ligero-verify-resolved-discrepancies` (appendix) and the relmin-lookup lane note.
  Post-checkpoint (6b79b06 / 87c94b6): the folded `-v2x4` relations (`bf16-hopper-v2x4` 1045 rows / 24 columns per VU, `bf16-ampere-v2x4` 1717 / 24, `fp8-hopper-v2x4` 647 / 12, `fp8-ada-v2x4` 1255 / 12) put four instruction steps in one column (`GroupSum(groups x 4).step` == four chained steps, tested); same claim as the base v2 VU, the three intermediate accumulators private; the BF16 folded units take the empty tag like their base; pinned `f1211dba / dbb92408`, `8ce6ca34 / c9611303`, `3463ce9f / 8d679547`, `d2272d3f / 87be1aea`; GPU gates r20260923-104801-fc96 (0 failures).  Full text in the relmin-lookup report note.

~~~json
{"id": "BV-D12", "issue": "v2 pins are verifier-computed group maxima / truncated sums; K fixed by the relation; unusable with private operands",
 "code": "pubsel/relation.py, relations.py", "verifier": "src/relation.rs pubsel::PubSel; verify.rs public_pins_pubsel",
 "status": "open (finding)", "owner": "relmin-lookup", "evidence": ["tests/relations.rs", "pubsel/relation_test.py", "lane note relmin-lookup"]}
~~~

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

## D14. The v3 private-operand relations (lane `relmin-private`): the pins are the operand words again, packed for FP8, and the verifier computes nothing

`bf16-hopper-v3`, `bf16-ampere-v3`, `fp8-hopper-v3`, `fp8-ada-v3` (`backends/direct/ligero/privsel/relation.py`, registered
in `relations.py` beside the v1 and v2 names, both untouched) claim what their v1 counterparts claim -- the same pinned
instruction semantics, statement widths and chain-end word -- with a third compiled unit: 1519 / 1629 / 1673 / 1806
rows against v1's 3292 / 3516 / 3396 / 3769 (2.17x / 2.16x / 2.03x / 2.09x fewer), every function of the operands proven
inside the relation as in v1 (products, alignment, group maximum, exact sum, normalisation, epilogue), on operand words
that are committed witness wires.  What a verifier sees:

* **The pins are the words themselves, and nothing is computed from them.**  A v3 system pins `a[i]` / `b[i]`, the
  16-bit BF16 word as written in the statement (one word per pin), or `a[lo:hi]` / `b[lo:hi]`, the run of three E4M3
  words `lo <= j < hi` packed little-endian `sum_j w_j 2^(8 (j - lo))` (`ShiftPlan.words_per_pin` = 3: 32 words -> 11
  pins per operand, the last run holding two words); the linear constraint `sum_j a_words[j] 2^(8 (j - lo)) = a[lo:hi]`
  binds the pin to the witness's word wires `a_words[j]` (the weighted sums of their hint bits).  The verifier's public
  work is the v1 word check (a non-finite word is refused: BF16 `t = 255`, E4M3 `0x7F` / `0xFF`; `Operand::decode`) and
  the packing; there is no group maximum, no aligned sum, no shift -- the soundness surface of D12's public computation
  is gone, and the unit's row count is the only thing a private-operand relation (lane hash-relation: the pin rows
  replaced by a Poseidon2 sponge over the same `a_words` / `b_words` wires) inherits.  `PrivSel::unit_pins` /
  `pin_index` (`relation.rs`) compute the pinned vectors and place them by system pin name; `public_pins_privsel`
  (`verify.rs`) does it for every column, pad columns included (all-zero words are on the domain).  Pinned against
  `privsel.relation.public_vectors_v3` by hand-computed words (`privsel_unit_pins_match_public_vectors_v3`).
* **K comes from the relation** (`PrivSel::k` = 16 / 16 / 32 / 32, the sum of `Params.groups`), as for v2; a statement
  whose `K` differs is refused before any pin is computed.
* **Off-domain encodings the relation excludes itself**: `x = t - nz` and `nz` are hint bits, `nz = 0 -> x = 0` by a
  zero-product and `x` off the top by one inverse (`floor(x/2) != 127` for BF16, excluding `t = 255` and the `t = 256`
  collision of `(x, nz)` with the sign bit; `x != 15` for E4M3), so the only word a witness can encode that the model
  refuses is the E4M3 NaN (`t = 15, f = 7`), which the verifier's word check refuses in the pin variant (the sponge
  variant must exclude it in the relation or refuse it at the hash's input).  A witness whose word wires encode a
  different word than the pinned one, or an encoding off the domain, fails the linear test
  (`test_committed_words_are_bound_and_canonical`).
* **Tags and pins.**  All four absorb `|rel=<name>|v1` (BF16 included: the pins differ from every other BF16 relation's
  and nothing shares a transcript with them).  Pinned digests (`sys_id` / table digest): `e3ff1f9f... / 87ec0bfd...`,
  `c961ba70... / 72cc7c90...`, `c5668bcf... / 36680ea2...`, `1bf8ce3e... / 2f74b24d...` -- laptop compilations from the
  lane tree, torch-free, twice each, byte-identical, the `sys_id`s equal to `protocol.system_id`; fixtures under
  `fixtures/systems/`.  (The previous pins `2279f2ac / f962956b / bc6f8780 / adaea5d9` are the systems before the
  exact-zero-sum fix below; dumps made from them verify as unpinned systems.)  The dumps of the CPU bench (`bench-vu --dump-dir`, 8 VUs, ZK, Fiat-Shamir and interactive) of
  every relation verify under the pin (`ligero-verify batch`: ACCEPT, python agreement 1/1); a statement with one operand
  bit flipped, a sign flipped, a claimed word bit flipped, two words swapped, `a` and `b` swapped, or a pad column's word
  changed is rejected (the transcript binds the statement digest: "column challenge mismatch"), a NaN word with the
  decode's reason.
* **The exact-zero sum.**  v2's normalisation rule (D12; `pubsel/relation.py`: `hid = 0 -> t = tc(E)`, the subnormal
  grid) has no satisfying shift when the exact sum is zero and `E - width > acc_e_min` (`tc < t_lo = -nb`): the honest
  unit is rejected -- a completeness gap, found by the lane's 1e5-unit differential (fp8-hopper-v3 unit 92536 of the
  seeded set: `c = 0`, two live products cancel, the rest dead; the model's word is `+0`, v1 accepts, **`fp8-hopper-v2`
  rejects the same unit at `g0.norm.clamp`** -- the v2 relations in `main` carry the gap; relmin-lookup's to fix).  v3's
  rule is `(1 - hid - nsel[t_lo]) (t - tc) = 0`: at `t = t_lo` the window is `2^nb x popcount(v)` and `Mn < 2^nb` forces
  `v = 0`, hence `Mn = 0`, `hid = 0`, `z = 1` and the zero output whatever `t`; `hid = 1` at `t_lo` is unsatisfiable, so the
  factor is never -1; the honest prover picks `t = max(t_lo, tc)` (it already did).  No row, no new pin; the systems'
  bytes (one quadratic) and hence the digests changed.
* **Hint rows may be negative.**  The signed aligned product `(1 - 2 (s_a xor s_b)) X` is a hint row `Xs` (`XsL` /
  `XsH` for the wide BF16 Hopper sum) under two row-free quadratics `(Xs - X)(1 - s_a - s_b) = 0` and
  `(Xs + X)(s_a - s_b) = 0`; the prover writes it as a signed int64 and the witness program reduces it mod p
  (`modp_signed`), like v2's `g0.Q[j].hi` pin.  A verifier needs nothing new: the row is a witness column.

~~~json
{"id": "BV-D14", "issue": "the v3 relations pin the operand words themselves (packed three per pin for E4M3); the verifier computes nothing from them beyond the word check; K is fixed by the relation",
 "document": "PROTOCOL.md describes the BF16 unit only; privsel/relation.py's module comment is the reference",
 "code": "backends/direct/ligero/privsel/relation.py (compile_v3_unit, public_vectors_v3, pin_name), relations.py (the four names)",
 "verifier": "src/relation.rs privsel::PrivSel (unit_pins, pin_index, pins; the four parameter sets), Decode::PrivSel; verify.rs public_pins_privsel",
 "status": "open", "owner": "relmin-private (PROTOCOL.md section for the private-operand unit; the sponge variant's NaN exclusion is lane hash-relation's)",
 "evidence": ["src/relation.rs tests (privsel_unit_pins_match_public_vectors_v3)", "tests/relations.rs system_digest_names_every_pinned_relation", "backends/direct/ligero/privsel/relation_test.py", "lane note relmin-private"]}
~~~

## D15. Folded (x4) private-operand-safe relations (lane `fold-private`): same VU claim, four steps per column, K = 4k

`relations._folded(base, 4)` (`Relation.fold`, `folded_from`) gives eight additive relations: v3 folds `fp8-ada-v3x4`,
`fp8-hopper-v3x4`, `bf16-hopper-v3x4`, `bf16-ampere-v3x4`; v1 folds `fp8-ada-x4`, `fp8-hopper-x4`, `bf16-hopper-x4`,
`bf16-ampere-x4`.  Base systems byte-identical to their fixtures.  A folded unit's statement carries the
column's K = 4·k_base operands (128 FP8 / 64 BF16 words; the v3 `PrivSel` decoders take that k and refuse the base k,
FP8 packed 3 words / pin as in D14) and the column-end accumulator; the three intermediate accumulators are witness
rows (the statement says less than 4 base statements, D11).  The fold is exactly 4 chained base steps; the folded-boundary negatives reject an intermediate
word claimed as final and operand swaps across base-step / column edges.  Test hazard: `relchain.marshal` caches
bundles by `id()` of the VU tuples; fresh temporaries can replay the previous bundle.

~~~json
{"id": "BV-D15", "issue": "folded x4 relations pin K = 4k operand words and the column-end word only; intermediates private",
 "document": "PROTOCOL.md: one step per unit; relations._folded docstring is the reference",
 "code": "relations.py (_folded), relchain.py (folded-boundary negatives, frozen_instances, cache reuse), fold_test.py",
 "verifier": "src/relation.rs (8 folded statics; PrivSel k=64/128), tests/relations.rs (folded k rules)",
 "status": "open", "owner": "fold-private (PROTOCOL.md folded-column section; marshal id() cache key)",
 "evidence": ["tests/relations.rs (25 pinned)", "fold_test.py", "lane note fold-private"]}
~~~
