# Flock verifier of record: specification outline

The verifier of record is the Lean verifier, `verity/core/service/flock/` (`flock-verify`, Lean 4, no dependencies). Its
definitions are the specification; this file is their outline, one section per part of the verifier, each naming the
Lean module that holds it. The section numbers are stable, since the Lean sources and other documents cite them. Where
the outline and the Lean differ, the Lean wins.

**Sources.** The protocol is the Flock paper (Bünz, Rothblum, Wang, ePrint 2026/1329: §4, Appendices A–C). Upstream's
code (succinctlabs/flock `b684b12`) was read only for byte formats, constants and the order of operations; every place
where the paper is silent or differs and the verifier follows upstream is listed in §19. The statement layer
(`verity/ml/flock/live/src`) and the session protocol are verity's own.

**Scope.** Interactive mode only: every coin comes from the verifier's live session record. Fiat–Shamir, the union
prover, the AG flavour, the prime (A-GKR) half, forks and grinding credit are out of scope and refused where they could
appear. Upstream's Rust verifier is a cross-check (§17.1), never a verifier.

**Style.** Every function is pure and total over explicit inputs with explicit byte formats. Every length read from an
input is compared with one computed from the configuration before it drives a loop. Nothing reads a clock, a random
source or a file other than its inputs.

## 1. Conventions

`‖` is concatenation; `le32`, `le64`, `be32`, `be64` are fixed-width encodings. `F` is GF(2^128) (§2.1), `K` is
GF(2^256) (§2.4), `F8` is GF(2^8) (§2.2); `+` on field elements is XOR. `eq(r, b) = Π_j (r_j if bit_j(b) else 1 + r_j)`,
bits least-significant first. A check fails by returning `Reject reason`; reasons are informative, and only
`Accept`/`Reject` is normative.

## 2. Fields and hashes

`Flock.Field`, `Flock.Hash`. The field code is proved to be the fields it names (`Security/Proofs/Flock/`).

### 2.1 F = GF(2^128), GHASH form

`F = F2[x] / (x^128 + x^7 + x^2 + x + 1)`, held as `(lo, hi)` 64-bit limbs, encoded `le64(lo) ‖ le64(hi)` on the wire and
in the transcript, and as `hex(be64(hi) ‖ be64(lo))` in the record. `inv(0) = 0`.

### 2.2 F8 and φ8

`F8` is the AES field. `φ8 : F8 → F` is the field embedding with `φ8(α) = 0dcb364640a222fe6b8330483c2e9849`. The
univariate-skip nodes are `φ8(0 … 63)` and their coset `φ8(64 … 127)`.

### 2.3 Encodings of field elements in proofs

An F element in a proof is 16 bytes, `le64(lo) ‖ le64(hi)`, unless §8 says otherwise.

### 2.4 K = GF(2^256)

`K = F[u] / (u^2 + u + x^{−1})`, an element `(c0, c1) = c0 + c1·u`. A K coin is one `sample_f128_vec(2)` squeeze.

### 2.5 Hashes

The current statements (§16.10 on) use SHA-512 on every binding path: the proof's Merkle trees (§14), the round
digests (§6), Σ, the statement digest, the frame-v3-sha512 roots and the record's digests. The proof's hiding leaves
are Halevi–Micali on SHA-512 (`Flock.Hm96`, `hm96-sha512/v1`). SHA-256 and keyed BLAKE3 survive only in the archived
statement versions the verifier still parses and refuses for their form (§16.5, §16.15). The transcript's own hash never
enters the verifier (§6.3).

## 3. Inputs, outputs and the shape of the verifier

`Flock.Verify`: `verify : Config × Record × Proofs × PublicInput → Verdict`, `Verdict = Accept | Reject Reason`.
`Config` is the verifier's own (§4) and nothing in it is read from the record or the proofs. The pipeline:

1. `setup(Config, PublicInput)` (§16): Σ, the statement digest, `Hello`, the regions, the lincheck fold, the PCS
   parameters;
2. decode the record and run the session checks (§5.3);
3. per table and rep: decode the proof (§8), replay it on a fresh transcript (§6) through §9–§14, then check the stream
   was consumed (S15) and the proof's cap is the committed root (S16);
4. the proof-file check S17 and the cross-rep check S18;
5. the fail-closed rules (§16.15), last, so a session that fails a check keeps that check's reason.

## 4. Configuration pinned by the verifier

Profile `fast100` (§15); two reps on one committed root; the `rs` flavour (univariate skip over the AES subfield);
`PcsParams(m)` computed from the statement; a link of two points with `y = 0`; `require_link = true`; no prime half; the
coin source a plain session's identity names (`--coins seed`, the default, or `--coins os`, §7; a `--zk` session's coins
are its coin tree, §7.4); no grinding: every proof-of-work nonce must be `0` (R4).

## 5. The session record

`Flock.Record`, `Flock.RecordText`.

### 5.1 Format

`flock-live-session/v1` JSON, in exactly one text: Lean's `Json.compress` of it, which `live/src/record_text.rs` and
`verity_flock.record_text` also write (held to Lean's output by `live/src/record_text_vectors.json`). The verifier reads
`kind`, `aborted`, `hello`, `config.require_link`, `link_mode`, the `link` exchange (Σ, root_F, each table's root_B and
public words, the points, `y`, each with its digest), the `streams` with their rounds (`k`, `n`, the framed bytes' length
and digest, the coins, and the retained bytes `msg`), `proof_sha512`, `unit_draw` and, under `--zk`, `coin_tree`. It
ignores every other field, the live server's own verdict included.

### 5.2 Streams and rounds

A root stream `<table>/rep<r>` is one transcript whose coins came from the verifier. Round `k` records the framed bytes
absorbed since the previous coin request (their digest, and the bytes themselves) and the coins returned. No statement
reaches a fork, so a child stream is refused.

### 5.3 Session checks

| id | check |
|---|---|
| S0 | the record is in its canonical text (checked last) |
| S1 | `kind = "flock-live-session/v1"`, `aborted = null` |
| S2 | `hello` is the verifier's own `Hello` byte for byte (profile, reps, flavour, tables, Σ, link shape, coin scheme) |
| S3 | the link was exchanged (`require_link`, `link_mode = link.mode = "exchange"`) before any coin |
| S4 | `link.sigma = Σ` |
| U1–U4 | the unit draw (§7.3) |
| S5–S7 | each configured table's root_B, root_F and public words are present and match their digests; `root_F = Σ` |
| S8 | the committed public words are consistent with the public input |
| S9–S10 | the link points have the configured shape; every `y` is zero |
| S11 | the streams are exactly `<t>/rep<r>` for the configured tables and both reps, each a root with its domain label |
| S12 | rounds are numbered in order, `1 ≤ n ≤ 65536`, `|coins| = n`, and retained bytes hash to the round's digest |
| S13 | every root stream bound its table's root_B |
| S14 | the link digest is over the exchange the coins followed |
| S15 | each rep consumed exactly its stream's rounds |
| S16 | each proof's cap is the committed root_B |
| S17 | the proof files are the ones the record's digests name |
| S18 | both reps' caps are equal |

## 6. The replay transcript

`Flock.Transcript`. The transcript never hashes to derive a challenge: every coin is read from the record after the
bytes that preceded it are checked against that round (R2). This is the whole interface between the protocol (§9–§14)
and the session.

### 6.1 Framing (transcript-v2)

Every absorbed item is a frame: a 16-byte header `[op u8][kind u8][0u8; 6][len u64 LE]`, then the payload zero-padded to
16 bytes (`domain`, `observe_label`, `observe_f128`, `observe_f128_slice`, `observe_bytes`); a squeeze appends its header
and ends the round.

### 6.2 State and rounds

A stream's state is the pending bytes and the round index. A squeeze of `n > 0` coins requires round `k` to exist with
`n` coins and its retained bytes to equal the pending bytes (or, for a record without them, their digest to match), and
returns the recorded coins. `squeeze_answers` and `squeeze_retained` prove it.

### 6.3 What the transcript does not contain

The prover's transcript hash (BLAKE3 on the CPU prover, SHA-256 on the CUDA prover) never enters the replay: only the
framing does, so a CPU proof and a CUDA proof of one witness replay identically.

### 6.4 Forks

The single-table block-R1CS path makes no fork, so the verifier implements none and S11 refuses a child stream.

## 7. Coin sources

The verifier reads every coin from the record; how the server drew them decides only what soundness assumes.

### 7.1 `os`

The server draws each round's coins from the OS after recording the round's digest. Soundness assumes only that they are
uniform and unknown to the prover before it sent the round. `verify --coins os` binds the identity that mode
(`FC_COINS=os`) writes; it binds the identity, not who drew the coins.

### 7.2 `seed`

The server commits to a seed at `Hello` (`coin-commit/sha512`) and derives every coin from it with
`verity.ml.randomness` v2; `flock-circuit replay-coins` audits the commitment and the derivation. The verifier still
reads the coins from the record; with a seed, soundness also assumes the commitment hides it and SHA-512 is a PRF.

### 7.3 The unit draw

`Flock.Draw`, `flock-verify draw`. After serving registers its roots and before the session, the verifier draws the
sampled units from its own OS randomness with `verity.ml.randomness`'s exact samplers, in the clear and under no hash.
The draw is the public header's `unit_draw` (so under the public digest and Σ), instance `i` being unit `u_i`.

- **Sampling designs.** `subset(k, n)` (partial Fisher–Yates), `bernoulli(p)`, `stratified` (the default: strata by the
  template descriptor id of each unit's root node, derived from the verifier's own program and partition object, stratum
  `s` drawing `max(1, round(K n_s / N))`), and `work` (the stratified strata sized by each template's work share from the
  verifier's own `--work-table`, with floors, optionally with a `closure` from the verifier's own closure map). The
  verifier never takes a draw's `K`, strata, works, floors or closure as stated: it derives them and compares.
- **Checks.** U1: the record keeps the statement's draw. U2: the draw is well formed and the verifier's own derivation.
  U3: the statement's instances are the drawn units and their closure. U4: every unit that reads a round's commitment
  (`rec-c<i>`, `rec-top-L<l>`, `rec-opening-c<i>`, `rec-top-stacked-L<l>`, `rec-sparse-<p>`) is drawn in full.
- Which designs a soundness claim covers is §16.15's rule; the guarantees are in `Security/Proofs/Flock/`.
  `stratified_agree.py` holds the Lean draws to core's (`verity.core.protocols.sampled_proofs.one_stage.draw`).

### 7.4 The keyed coin tree

`Flock.CoinTree`. A `--zk` session's coins are committed at `Hello` in a `coin-tree/hm96-sha512/v2` tree whose every hash
is prefixed by the prover's fresh nonce; the server answers with the root and a fresh hm96 key. The verifier checks
`Hello`'s nonce and canonical form (R7), the record's `coin_tree` (`key`, `nonce`, `root`, `spec`) and that no earlier
session of the run used the key.

## 8. The proof container

`Flock.Proof`.

### 8.1 Encoding

A proof file is `bincode 1.3` of `(Commitment, R1csProofLigerito)`, then the zk patch's `zk_level0 = None` tag, as every
deployed prover writes it (`--upstream-plain` takes upstream's builds' form, without the tag). Decoding is total and
refuses a truncated input, an out-of-range variant, a byte after the end and a non-empty `matrix_evals`; a file is
accepted only if it is the one encoding of the proof it decodes to (`Flock.decodeProof_canonical`), since S17 hashes the
whole file.

### 8.2 Layout

`Commitment = cap ‖ PcsParams`; `R1csProofLigerito = zerocheck ‖ lincheck ‖ pcs_open`, the opening being one ring
switch per claim and a Ligerito proof (initial cap and level, recursive caps and levels, final level, the K sumcheck
messages, out-of-domain values and nonce lists). The field order is `Flock.Proof`'s.

### 8.3 Expected shape

Every length is fixed by `m`, `k_log`, the claim count and the §15 schedule, and checked before the elements are read:
`params` must be exactly `PcsParams(m)`, `initial_cap = cap`, and every nonce `0`.

## 9. Statement binding and zerocheck

`Flock.Piop`. A witness bit address is `s + 64·t + 2^{k_log}·o` (skip index, inner rest, block).

### 9.1 Binding (round 0, first part)

Before the first coin the transcript absorbs `"flock-r1cs-v0"`, the statement digest and the commitment's cap.

### 9.2 Zerocheck transcript and checks

The univariate-skip zerocheck of the paper's §4.3: six unread `r_skip` coins (§19 A1), the free coordinates, the first
round's 64 + 64 values, the skip challenge `z`, then `m − 6` residual rounds `(G(1), G(∞))`. Address bits 6–12 take
seven pinned coordinates (`φ8(0xF7)`, `φ8(0x53)`, `φ8(0xB5)`, `x/(1+x)`, `x²/(1+x²)`, `x⁴/(1+x⁴)`, `x⁸/(1+x⁸)`). Checks:
the c-claim's interpolation equals `final_c_eval`, and the last claim equals `final_a_eval · final_b_eval`.

## 10. Lincheck

The verifier folds the block matrices itself: `comb = α·A_0ᵀe + B_0ᵀe` with the constant-wire coin `β` on the pin
column, over `k_log − 6` rounds (top bit first), then checks the prover's 64 skip evaluations against the claim and
draws a fresh skip coin. The fold is computed once per slot type (`Stmt.fold`).

## 11. The claims handed to the opening

`[ab, c, e_1 … e_K]`: the lincheck's claim, the zerocheck's c-claim, and the statement's region claims (§16.2) in order.

## 12. Opening: claim batching and ring switching

`Flock.Opening`. Each claim is ring-switched with its own seven coins (its 128 values checked against the claim's
weight), the switched claims are batched with independent coins into one inner-product claim `T`, and Ligerito decides
`Σ_y f(y)·b(y) = T` for the committed packed polynomial and the batched basis `b̂`.

### 12.1 Packing

The committed table packs the 7 low address bits into one F element, basis `x^v`.

### 12.2 Transcript and checks

One `flock-pcs-open-batch-v0` label, then per claim a `flock-ring-switch-v0` label, its 128 values, its check and seven
coins, then one batching squeeze. Claim values and points are not absorbed: the session order fixes them first.

### 12.3 Recombination

Each claim's 128 values are repacked as a bit matrix's columns and combined with its coins; `T` is their batch.

### 12.4 The batched basis

`b(y) = Σ_k γ_k · Φ_k(eq(z_k, y))`, evaluated at Ligerito's points in K by the paper's width-128 branching program.

### 12.5 Hand-off

`T` enters Ligerito's transcript as an F element and its running claim as a K element.

### 12.6 Rejections

A wrong count of ring switches or values, a failed claim check, a nonzero nonce, a claim of the wrong length, or a
Ligerito rejection.

## 13. Ligerito (dense opening)

`Flock.Ligerito`. The paper's Appendix C with four changes (§19 A11): claims and folding challenges in K, so each
recursive level commits the split of a K-table; stratified queries; a Merkle cap instead of a root; one out-of-domain
sample at level 0 and two at every later level.

### 13.1 Commitment and code

Level 0 has `k0` lanes; the code is additive Reed–Solomon over F in the LCH novel basis; each level commits a Merkle cap
whose leaf `q` is column `q`. Later levels have 16 lanes.

### 13.2 Round algebra

A sumcheck message is `(u0, u2) ∈ K²` with `q(0) + q(1) = s` by construction; new claims are batched in with a coin.

### 13.3 Transcript

Level 0's label, `T` and cap; its out-of-domain sample; the `k0` folding rounds; then per recursive level its cap,
out-of-domain samples, queries on the previous commitment, the consistency batch and its folding rounds; the last level
sends its clear residual `yr`.

### 13.4 Consistency weights

The multilinear extension of the code's basis at an opened position, `Π_i (1 + p_i · (1 + Ŵ_i(ω_q)))`.

### 13.5 Query positions

Each query reads only the low `d − c_j` bits of its coin's `lo` word, within its stratum; sampling is with replacement.
Flipping an unread bit must not change the verdict.

### 13.6 Final check

The residual equation over `yr` with every out-of-domain and consistency context, and every list exhausted.

## 14. Merkle trees

`Flock.Merkle`. The scheme is a parameter named by the circuit file's `leaf_scheme.id`: `hm96-sha512/v1` hiding leaves
(each opened row with its 192-byte salt) on the current statements, `flock-leaf/sha512-unsalted` for a recursive
session's inner statement (§16.16), and the unsalted SHA-256 leaves of the archived versions. A level's commitment is
its cap at depth `⌊log2 Q⌋`; each query climbs `d − c` siblings to its cap entry. `merkle_binding` reduces two accepted
openings with different rows to an explicit collision.

## 15. Parameters: `fast100`

`Flock.Schedule` holds the schedule as data, for `m = 22 … 35` only. Level 0 has rate 1/2 and 218 queries; levels ≥ 1
fold 4 variables at rates 1/4 to 1/128; no level grinds. Per table, two reps on one root with independent live coins:
`flock_verify_sound_205` proves 2^-205 for every statement accepted without `--zk`.

## 16. Statements

Each statement supplies `setup`: the table, the domain labels, Σ, the statement digest, `Hello`, the block R1CS, the
link shape, the regions and their claims. The session shape is `Commit(root_F = Σ, root_B, publics)`, two link points,
`Link(y = 0)`, then Flock's coins.

### 16.1 Block matrices

`A·z ∘ B·z = z` over F2 with `A = I ⊗ A_0`, `B = I ⊗ B_0`: each block is cut into aligned slots (compression circuits,
units, tail stages, lookup slots, one mask slot), Δ wires slots together, and one pin column is fixed to 1.

### 16.2 Region claims

A region is a set of in-block bits whose values the verifier knows; each yields one extra claim per link point. Free
bits must be distinct.

### 16.3 `verity/flock-pure-block/v2`

An archived statement (BLAKE3 row digests, public intermediate values). The Lean verifier does not implement it; its
recorded sessions are upstream's only.

### 16.4 IR frame statements

`verity/flock-ir-frame/v3`, `verity/flock-ir-sampling/v1` and `verity/flock-ir-block/v1` are archived, like §16.3.

### 16.5 The circuit statement

`Flock.Circuit`, `Flock.Statement`, `Flock.Tags`. The circuit file is `<format> <name>`, a `META <json>` line, then
`CIRCUIT` blocks, each an expanded circuit in `flock-ir-unit/v2` form. The parser refuses everything by which the
verifier would evaluate part of the computation (`CUT` lines, tail or public-port keys): it checks proofs, never
evaluates. Each expanded circuit's rows are in topological order, its input groups disjoint and below the constant, its
output groups inside its slot. Lookup slots are built from the pinned MUFU tables (`rsq`, `sqrt`, `rcp`, `ex2`). Every
statement must lie in the proved range: `k_log ≤ 27`, at most 1024 regions, `m_pts ≤ 64` (`checkInRange`). The byte tags
are a parameter, one `Tags` value per version; the versions before §16.10 (`verity/flock-netlist/v1`, `@19c7269a`,
`@fd02e847`, `@631567f7`, with keyed-BLAKE3 row leaves) are archived: still parsed, always refused (§16.15).

### 16.6 frame-v3 roots

The public file's roots are frame-v3 trees (`verity/ml/commitments/frame_v3/PROTOCOL.md`), on SHA-512 for the current
statements. Recomputing them over the public digests and outputs is a check of the commitment on public data.

### 16.7 The partition invariant

`Flock.Partition`, `Flock.Qword`, `Flock.Qcall`, `Flock.Program`, `Flock.Extract`, `Flock.Sigma`. The verifier checks a
partition on its own copy of the program: each gate certified by exactly one unit, the committed set the boundary set
(the failure codes of `verity.ml.circuits.partition.validate_unit_cut`), and the width rule `Q_word_v1{X, W, EXTRAS}`.
`Q_word` v1 and v2 and `Q_call` v1 evaluate over the decoded `verity-ir/descriptor/v1` program, per Definition, never
over the unrolled circuit. σ checks a Boolean Program's lift of its word Program on the two descriptors.

### 16.8 Objects by content

With `--archive DIR`, every object (circuit, public file, partition, program, tables) is read from the verifier's own
store by SHA-512, never from the prover's session directory.

### 16.9 Strict conformance

The verifier checks proofs and never evaluates any part of the computation; circuits are total; the circuit is public
and every value private. Only strict statements exist in the Lean verifier.

### 16.10 The hm96-sha512 row statement

`Flock.HmRow`. Serving rows are `hm96-sha512/row/v1` leaves (or `row/v2` bit rows, or `row-seg/v1` segmented bit rows)
proved in the circuit: `x = SHA-512(prefix ‖ row)`, `b = x ⊕ M(key)·y`, `c = SHA-512(salt_prefix ‖ y)`, under the pinned
default key. Every identity and binding digest is SHA-512, and the unit draw is live (§7.3).

- **Slots and nets.** `sha512x3` and `hm96` slots compute each row's leaf; their nets must be the verifier's own
  (`Flock.HmNets`). So must the recursion's units (`rec-open/…`, `rec-open-stacked/…`, `rec-cap/…`, `rec-residuals/…`,
  `rec-opening/…`), each named with its parameters and checked against the verifier's builder before its net is built
  (`Flock.RecOpen`, `Flock.RecResiduals`, `Flock.RecOpening`, `Flock.RecVStar`).
- **Layout.** At least one unit per VU and a power of two of VUs per block; header sizes bounded before any power is
  computed; row wires, the pin and Δ derived by the verifier and compared; the region-word and region-size checks.
- **Public file.** `flock-circuit-inputs`: header, each instance's `b ‖ c` per port, then the outputs; the public digest
  is SHA-512 over the canonical header and the body. Shared-row files commit each port's rows once as a table, with
  per-instance refs.
- **Digests.** Circuit pin, statement digest and `Σ = SHA-512("verity/flock-circuit/sigma" ‖ circuit ‖ statement ‖
  public)`, which is `Hello`'s `link.sigma` and root_F.
- **End-to-end bindings.** META binds `program_sha512` and the partition (`verity/partition/v1` with the verifier's own
  `--partition` object, or the provisional unit cover), and each instance's class.
- **Template queries and the partition finding.** `Flock.TemplateQuery`: with the verifier's own `--program`, the
  verifier derives the statement's units itself; every verdict carries `partition = {rule, query, digest, units, note}`,
  `units` being `derived` or `as stated`.
- **Registered reads.** `Flock.Registered`: with the verifier's own `--registered F`, each read row must open its
  registered root at its position.
- **Tail outputs, packed ranges and leaf cuts**, for the GEMM coordinate's statements.

### 16.11 The typed statement

`Flock.Typed`. `verity/flock-circuit/types`: the unit given as circuit types and layouts instead of rows, from which
the verifier derives every row itself (`DeriveAll.deriveChecked`). No typed statement carries rows.

### 16.12 Sessions of several tables

`verify --session-tables J` (the verifier's setting, never the record's): the session's units split across `J` tables,
the draw of a fixed count (`subset` or `stratified`).

### 16.13 Hidden outputs

The default `verity/flock-circuit` and `verity/flock-circuit/types`: every output is a committed row, so the public file
holds roots, hiding leaves and structure only (`Stmt.setupHidden`). The plain transcript is not hiding (`NON_ZK_PROOF`);
only `--zk` targets zero knowledge.

### 16.14 `--zk` sessions

`Flock.Zk`, `Flock.ZkProof`, `Flock.ZkLigerito`. `verify --zk` checks `flock-circuit --zk` sessions by the live
`PROTOCOL.md`'s masked protocol on §7.4's coin tree, with the prover's randomness the OS's bytes (`Tags.osMasks`). Each
table needs a mask slot of at least 2^13 bits (`Zk.SLOT_LOG_MIN`), and no Δ pair or region may meet it (`Zk.maskFree`).

### 16.15 The forms the soundness proof covers

`Flock.ProvedScope`. `verify` fails closed: soundness is one theorem, `FailClosed.flock_verify_sound`, over every
session the verifier accepts, so every statement outside the forms it covers is refused, once every check has passed,
with `refused: <why>: a statement form the soundness proof does not cover`. The rules, in order:

1. hm96-sha512 rows (§16.10 on);
2. untyped (§16.11);
3. retained rounds and `hm96-sha512/v1` leaves, or plain SHA-512 leaves at hidden outputs without `--zk` (§16.16);
4. one table without `--zk`, at most 2^16 with it; under `--plain-tables`, up to 64 tables with the verifier's own
   `--program` matching META's `program_sha512`;
5. no public input ports;
6. no shared-row public file, except at hidden outputs and `hm96-sha512/v1` leaves with the verifier's own `--registered F`;
7. where wires feed the unit or the outputs, an order of the wires that computes every slot from the leaves;
8. `Layout.scopeOk` (with hidden outputs `scopeOkI`, the general instance form);
9. every table's `m` at least 22, or 25 under `--zk`.

Also refused: a draw by a design no claim covers (bernoulli, work, a work draw with a closure, and a subset draw without
`--zk` at public outputs; under `--plain-tables` the subset design alone), a session without a draw, a session verified
without the verifier's own `--program`, and under `--rec-inner` a V* session that does not draw every unit of its one
full table (`RecSetup.fullDraw`, `RecSetup.oneFull`), over an inner session whose record holds no draw
(`RecSetup.innerDrawn`), or at a holo or sparse inner shape (`RecSetup.denseOnly`: `RecursiveKS` covers the dense form
only, and the compiled model refuses the same sessions, `RecDeploy.acc_dense`); a recursive session without `--rec-inner`
or without `--zk` (`RecSetup.zkOnly`) is refused too.
`flock-verify rec-compose` (`Flock.RecCompose`) gives a recursive statement's verdict from its parts' verdicts.

### 16.16 Plain leaves

`verity/flock-circuit+plain-leaves`: §16.13's statement with `flock-leaf/sha512-unsalted` Merkle leaves, a recursive
session's inner statement, whose firewall commits every tree top under its own salt. Checked without `--zk` only.

## 17. Acceptance, assumptions and trust

### 17.1 Agreement with upstream

`agree.py` and `ci.py` (the README) hold the Lean verifier to upstream's Rust verifier (`Server::from_record`, then
`finish`) on every set of `vectors.json`, on mutants and on seeded fuzzing: the same verdict on each session, except where
the Lean verifier is deliberately stricter. A disagreement is resolved against the paper and this outline, never by
copying upstream. The deliberate divergences, each refused by Lean and accepted by upstream:

| id | Lean refuses |
|---|---|
| D1 | `root_F ≠ Σ` (S6) |
| D2 | a stream other than the configured rep streams (S11) |
| D3 | `link.sigma ≠ Σ` (S4); closed upstream from `b3baabd8` |
| D4 | retained round bytes that don't hash to the round's digest (S12); closed upstream from `631567f7` |
| D5 | a row that does not open its registered root, or a port named twice under `--registered` |
| D6 | a malformed `--zk` `coin_tree` (§7.4) |
| D7 | `sha512x3` or `hm96` nets other than the verifier's |
| D8, D9 | no unit in a VU; no VU in a block, or a count that is not a power of two |
| D10 | an `identifier` other than the verifier's own `--identifier F` |
| D11 | an output group past its slot |
| D12 | a statement form, draw design or session the soundness proof does not cover (§16.15), and a `--zk` mask slot below 2^13 bits or met by Δ or a region; `vectors.json`'s `lean_refuses*` marks compare such a session as accepted |
| D13 | a `rec-open` unit whose name, bits, rows or net are not the verifier's |
| D14 | a draw that does not draw in full each unit reading a round's commitment (U4) |
| D15 | an inner statement's registered description that is not the registration |
| D16 | a V* session whose instance labels are not its drawn units |
| D17 | a plain file in the form its statement doesn't pin, or a byte after a proof's end |
| D18 | a non-empty `matrix_evals` |
| D19 | a V* cap session that does not read its level's tops at the verifier's shared rows |
| D20 | a record not in its canonical text (S0) |
| D21 | a nonzero proof-of-work nonce |

### 17.2 Assumptions

A verdict's soundness rests on: the protocol's round-by-round errors at `fast100`, proved in Lean with no named
assumption (`Security/Proofs/Flock/`); live coins uniform and unknown to the prover before the round they answer (with
seeded coins, also that the commitment hides the seed and SHA-512 is a PRF); SHA-512 collision resistance for the
Merkle trees, the hiding leaves, the round digests and the statement binding (with retained round bytes, R2 needs none);
the pinned circuits meaning what the statement says, which the lowering proofs cover against their reference functions;
and the verifier's custody of the record and the proofs.

### 17.3 Trusted components

Lean's kernel, for the theorems. Lean's compiler, runtime and the C compiler, for the running binary: the agreement tests
of §17.1 narrow that trust (an independent implementation must agree on the corpus and on fuzzed mutations), they don't
close it. No `@[extern]` or `@[implemented_by]` substitution in verification logic (`tests/test_lean_verifier.py`
refuses one). Upstream's Rust verifier is only a test oracle.

### 17.4 Assurance stages

The proofs are in `Security/Proofs/Flock/` (its README), under the Lean check (`tools/verity/lean/check.py`, with
`verity/core/service/flock/lean-audit.json`): only `propext`, `Classical.choice` and `Quot.sound`. The verifier
certifies each pinned circuit with a checker the proofs are about, at setup, so no proof evaluates a pinned file in the
kernel and none uses `native_decide`. Prover completeness is outside them: every honest session in the corpus is
accepted.

## 18. Test vectors

`verifier/vectors.json` (`flock-verifier-vectors/v1`) lists the honest sessions, the record-level negatives, the
forgeries and the replayable and live sets; the bytes stay in the evidence store. `transcript_check.py` rebuilds every
round of a session from this outline's §6 and §8–§13 and compares each round's digest and coin count with the record.

## 19. Ambiguities resolved from upstream's behaviour

| id | topic | the verifier (following upstream) |
|---|---|---|
| A1 | zerocheck coins | six extra coins `r_skip`, squeezed and never read |
| A2 | last coin | Ligerito's final claim-batching coin is read, but an honest proof passes for every value |
| A3 | vanishing on S | imposed by interpolating over `S ∪ Λ` with `S = 0`: degree ≤ 127 |
| A4 | `v†` | the prover sends `final_c_eval`; the verifier compares it with its interpolation |
| A5 | residual sumcheck | `(G(1), G(∞))` of the eq-stripped quadratic, `G(0)` derived, `inv(0) = 0` |
| A6 | pinned coordinates | address bits 6–8 in F8 via φ8, 9–12 in F |
| A7 | lincheck | the verifier computes `α·A_0ᵀe + B_0ᵀe`; `k_log − 6` rounds, top bit first; 64 skip values; a fresh φ8 coin |
| A8 | constant wire | an extra coin `β` on the pin column, which must be 1 in every block |
| A9 | ring switching | each claim switched with its own 7 coins, then batched; Ligerito's K sumcheck replaces the standalone one |
| A10 | packing | the 7 low address bits, basis `x^v` |
| A11 | Ligerito | stratified queries, K folding challenges, out-of-domain at level 0, a clear residual of `2^e` values |
| A12 | proof of work | none: every nonce must be `0` |
| A13 | statement binding | label, statement digest and cap absorbed before the first coin |
| A14 | extra claims | region claims at the link points, opened in the same batch, their points and values not absorbed |
| A15 | `root_F` | required to be Σ (S6) |
| A16 | `m > 35` | refused: no `fast100` schedule |

## 20. Mapping onto Lean

~~~text
Flock.Field                 §2.1, §2.2, §2.4
Flock.Hash, Flock.Hm96      §2.5
Flock.Verify                §3
Flock.Record, RecordText    §5
Flock.Transcript            §6
Flock.Draw, CoinTree        §7
Flock.Proof                 §8
Flock.Piop                  §9–§11
Flock.Opening               §12
Flock.Ligerito              §13
Flock.Merkle                §14
Flock.Schedule              §15
Flock.Statement, Circuit    §16.1, §16.2, §16.5
Flock.Partition, Qword, …   §16.7
Flock.HmRow                 §16.10, §16.13
Flock.Typed, Derive*        §16.11
Flock.Zk*                   §16.14
Flock.ProvedScope           §16.15
~~~
