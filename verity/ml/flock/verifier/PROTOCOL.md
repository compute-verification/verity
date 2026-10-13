# Flock verifier of record: specification

**Status: phase 1 of lane `flock-verifier` (2026-09-26).** This document specifies the verifier that decides Flock sessions
in verity: the 100-bit profile (`fast100`) run twice on one committed root, with live verifier coins, for the pure-block,
IR frame and PR #83 circuit statements. Phase 2 implements it in Lean (levels 1–3 of the formal-verification plan). Upstream
Flock's Rust verifier (succinctlabs/flock `b684b12`, with our live-coin layer `verity/ml/flock/live`) is a cross-validation
test in CI, not a production verifier: once the Lean verifier agrees with it on the whole corpus (§17), only the Lean
verifier runs.

**Sources.** The protocol is the Flock paper (Bünz, Rothblum, Wang, ePrint 2026/1329: §4, Appendices A–C). Upstream's code
was read only for byte formats, constants and the order of operations; every place where the paper is silent or differs
and the spec follows upstream's behaviour is listed in §19. Our statement layer (`verity/ml/flock/live/src`) and the session
protocol are verity's own and are specified from that code.

**Scope.** Interactive mode only: coins come from the verifier's live session record. Fiat–Shamir (C-fs), the union prover,
the AG flavour, the prime (A-GKR) half, forks and grinding credit are out of scope and are rejected where they could appear.

**Style.** Every function is pure and total over explicit inputs, with explicit byte formats, so that it can be written in
Lean and reasoned about. §20 maps the sections onto Lean modules.

## 1. Conventions

- `‖` is concatenation. `le32`, `le64` and `be32`, `be64` are fixed-width little- and big-endian encodings.
- Integers are unbounded naturals unless a width is given. Every length read from an input is compared with a length
  computed from the configuration before it drives a loop.
- `hex(b)` is lowercase hex. `SHA-256` is FIPS 180-4. `BLAKE3` is the BLAKE3 compression function and keyed hash.
- `F` is GF(2^128) (§2.1), `K` is GF(2^256) (§2.4), `F8` is GF(2^8) (§2.2). `+` on field elements is XOR.
- `eq(r, b) = Π_j (r_j if bit_j(b) = 1 else 1 + r_j)`, bits least-significant first, unless a section says otherwise.
- A check "fails" by returning `Reject reason`. Reasons are informative only: the verdict is `Accept` or `Reject`.
- Byte strings in `"quotes"` are ASCII, exact, without a terminator unless `\0` is written.

## 2. Fields and hashes

### 2.1 F = GF(2^128), GHASH form

`F = F2[x] / (x^128 + x^7 + x^2 + x + 1)`. An element is the 128-bit integer whose bit `i` is the coefficient of `x^i`,
held as `(lo, hi)` = bits 0–63 and 64–127. Its byte encoding (transcript, wire, `link_sha256`) is `le64(lo) ‖ le64(hi)`
(16 bytes); its record hex is `hex(be64(hi) ‖ be64(lo))`. `inv(0) = 0` (Fermat inversion `a^(2^128 − 2)`), and the spec
keeps that convention where upstream relies on it (§9.2).

### 2.2 F8 and φ8

`F8 = F2[α] / (α^8 + α^4 + α^3 + α + 1)` (the AES field); a byte `v` is `Σ bit_i(v) α^i`. The embedding `φ8 : F8 → F` is the
field homomorphism with

~~~text
φ8(α) = g = 0dcb364640a222fe6b8330483c2e9849        (a root of X^8 + X^4 + X^3 + X + 1 in F)
φ8(v) = Σ_{i<8} bit_i(v) · g^i
~~~

(Checked against upstream's 256-entry table: identical.) The univariate-skip nodes are `φ8(0 … 63)` and `φ8(64 … 127)`.

### 2.3 Encodings of field elements in proofs

In the proof container (§8) an F element is 16 bytes `le64(lo) ‖ le64(hi)` unless §8 says otherwise.

### 2.4 K = GF(2^256)

`K = F[u] / (u^2 + u + x^{−1})`, `x^{−1} = 80000000000000000000000000000043` (hex `hi‖lo`). An element is `(c0, c1) = c0 + c1·u`.
Product: `p0 = a0 b0`, `p1 = a1 b1`, `p2 = (a0 + a1)(b0 + b1)`, result `(p0 + x^{−1} p1, p2 + p0)`. A K coin is one
`sample_f128_vec(2)` squeeze `(w0, w1) → (c0, c1) = (w0, w1)`; absorbing a K element is one two-element slice frame
`[c0, c1]`.

### 2.5 Hashes

- SHA-256: the Merkle trees (§14) of the recorded statements, the round digests (§6), Σ, statement digests, frame-v3
  (§16.6), `link_sha256`.
- SHA-512 (FIPS 180-4): the target statement's commitment paths (below). The Merkle scheme is a parameter (§14), and a
  SHA-512 tree has 64-byte cap entries and siblings.
- BLAKE3 (compression, keyed hash, parent tree): only in the statements (row digests, legacy L-CV checks, padding
  constants). The keyed parent tree over chunk chaining values `cv_0 … cv_{n−1}` (n ≥ 2) splits at the largest power of two
  below `n` and compresses `(key, left ‖ right, counter 0, block_len 64, PARENT | KEYED_HASH [| ROOT at the top])`; flags
  `CHUNK_START = 1`, `CHUNK_END = 2`, `PARENT = 4`, `ROOT = 8`, `KEYED_HASH = 16`.

The transcript's own hash never enters the verifier (§6.3).

**Target: SHA-512 on every commitment path** (Daniel, 2026-09-26; this supersedes the SHA-256 target of the same day).
- The proof's Merkle leaves and nodes are SHA-512.
- The proof's internal leaves are Halevi–Micali on SHA-512 (`Flock.Hm96`, generic over the hash, and tested on
  `hm96-sha256/v1`'s vectors), with the key fixed in the statement.
- The data commitments move at the re-baseline.

The recorded sessions this spec covers use SHA-256 trees and keyed-BLAKE3 row leaves, and the spec specifies them as they
are. The target changes the Merkle scheme (§14), the row leaf (§16.5) and the statements' digests. The protocol of §3–§13 is
unchanged. The target's ids, its `HashKind` index and its leaf layout are published by PR #83's lane; until then, this
verifier's SHA-512 scheme carries placeholder ids.

## 3. Inputs, outputs and the shape of the verifier

The verifier is one total function

~~~text
verify : Config × Record × Proofs × PublicInput × Publics? → Verdict
Verdict = Accept | Reject Reason
~~~

- `Config` is the verifier's own configuration (§4): the statement kind and its pinned files, `profile = fast100`,
  `reps = 2`, `flavor = rs`, the table list, the link shape and the coin source. Nothing in `Config` is read from the
  record or the proofs (condition R7).
- `Record` is the session record the live coin server wrote, `flock-live-session/v1` JSON (§5.1).
- `Proofs` maps `(table, rep)` to the proof file the server received, `bincode(Commitment, R1csProofLigerito)` (§8).
- `PublicInput` is the statement's public data (§16): frame-v3 roots, row digests, output words, and the pinned layout.
  For the circuit statement it holds no row; for the older statements it is their instance file (§16.4).
- `Publics?` supplies the committed public words for records written before G2 (§5.1, footnote). A record that carries
  `link.publics` never needs it.

Every function below is total: it returns a value or a `Reject` with a reason, and never loops on a length it has not
first compared with a length computed from `Config`. No function reads a clock, a random source, a file other than its
inputs, or the transcript's hash state. The verdict does not depend on the order in which independent checks run; this
document lists checks in the order upstream's pipeline reaches them, so that a `Reject` reason can be compared with
upstream's, but only `Accept`/`Reject` is normative.

The pipeline:

1. `statement ← setup(Config, PublicInput)` (§16): Σ, the statement digest, `Hello`, domains, the region list, the
   lincheck circuit, the PCS parameters. Pure in its inputs; it fails on malformed public input.
2. `session ← decode_record(Record)` and the session checks S1–S14 (§5.3).
3. For each table `t` and rep `r ∈ {0, 1}`: `proof ← decode_proof(Proofs[t, r])` (§8), a fresh replay transcript
   (§6), `verify_rep(statement, session, proof, transcript)` (§9–§14), then the stream-exhaustion check (S15) and the
   root check (S16).
4. The proof-file check S17 and the cross-rep check S18 (R1).
5. `Accept` iff every step passed.

## 4. Configuration pinned by the verifier

| item | value | source of the value |
|---|---|---|
| profile | `fast100` (Ligerito `LigeritoProfile::Fast100`, §15) | configuration; the proof's `commitment.params` must equal it byte for byte |
| reps | 2, both on one committed root | configuration (R1, R7) |
| flavour | `rs`: the univariate-skip zerocheck over the AES subfield (`SkipPoint::Phi8`), not the AG flavour | configuration (R7) |
| PCS | `PcsParams { m, log_inv_rate = 1, log_batch_size = k0(m), profile = Fast100, num_lanes = None, merkle_hash = Sha256 }` | computed from the statement's `m` (§15) |
| link | `points = 2`, `m_pts` coordinates per point, `y = 0` | statement (§16) |
| require_link | `true` | configuration (R5, E0, F1) |
| prime_live | `false` | none of these statements has a prime half |
| coin source | `os` (default) or `seed` (`coin-commit/sha256/v1`, PR #83 only when its `Hello` names it) | configuration (§7) |
| grinding | none: every proof-of-work nonce is a message and is never checked (R4) | configuration |

## 5. The session record

### 5.1 Format

The record is UTF-8 JSON. The verifier reads only the fields below; it ignores every other field, including `verdict`
(the live server's own opinion) and timing fields. Hex strings are lowercase, even-length.

| field | type | meaning |
|---|---|---|
| `kind` | string | must be `"flock-live-session/v1"` |
| `aborted` | null or string | must be null |
| `hello` | string | the exact `Hello` the prover sent |
| `config.require_link` | bool | must be `true` |
| `link_mode` | string | must be `"exchange"` |
| `link.mode` | string | must be `"exchange"` |
| `link.sigma` | hex(32) | Σ |
| `link.root_f`, `link.root_f_sha256` | hex, hex(32) | the committed root_F and its SHA-256 (G2: `root_f` present from G2 on) |
| `link.root_b.<t>`, `link.root_b_sha256.<t>` | hex, hex(32) | table `t`'s committed root_B: the flattened Merkle cap (§14) |
| `link.publics.<t>`, `link.publics_sha256.<t>` | hex, hex(32) | table `t`'s committed public words (§16) |
| `link.points` | list of `points` lists of `m_pts` F128 hex | the link points the server drew after `Commit` |
| `link.y` | list of `points · |tables|` F128 hex | y, table-major |
| `link_sha256` | hex(32) | SHA-256 over the exchange (S14) |
| `streams[i]` | object | one transcript stream: `stream`, `parent`, `label`, `at`, `root_sha256`, `rounds` |
| `streams[i].rounds[k]` | object | `k`, `n`, `msg_len`, `msg_sha256` (hex(32)), `coins` (list of `n` F128 hex) |
| `proof_sha256[j]` | object | `stream` (`<t>/rep<r>`), `sha256`, `bytes` |
| `verdict.coin_seed` | object | seed mode only: `scheme`, `domain`, `commitment`, `seed`, `nonce`, `coin_requests` (§7.2) |

An F128 hex string is 32 hex digits: the 128-bit integer `hi · 2^64 + lo` in big-endian (`{hi:016x}{lo:016x}`).
Everywhere else (transcript, wire, `link_sha256`) an F128 is the 16 bytes `le64(lo) ‖ le64(hi)` (§2.3).

Footnote (G2). Records written before G2 (the earliest pure-block cells, §18) lack `link.root_f`, `link.root_b` and
`link.publics`. For them the verifier takes `root_b.<t>` as the proof's flattened cap and requires
`SHA-256(root_b) = root_b_sha256.<t>`; takes root_F as Σ (the pure-block statement commits Σ in root_F's place);
and takes the public words from the `Publics?` input, which must satisfy `SHA-256(publics) = publics_sha256.<t>`.
Producing `Publics?` for a pre-G2 record means evaluating the witness from the rows; that belongs to the vector
tooling, never to the verifier.

### 5.2 Streams and rounds

A stream is one Fiat–Shamir-shaped transcript whose coins came from the verifier. Root streams are named
`<table>/rep<r>`, with `parent = ""`, `at = 0` and `label = domain(t, r)` (§16). Upstream can fork child streams
(`<parent>/f<i>`); none of the statements here reaches a fork (§6.4), so a record with a child stream is rejected.

Round `k` of a stream records what the prover committed before the server drew that round's coins: the SHA-256 and
length of the framed bytes absorbed since the previous coin request on that stream, the squeeze header included (§6),
and the `n` coins returned.

### 5.3 Session checks

Each check is a total predicate on the decoded record and the configuration. Failing any one rejects.

| id | check | condition it enforces |
|---|---|---|
| S1 | `kind = "flock-live-session/v1"` and `aborted = null` | the session completed |
| S2 | `hello = Hello(statement)` byte for byte (§16) | R7: profile, reps, flavour, tables, Σ, link shape and coin scheme are the verifier's |
| S3 | `config.require_link = true`, `link_mode = "exchange"`, `link.mode = "exchange"` | F1/E1: the statement was committed before any coin |
| S4 | `link.sigma = hex(Σ)` | the session is about this statement |
| U1–U3 | the record's `unit_draw` is the statement's (for several tables, the session's draw, §16.12), well formed, and the statement's instances are the drawn units (§7.3) | the drawn units are the proved ones |
| S5 | for every configured table `t`: `root_b.<t>` present, non-empty, and `SHA-256(root_b.<t>) = root_b_sha256.<t>` (keys for other tables are ignored, as upstream does) | L1/L3 |
| S6 | `SHA-256(root_f) = root_f_sha256` and `root_f` non-empty; for these statements `root_f = Σ` (intended divergence D1, §17.1) | L1: statements without a prime half commit Σ as root_F |
| S7 | for every configured `t`: `SHA-256(publics.<t>) = publics_sha256.<t>` (absent key = empty bytes; other keys ignored) | C4 prerequisites |
| S8 | `statement_publics(t, publics.<t>)` accepts (§16) | C4: the committed public words are consistent with the public input |
| S9 | `link.points` has exactly `points = 2` entries of exactly `m_pts` F128 each | the link shape |
| S10 | `link.y` has exactly `points · |tables|` entries, and every entry is zero | these statements have no cross-field link: y is pinned to 0 |
| S11 | the stream set is exactly `{ <t>/rep<r> : t ∈ tables, r ∈ {0,1} }`; each has `parent = ""`, `at = 0`, `label = domain(t, r)` and at least one round (intended divergence D2, §17.1) | R7 (no third rep, no lone rep), no fork on this path |
| S12 | in every stream, `rounds[k].k = k`, `1 ≤ n ≤ 65536`, `|coins| = n`, `msg_sha256` is 32 bytes, and retained bytes (if any) hash to `msg_sha256` | well-formed rounds |
| S13 | for every root stream `<t>/rep<r>`: `root_sha256 = SHA-256(root_b.<t>)` | R1/L1: every rep bound the one committed root_B |
| S14 | `link_sha256 = SHA-256(root_f ‖ root_b.<t> for t in lexicographic table order ‖ points ‖ y)`, points point-major, each F128 as 16 LE bytes, y likewise | the recorded exchange is the one the coins followed |
| S15 | after rep `r`'s verification, every round of `<t>/rep<r>` was consumed, and none was requested beyond the record | R2: no extra, missing or split round |
| S16 | the proof's flattened cap equals `root_b.<t>` | R1: the proof is about the committed root |
| S17 | `proof_sha256` lists exactly the streams `<t>/rep<r>`, and `SHA-256(Proofs[t, r]) = sha256` for each | the proofs are the ones the server received |
| S18 | the caps of rep 0 and rep 1 are equal (implied by S16 for both; kept as a separate check) | R1 |

S6 and S11 are stricter than upstream's `Server::from_record` and `Server::finish` (D1, D2 in §17.1). No honest record
differs between the two.

## 6. The replay transcript

The transcript is a byte stream per root stream. It never hashes to derive a challenge: every coin is read from the
record, after checking the bytes that preceded it (conditions R2, R4). This section is the whole interface between the
Flock verifier (§9–§14) and the session.

### 6.1 Framing (transcript-v2)

Every absorbed item is one frame: a 16-byte header `[op u8][kind u8][0u8; 6][len u64 LE]`, then the payload, zero-padded
to a multiple of 16 bytes.

| operation | op | kind | len | payload |
|---|---|---|---|---|
| `domain(label)` | 1 | 0 | byte length of `label` | `label`, padded |
| `observe_label(label)` | 2 | 0 | byte length | `label`, padded |
| `observe_f128(v)` | 3 | 1 | 1 | `v` as 16 bytes |
| `observe_f128_slice(vs)` | 3 | 2 | `|vs|` | each element as 16 bytes, concatenated (no padding needed) |
| `observe_bytes(b)` | 5 | 0 | byte length | `b`, padded |
| `sample_f128()` | 4 | 1 | 1 | none (the header ends the round) |
| `sample_f128_vec(n)` | 4 | 2 | `n` | none |
| `verify_pow(nonce, bits)` | as `observe_bytes(le64(nonce))` | | | returns true; `bits` is unused |

### 6.2 State and rounds

The replay state of one root stream is `(pending : Bytes, k : Nat)`, initially `(frame(domain(label)), 0)`. Absorbing
appends a frame to `pending`. A squeeze of `n` coins appends its header, then:

- if `n = 0`: returns no coins and consumes no round (the header stays in `pending`);
- otherwise, let `c = pending`, `pending := []`. Round `k` of the stream must exist, must have `n` coins, and must satisfy
  `SHA-256(c) = msg_sha256`; the squeeze returns its recorded coins and sets `k := k + 1`. Anything else rejects (R2).

Recording `msg_len` is informational; the verifier may check `|c| = msg_len` as a cheaper first test.

**Retained round bytes.** The coin server may retain each round's framed bytes in the record (`rounds[k].msg`, hex; the
field name is this verifier's proposal to PR #83's lane). When they are present, S12 requires `SHA-256(msg) = msg_sha256`,
and the squeeze requires `c = msg` byte for byte. R2 then holds without SHA-256 collision resistance. Upstream's
`from_record` ignores the field, so a record whose bytes differ from its digest is an intended divergence: the Lean
verifier rejects it at S12, and upstream accepts on the digests.

### 6.3 What the transcript does not contain

The transcript's hash function (upstream's chained BLAKE3 on the CPU prover, SHA-256 on the CUDA prover) never enters
the replay: only the framing does. So a CPU proof and a CUDA proof of the same witness replay identically.

### 6.4 Forks

Upstream's `fork(label)` squeezes two scalar coins on the parent and opens a child stream seeded with them. The
single-table block-R1CS path (§9–§14) makes no fork: the union prover's merged opening forks, this path's mixed-batch
opening does not (§12). A verifier for this path therefore implements no fork, and S11 rejects a record with a child.

## 7. Coin sources

The verifier reads coins from the record; how the server produced them decides only which assumptions soundness rests
on.

### 7.1 `os` (default)

The server draws `16 n` bytes from `/dev/urandom` for each round after recording that round's digest, and `16 · 2 · m_pts`
bytes for the link points after recording `Commit`. Each 16-byte group is one F128 (`lo = le64(bytes[0..8])`,
`hi = le64(bytes[8..16])`). Soundness assumes only that these bytes are uniform and unknown to the prover before it sent
the round.

M0 draws `os` coins under `FC_COINS=os`; `verify --coins os` binds its live-OS identity (`Tags.withCoins`).

### 7.2 `seed` (PR #83, `coin-commit/sha256/v1`, `coin-commit/sha256`, `coin-commit/sha512`)

The server draws a 32-byte `seed` and a 32-byte `nonce` when it accepts `Hello`, answers `Hello` with the commitment
`SHA-256("verity/flock-netlist/coin-commit/v1\0" ‖ nonce ‖ seed)` (as two F128 in the 16-LE-byte encoding, so the 32
commitment bytes in order), and derives every coin. The SHA-512 statement (`631567f7`) commits with
`SHA-512("verity/flock-circuit/coin-commit\0" ‖ nonce ‖ seed)`, sent as four F128. The derivation below is
`verity.ml.randomness` v2, on SHA-512 (one hash, Daniel, 6 Oct), with the domain `verity/flock-circuit/coins/v1`:

~~~text
frame(b bytes)  = 'b' ‖ be64(|b|) ‖ b
frame(s string) = 's' ‖ be64(|utf8 s|) ‖ utf8 s
frame(i nat)    = 'i' ‖ be64(|body|) ‖ body,  body = big-endian bytes of i without leading zeros (i = 0: one byte 00)
derive(src, dom, ctx) = first 32 bytes of SHA-512("verity/randomness/derive/v2\0" ‖ frame(b src) ‖ frame(s dom)
                                ‖ frame(b (concat over keys of ctx in byte order: frame(s key) ‖ frame(value))))
stream(key, parts, n) = first n bytes of  SHA-512(head ‖ be64(0)) ‖ SHA-512(head ‖ be64(1)) ‖ …
                        head = "verity/randomness/stream/v2\0" ‖ key ‖ frame(b (concat frame(part)))
key    = derive(seed, "verity/flock-netlist/coins/v1", {"hello": s hello})
coins_q(n) = stream(key, [s "coins", i q], 16 n)        q = 0, 1, 2, … in the order the server issued coin requests
~~~

M0's identity names the derivation (`hashes.coin_derivation`): `sha512 (verity.randomness)` for v2. The sessions recorded
before it name `sha256 (verity.randomness)` (the statements pinned `@f5df3bbc` and earlier) and derived with v1: SHA-256
for SHA-512 and `v1` for `v2` in both tags, the whole 32-byte digest as the key, and 32-byte stream blocks.

The link points are request `q = 0` (`2 · m_pts` coins in one request); every round's coins follow in the order the server
handled them. The verdict reveals `seed`, `nonce`, `commitment` and the request count.

For the verifier of record the seed changes nothing it checks: the coins are still read from the record. It adds two
optional checks that a prover (or an auditor) runs: the commitment opens to the revealed seed and nonce, and each
round's coins equal `coins_q`. The second needs every round's global request index `q`: M0's records carry it as `g`
since `34c3d0815`, and `flock-circuit replay-coins` runs both checks. Older ones lack it (rounds are grouped by stream,
and reps may interleave), so an auditor recomputes `q` only if the reps ran sequentially (points, then all of rep 0,
then all of rep 1). With a seed, soundness additionally assumes that the commitment hides the seed and that `stream` is
pseudorandom (SHA-512 as a PRF, `prf/sha-512`; SHA-256 for a v1 session); with `os` coins it needs neither.

### 7.3 The unit draw

The verifier draws the sampled units itself, in the clear, between registration and the session
(`note:20260927T0210Z-finding-unit-draw-placement`). Serving registers its roots; the verifier draws the unit set from
its operating system's randomness and sends it; the drawn units' statements are fixed; only then does the verifier send
its coin commitment. The draw is not under the coin commitment, and no hash derives it.

The samplers are `verity.ml.randomness`'s exact ones, read straight from the OS bytes (`Flock.Draw`, `flock-verify draw`).
`uniform(n)` takes big-endian words of `w = ⌈(bitlen(n) + 64) / 8⌉` bytes and returns the first `x < 2^(8w) − 2^(8w) mod n`,
mod `n`. `subset(k, n)` is the partial Fisher–Yates of `Key.subset`: for `i < k`, `j = i + uniform(n − i)`, swap `i` and
`j`, and return the first `k`, sorted. `bernoulli(p)` over `[0, N)` takes unit `u` when `uniform(den) < num`. A stream
that runs out is extended and the sampler rerun, so the draw stays exact.

The draw is the object

~~~text
{"law": "subset",     "population": N, "k": K,                       "units": [u_0 < u_1 < …]}
{"law": "bernoulli",  "population": N, "p": "num/den",               "units": [u_0 < u_1 < …]}
{"law": "stratified", "population": N, "k": K, "strata": [stratum…], "units": [u_0 < u_1 < …]}
{"law": "work",       "population": N, "k": K, "strata": [work stratum…], "units": [u_0 < u_1 < …], ["closure": [c_0 < c_1 < …]]}
~~~

**The stratified sampling design** is the default (Daniel, Sep 27; the encoding is
`note:20260927T1550Z-handoff-from-stratified-law`, the same as `verity.core.protocols.sampled_proofs.one_stage`'s, in
`verity/core/protocols/sampled_proofs/README.md`):
- **Strata.** The verifier derives them from its own program and partition object, never from anyone else
  (`HmRow.stratifiedLaw`, `flock-verify draw --program P --partition Q --stratified K`). It applies the query, groups the
  units by the template descriptor id of their root node, and writes each stratum's units as ascending, disjoint, non-empty
  ranges `[lo, hi)`, adjacent ones merged, the strata in the order of their first unit. Only `Q_template_instance` v0 and
  `Q_template_instances` v0 define strata; any other query refuses the sampling design.
- **Sizes.** For `1 ≤ K ≤ N`, stratum `s` of `n_s` units draws `k_s = max(1, ⌊(2 K n_s + N) / (2N)⌋)`: `K n_s / N` rounded
  half up, never below one. A stratum is `{"template": id, "units": [[lo, hi], …], "k": k_s}`, and `Σ k_s` units are drawn.
- **The sampler.** Each stratum draws `subset(k_s, n_s)` on its own stream of the verifier's randomness, of
  `2 k_s · w(n_s) + 64` bytes, extended and rerun on its own when it runs out. Local index `j` is the stratum's `j`-th unit,
  and the draw is the union, ascending. Given bytes (a test's `--stream`) are cut into consecutive per-stratum segments of
  those sizes, and aren't extended.
- A session that proves one template's units is served that template's share as a `subset` draw over its `n_s` units.
- **The verifier's own `K` and strata.** A draw's `K` and strata set every `k_s`, so `verify` never takes them as stated.
  It refuses a stratified draw unless it holds its own `K` (`--stratified K`), program and partition, refuses one drawn
  at any other `K`, and then holds the sampling design to its own derivation.

**The work sampling design** is the stratified one's second version (Daniel approved work-proportional sizing on Sep 28,
`note:20260929T0340Z-handoff-from-pous-rulings-2`). It keeps the stratified sampling design's strata and sampler; only
the sizes differ. The stratified sampling design is unchanged.
- **Work and floors.** Each template's PoUW work per unit, and its floor, come from the verifier's own work table
  (`--work-table F`: a JSON object from template descriptor id to the work, a natural number in any fixed unit, or to
  `{"work": w, "floor": f}` with `f ≥ 1`; the floor is one when the table gives only the work). Units that do no PoUW
  work, such as dequantization, the quantizer and attention, have work 0. A template the table does not name refuses the
  sampling design (`HmRow.workLaw`, `flock-verify draw --program P --partition Q --work K --work-table F`). A draw's
  `K`, works and floors set every `k_s`, so `verify` never takes them as stated. It refuses a work draw unless it holds
  its own `K` (`--work K`), work table, program and partition, refuses one drawn at any other `K`, and then holds the
  sampling design to its own derivation.
- **Sizes.** With `W = Σ_s w_s n_s ≥ 1` and `1 ≤ K ≤ N`, stratum `s` draws `k_s = min(n_s, max(f_s, ⌈K w_s n_s / W⌉))`
  (`Draw.workK`). This is its work share of `K`, rounded up. It is never above `n_s`, and never below its floor `f_s`. A
  work stratum is `{"template": id, "units": [[lo, hi], …], "work": w_s, "floor": f_s, "k": k_s}`.
- **Floors.** Every floor is at least one, and U2 refuses a floor of 0. With every floor one, the integrity floor binds
  exactly for the strata that do no work, and at most `K` plus one unit per stratum are drawn. In general at most
  `K + Σ_s f_s` are drawn. The table sets each stratum's floor by what the stratum needs: a stratum with `b_s` of its
  `n_s` units faulty has acceptance probability at most `((n_s − b_s)/n_s)^(f_s)`, so `f_s ≈ ln(1/δ)/ε_s` catches a
  faulty fraction `ε_s` of it up to `δ`. Floors of at least the stratified sampling design's sizes at `C` for the strata
  that do no work keep that design's guarantee for them.
- **What it guarantees** (`soundness/FlockSoundness/Audit/Work.lean`). Faulty units that do a fraction `ε` of the work
  have acceptance probability at most `(1 − ε)^K` (`work_escape_le`), whatever the floors. At the sizing of record,
  `δ = 2^-40` and `ε = 0.1%`, `K = 27,713` meets it (`record_sizing`) and `K = 27,712` does not
  (`record_sizing_one_fewer`). Rounding up keeps every unit's inclusion probability at least `K w_s / W`, which the bound
  needs. Every non-empty stratum draws at least one unit, so a wholly faulty stratum is caught
  (`audit_work_whole_stratum`). Each stratum's floor gives its own bound on acceptance probability
  (`work_escape_floor_le`, `audit_work_floor`). When the strata that do no work have floors of at least the stratified
  sampling design's sizes at `C`, a set's acceptance probability is at most that of its units in those strata under
  `stratified:C` (`work_escape_le_count`, `audit_work_rest`).
- **Closure draws.** A closure draw proves each drawn unit together with its closure: for a tile, the two strips it
  reads and the node units on its X strip's path. The closure comes from the verifier's own closure map (`--closure F`: a
  JSON object from a unit's index to the units its credit depends on).
  - The draw keeps its sampled `units`, so U2's per-stratum counts are unchanged, and adds `"closure"`: the units in the
    drawn units' closures that are not themselves drawn, ascending (`Draw.closureOf`).
  - The statement proves both (`UnitDraw.proved`), and U3 counts both.
  - `verify` refuses a closure draw unless it holds its own closure map, and then the closure must be its own derivation
    from the drawn units. With the map, a plain work draw is refused too, since it lacks its closure.
  - What it guarantees (`soundness/FlockSoundness/Audit/Closure.lean`): a set's acceptance probability under the closure
    draw is the work sampling design's for the set with the units it makes unsound, those whose closure meets it
    (`closure_escape`). The unsound work is the work of both: the closure map need not list a tile in its own closure,
    so the faulty units' own work is counted explicitly. A faulty unit's harm is its own work and that of every unit
    above it, so one that is never drawn on its own, such as a node, is charged all of it (`harm_le_unsoundWork`). At
    any floors, the audit accepts while the unsound work is at least a fraction `ε` of the work with probability at most
    `(1 − ε)^K + ε_ks + ε_bind` (`audit_work_closure`). At K = 27,713 and ε = 0.1% that is `2^-40`
    (`audit_work_closure_of_record`).
  - `Law.widen`'s `escape_widen_le` is weaker. It carries only the base sampling design's bounds on acceptance
    probability, and only to a widening on the same coins, and those bound the faulty units' own work, not the unsound
    work.

The statement carries it as the public header's `unit_draw` (so under `pub.sha` and Σ), instance `i` being unit `u_i`,
and the record keeps it as `unit_draw`. The offline checks:

| check | rule |
|---|---|
| U1 | the record keeps a draw iff the statement carries one, and they are canonically equal |
| U2 | the draw is well formed: exactly the sampling design's keys; units ascending, distinct and in `[0, N)`; `subset` has `K ≤ N` units; `bernoulli` has `0 ≤ num ≤ den`, `den > 0`, no unit at `p = 0` and every unit at `p = 1`; `stratified` has `1 ≤ K ≤ N`, each stratum exactly `{template, units, k}` with a distinct template, its ranges ascending, disjoint, not adjacent and non-empty, its `k` the rule's, the strata's ranges tiling `[0, N)` exactly, and exactly `k_s` units in each; `work` has the same checks, with each stratum exactly `{template, units, work, floor, k}`, a floor of at least one, `W ≥ 1`, each `k` the work rule's above its floor, and a closure (when there is one) of non-empty, ascending, distinct units of the population, none of them drawn. When the verifier holds the program and a template-query partition object, a stratified draw's sampling design must be its own derivation from them, and `verify` refuses a stratified draw without them and its own `K` (`--stratified K`), or at another `K`. A work draw's sampling design must always be its own derivation from them, its work table (works and floors) and its own `K`, and `verify` refuses a work draw without all four or at another `K` |
| U3 | the statement's `instances` is the number of drawn units and their closure. A work draw's closure must be the verifier's own derivation from its closure map, and `verify` refuses a closure draw without the map |
| U4 | every unit that reads a round's commitment by the verifier's own registered reads (a port whose value is `rec-c<i>`, `rec-top-L<l>`, `rec-opening-c<i>`, `rec-top-stacked-L<l>` or one of the sparse phase's messages `rec-sparse-<p>`, §16.10) is drawn in full: its stratum's `k` is its size (a `subset` draw of `K = N`, a `p = 1` draw). A statement without a draw passes |

`stratified_agree.py` checks the Lean verifier's sampling designs, draws and U2 verdicts against core's (#164's
`verity.core.protocols.sampled_proofs.one_stage.draw`, #161's `Stratified`).

Offline, the draw's uniformity is not checkable; soundness rests on the verifier's honest draw after registration. A
session without a draw on either side is unchanged. The live side (PR #83) sends the draw after registration, stages the
public header with it and records it.

### 7.4 The keyed coin tree (`coin-tree/hm96-sha512/v2`, `docs/coin-tree-v2.md`)

The verifier's coins are committed at `Hello` in a coin tree whose every hash is prefixed by the prover's fresh 32-byte
nonce. The server answers with the root and the session tree's hm96 key, drawn after the nonce, never reused, and never
revealed before that answer. `Flock.CoinTree` checks the record; the openings need no check, since
the commitment protects the prover (§7 of the spec).
- **`Hello`.** `coins` is `{"block", "nonce", "rounds_log", "scheme", "streams"}`. The verifier's own `Hello` (`helloOf`
  with `Coins.tree`) leaves the nonce empty.
- **R7** (`CoinTree.checkHello`). The received `coins.nonce` must be 64 lowercase hex digits. The verifier's
  `Hello` with it, serialized canonically, must equal the received string byte for byte.
- **The record's `coin_tree`** (`CoinTree.checkRecord`).
  - It has exactly `key`, `nonce`, `root` and `spec`.
  - The nonce is `Hello`'s. The key is 256 bytes with bit 2047 zero (bit 7 of byte 255), and the root 64 bytes, both
    lowercase hex.
  - The spec is `{"block", "rounds_log", "scheme", "streams"}`, the verifier's.
- **Freshness** (`CoinTree.checkFresh`). `verify` refuses a session whose key an earlier session of the same run used.
- **Scope.** Only `--zk` sessions (§16.14) carry one; the checks run once `Hello` names the scheme (`Stmt.spec`'s
  `coinTree`). `flock-verify coin-tree-check` runs them alone, on §9's values (`test_lean_coin_tree.py`).

## 8. The proof container

### 8.1 Encoding

A proof file is `bincode 1.3` (default configuration) of the pair `(Commitment, R1csProofLigerito)`:

~~~text
u8                      1 byte
u32 / enum variant      le32 (variant index, declaration order)
u64 / usize             le64
[u8; h] (H)             h bytes, no length: the Merkle scheme's digest width (§14), 32 or 64
F                       16 bytes: le64(lo) ‖ le64(hi)
K                       F(c0) ‖ F(c1)
(A, B)                  A ‖ B
Vec<T>                  le64(len) ‖ T × len
Option<T>               u8 tag (0 = None, 1 = Some) ‖ T if Some
struct                  its fields in the order below, no tags
~~~

Decoding is a total function `decode : Bytes → Except Reason Proof`. It rejects a truncated input, an enum index out of
range and an `Option` tag other than 0 or 1. Every `Vec` length is compared with the expected length of §8.3 before the
elements are read; `matrix_evals` (unused) is read with the length it declares, bounded by the input. Bytes after the pair
are ignored, as upstream's `bincode::deserialize` ignores them; the record's `proof_sha256` (S17) binds the whole file
anyway.

### 8.2 Layout

~~~text
Commitment          = cap: Vec<H> ‖ params: PcsParams
PcsParams           = m: u64 ‖ log_inv_rate: u64 ‖ log_batch_size: u64
                      ‖ profile: enum { Fast = 0, Fast100 = 1, Slim = 2, Slim100 = 3, Secure = 4 }
                      ‖ num_lanes: Option<u64> ‖ merkle_hash: enum { Sha256 = 0, Blake3 = 1 }
R1csProofLigerito   = zerocheck: ZerocheckProof ‖ lincheck: LincheckProof ‖ pcs_open: BatchOpeningProof
ZerocheckProof      = round1_ab: Vec<F> ‖ round1_c: Vec<F> ‖ multilinear_rounds: Vec<(F, F)>
                      ‖ final_a_eval: F ‖ final_b_eval: F ‖ final_c_eval: F ‖ grinding_nonces: Vec<u64>
LincheckProof       = rounds: Vec<(F, F)> ‖ z_partial: Vec<F> ‖ matrix_evals: Vec<(F, F)> ‖ grinding_nonces: Vec<u64>
BatchOpeningProof   = ring_switches: Vec<RingSwitchProof> ‖ batching_nonces: Vec<u64> ‖ ligerito: LigeritoProof
RingSwitchProof     = s_hat_v: Vec<F> ‖ grinding_nonce: u64
LigeritoProof       = initial_cap: Vec<H> ‖ initial_proof: LevelProof ‖ recursive_caps: Vec<Vec<H>>
                      ‖ recursive_proofs: Vec<LevelProof> ‖ final_proof: FinalProof
                      ‖ sumcheck_transcript: Vec<(F, F)> ‖ sumcheck_transcript_f256: Vec<(K, K)>
                      ‖ grinding_nonces: Vec<u64> ‖ ood_values: Vec<F> ‖ fold_grinding_nonces: Vec<u64>
                      ‖ claim_batch_grinding_nonces: Vec<u64> ‖ consistency_batch_grinding_nonces: Vec<u64>
LevelProof          = opened_rows: Vec<Vec<F>> ‖ merkle_proof: Vec<H>
FinalProof          = yr: Vec<F> ‖ opened_rows: Vec<Vec<F>> ‖ merkle_proof: Vec<H>
~~~

A multilinear round is `(G(1), G(∞))`; a lincheck round `(q(1), q(∞))`; a Ligerito message `(u0, u2)` in K.

### 8.3 Expected shape

With `m`, `k_log`, the claim count `N = 2 + K` (§11) and the schedule of §15 (`n = m − 7`, `k0`, `R` = levels − 1,
per-level `Q_ℓ`, `d_ℓ`, cap depth `c_ℓ`, `ext`):

| field | required |
|---|---|
| `params` | exactly `(m, 1, k0, Fast100, None, Sha256)` (R7/F2: a relabelled or other-profile proof is rejected) |
| `cap`, `initial_cap` | `2^{c_0} = 128` hashes; `initial_cap = cap` |
| `round1_ab`, `round1_c` | 64 each |
| `multilinear_rounds` | `m − 6` |
| lincheck `rounds`, `z_partial` | `k_log − 6`, 64 |
| every `grinding_nonces` of the PIOP, `batching_nonces`, `fold_grinding_nonces`, `sumcheck_transcript` | empty |
| `ring_switches` | `N`, each `s_hat_v` of 128 and `grinding_nonce = 0` |
| `initial_proof` | `Q_0` rows of `2^{k0}` elements; `Q_0 · (d_0 − c_0)` hashes |
| `recursive_caps` | `R` entries, entry `ℓ − 1` of `2^{c_ℓ}` hashes |
| `recursive_proofs` | `R − 1` entries, entry `ℓ − 1`: `Q_ℓ` rows of 16 elements, `Q_ℓ · (d_ℓ − c_ℓ)` hashes |
| `final_proof` | `yr` of `2^{ext+1}`; `Q_R` rows of 16; `Q_R · (d_R − c_R)` hashes |
| `sumcheck_transcript_f256` | `k0 + 8 + 7(R − 1)` |
| Ligerito `grinding_nonces` (query), `consistency_batch_grinding_nonces` | `R + 1` each |
| `ood_values` | `1 + 2R` |
| `claim_batch_grinding_nonces` | `3R + 2` |

Upstream checks the same counts lazily (at use, or by requiring exhaustion at the end); checking them up front gives the
same verdict. The nonces' values are messages (§6.1) and are not checked, except `grinding_nonce = 0` in each ring switch,
which upstream checks structurally.

Check at m = 32 (a recorded pure-block proof, 543,929 bytes): `cap` 128, `multilinear_rounds` 26, lincheck rounds 14,
14 ring switches, level-0 rows 218 × 64 with 2,834 hashes, recursive caps 64, 64, 32, 32, 32, `yr` 32, 42 messages,
11 out-of-domain values, 17 / 6 / 6 nonces; the file is consumed exactly.

## 9. Statement binding and zerocheck

Notation for this section and the next: `m`, `k_log` from the statement, `n_log = m − k_log`, `k_skip = 6`. A witness bit
address is `a = s + 64·t + 2^{k_log}·o` with the skip index `s ∈ [0, 64)` in bits 0–5, the inner rest `t` in bits
6 … k_log−1 and the block `o` in bits k_log … m−1 (upstream's `WitnessLayout::RowMajor`). The quirky extension `ẑ(Y, x)`
has degree `< 64` in `Y`, interpolating over the nodes `φ8(s)`, and is multilinear in `x ∈ F^{m−6}` over address bits
6 … m−1 in order.

Skip nodes: `S = φ8({0…63})`, a 6-dimensional F2-subspace of F; the output coset `Λ = φ8({64…127}) = δ + S`,
`δ = φ8(0x40)`. For a node set `N` (a subspace or coset of dimension `d`), `Z_N(X) = Π_{n∈N}(X + n)`,
`den_d = Π_{i=1}^{2^d−1} φ8(i)`, and the Lagrange weight `L^N_n(ζ) = Z_N(ζ)·den_d^{−1}·(ζ + n)^{−1}`, except that for
`ζ ∈ N` the weights are the indicator of `ζ`. `L_s(ζ)` means `L^S_{φ8(s)}(ζ)`.

### 9.1 Binding (round 0, first part)

~~~text
observe_label("flock-r1cs-v0")
observe_bytes(statement_digest)            32 bytes (§16)
observe_bytes(flatten(commitment.cap))     32 · |cap| bytes, the Merkle cap in order (§14)
~~~

`commitment.params` is not absorbed; it must equal the configured `PcsParams` (§8.3). No out-of-domain sample follows the
commitment (paper Remark 11).

### 9.2 Zerocheck transcript and checks

Shape checks first: `m ≥ 13`; `|round1_ab| = |round1_c| = 64`; `|multilinear_rounds| = m − 6`; `grinding_nonces = []`.

| step | operation | value |
|---|---|---|
| Z1 | `observe_label("flock-zerocheck-v0")` | |
| Z2 | `sample_f128_vec(6)` | `r_skip`: **unused** (no check reads any bit of it; see §19) |
| Z3 | `sample_f128_vec(m − 13)` | `r[13 … m−1]` |
| Z4 | `observe_f128_slice(round1_ab)`, `observe_f128_slice(round1_c)` | 64 elements each |
| Z5 | `z ← sample_f128()` | the univariate-skip challenge (the paper's λ) |
| Z6 for `i = 0 … m−7` | `observe_f128(g1_i)`, `observe_f128(ginf_i)`, `ρ_i ← sample_f128()` | residual rounds |
| Z7 | `observe_f128(final_a_eval)`, `observe_f128(final_b_eval)` | after the final check |

The zerocheck vector is `r ∈ F^{m−6}` indexed by address bits 6 … m−1: seven pinned coordinates, then `r[13…]` from Z3.

| address bit | value (hex `hi‖lo`) | origin |
|---|---|---|
| 6 | `φ8(0xF7)` = `1cf7a0fe8922c83fd8a5ae31928b4da1` | `α/(1+α)` in GF(2^8) |
| 7 | `φ8(0x53)` = `ffdb65d9987f058ca0415e708193f42a` | `α²/(1+α²)` |
| 8 | `φ8(0xB5)` = `44020450758a036626cb812e5dc6f8a5` | `α⁴/(1+α⁴)` |
| 9 | `x/(1+x)` = `ffffffffffffffffffffffffffffff83` | GHASH, `y = x` |
| 10 | `x²/(1+x²)` = `5555555555555555555555555555557f` | `y = x²` |
| 11 | `x⁴/(1+x⁴)` = `eeeeeeeeeeeeeeeeeeeeeeeeeeeeee9a` | `y = x⁴` |
| 12 | `x⁸/(1+x⁸)` = `d3d3d3d3d3d3d3d3d3d3d3d3d3d3d3b9` | `y = x⁸` |

These are the paper's §4.3 two-level construction (weights `C·φ8(α)^k·x^j`, `k < 8`, `j < 16`); their 128 eq weights have
F2-rank 128, the hypothesis the paper's argument needs. The verifier never forms those weights: the pinned values enter only
as `r_eq` in rounds 0–6 and as coordinates of the c-claim point.

Computation after Z5:

~~~text
Pc(z)    = Σ_{i<64} L^Λ_{φ8(64+i)}(z) · round1_c[i]
check    Pc(z) = final_c_eval                                                        (CEvalMismatch)
Pcomb(z) = Σ_{i<64} Z_{S∪Λ}(z) · den_7^{−1} · (z + φ8(64+i))^{−1} · (round1_ab[i] + round1_c[i])
           (the unique polynomial of degree < 128 that is 0 on S and round1_ab + round1_c on Λ; indicator rule if z ∈ S ∪ Λ)
claim_0  = Pcomb(z) + Pc(z)                                                          (= Pab(z))
~~~

So "P_ab − P_c vanishes on S" is imposed by construction, not checked, and the interpolant may have degree 127 (the paper
states 126); the Schwartz–Zippel term at `z` is `127/|F|`.

Residual round `i` binds address bit `6 + i` with `r_eq = r[6 + i]`. The prover sends `(g1, ginf) = (G(1), G(∞))` of the
eq-stripped quadratic `G`:

~~~text
G(0)   = (claim + r_eq · g1) · (1 + r_eq)^{−1}          (inv(0) = 0: r_eq = 1 gives G(0) = 0, as upstream)
claim ← G(0)·(1 + ρ_i) + g1·ρ_i + ginf·ρ_i·(1 + ρ_i)
~~~

No per-round check. Final: `claim = final_a_eval · final_b_eval` (`SumcheckFinalFailed`). Outputs: `z`,
`ρ = (ρ_0 … ρ_{m−7})`, `r_rest = r[6 … m−1]`, `v_a = final_a_eval`, `v_b = final_b_eval`, `v_c = final_c_eval`.

## 10. Lincheck

Inputs: `x_ab = (z, ρ[0 … k_log−7], ρ[k_log−6 … m−7])`, i.e. skip point `z`, inner rest `ρ_in` (`k_log − 6` values) and
outer `ρ_out` (`n_log` values); `v_a`, `v_b`. Shape checks: `|rounds| = k_log − 6`, `|z_partial| = 64`,
`grinding_nonces = []`, and the circuit has `2^{k_log}` columns. `matrix_evals` is not read: every honest proof sends it
empty, and upstream ignores it, as does the spec.

| step | operation | value |
|---|---|---|
| L1 | `observe_label("flock-lincheck-v0")`, `α ← sample_f128()` | |
| L2 | `β ← sample_f128()` | the constant-wire coin |
| L3 for `j = 0 … k_log−7` | `observe_f128(q1_j)`, `observe_f128(qinf_j)`, `t_j ← sample_f128()` | |
| L4 | `observe_f128_slice(z_partial)` (64 elements) | |
| L5 | `r' ← sample_f128()` | the fresh skip coin |

Computation:

~~~text
e[s + 64·u]  = L_s(z) · eq(ρ_in, u)                  s < 64, u ∈ {0,1}^{k_log−6} (LSB first); length 2^{k_log}
comb[c]      = Σ_i e[i] · (α · A_0[i, c] + B_0[i, c])        for every column c < 2^{k_log}   (§16.1)
comb[pin]   += β
claim        = α · v_a + v_b + β
for j = 0 … k_log−7:                                  (round j binds column bit k_log−1−j: top bit first)
    e0 = claim + q1_j;   c1 = e0 + q1_j + qinf_j;   claim ← qinf_j · t_j² + c1 · t_j + e0
    h = |comb| / 2;  comb[i] ← comb[i] + t_j · (comb[i + h] + comb[i])  for i < h;  truncate comb to h
check        Σ_{s<64} comb[s] · z_partial[s] = claim                               (ConsistencyFailed)
w            = Σ_{s<64} L_s(r') · z_partial[s]
r_in'        = (t_{k_log−7}, …, t_0)                   (reversed: r_in'[j] is the coin for address bit 6 + j)
~~~

The `β` term is sound only if the pin column is 1 in every block, padding blocks included; every statement wires every
slot's constant to the pin and pads with blocks whose constant is 1. The paper's lincheck ends in claims on `Â_0`, `B̂_0`
that the verifier checks; here the verifier computes `comb` itself and the prover sends all 64 skip evaluations, so the
lincheck's last univariate step has degree 63 (error `63/|F|`).

## 11. The claims handed to the opening

~~~text
ab:  value w,             skip L_s(r'), x = r_in' ‖ ρ_out                      (|x| = m − 6)
c:   value final_c_eval,  skip L_s(z),  x = r_rest = r[6 … m−1]
e_k: the statement's extra claims (§16.2), in order
~~~

The opening (§12) receives `[ab, c, e_1 … e_K]`. Before it, the verifier checks `commitment.params = PcsParams(m)` (§8.3).

## 12. Opening: claim batching and ring switching

Input: the claim list `claims = [ab, c, e_1, …, e_K]` (§11), `n = 2 + K`, each claim `(value α_k ∈ F, skip_k ∈ F^64,
x_k ∈ F^{m−6})`, and `proof.pcs_open = BatchOpeningProof` (§8). The output is one inner-product claim handed to Ligerito
(§13). The paper's Appendix B treats one claim with its own sumcheck; this configuration ring-switches each claim with its
own coins, batches the results with independent coins, and lets Ligerito's sumcheck play the role of the paper's.

### 12.1 Packing

Write a witness bit address as `a = 128·y + v`, `v ∈ [0, 128)`, `y ∈ [0, 2^ℓ)`, `ℓ = m − 7`. The committed packed polynomial
is `f(y) = Σ_v z[128y + v] · x^v`: the basis is `b_v = x^v` in GHASH form, and bit `v` of `f(y)` (as a 128-bit integer) is
`z[128y + v]`. So the packed variables are the 7 low address bits: bits 0–5 (`σ`) are the univariate-skip variables and bit 6
is the first multilinear coordinate; `y`'s bit `j` is address bit `7 + j`.

A claim's weight on the 128 packed positions is

~~~text
w_k[v] = skip_k[v mod 64] · eq(x_k[0], ⌊v / 64⌋)          eq(ρ, 0) = 1 + ρ,  eq(ρ, 1) = ρ
~~~

which is the paper's `w(σ, b) = L_σ(ζ)·eq(ρ, b)` (B.3).

### 12.2 Transcript and checks

| step | operation | check |
|---|---|---|
| O1 | `observe_label("flock-pcs-open-batch-v0")` | `|proof.ring_switches| = n`; `proof.batching_nonces = []` |
| for `k = 0 … n−1`: | | |
| O2 | `observe_label("flock-ring-switch-v0")` | `|s_k| = 128` (upstream asserts; the spec rejects) |
| O3 | `observe_f128_slice(s_k)` (one frame of 128 elements) | `Σ_v w_k[v]·s_k[v] = α_k` |
| O4 | (no transcript operation) | `ring_switches[k].grinding_nonce = 0` |
| O5 | `r''_k ← sample_f128_vec(7)` | |
| O6 | `γ ← sample_f128_vec(n)` (one squeeze) | |

Claim values and points are not absorbed here. For `ab` and `c` they are functions of earlier absorbed messages and coins.
The extra claims' points come from the link points and their values from the committed public words (§16), both fixed
before any Flock coin by the session order (S3, R5). A verifier implementation must not add absorbs here.

### 12.3 Recombination

For each claim, view `s_k` as a 128 × 128 bit matrix `M[v][u] = bit_u(s_k[v])` and repack each column:
`t_k[u] = Σ_v M[v][u] · x^v` (bit `v` of `t_k[u]` is bit `u` of `s_k[v]`). Then

~~~text
β_k = Σ_{u<128} eq(r''_k, u) · t_k[u]        eq(r, u) = Π_{i<7} (r_i if bit_i(u) else 1 + r_i)
T   = Σ_k γ_k · β_k
~~~

### 12.4 The batched basis

With `z_k = x_k[1 .. m−6] ∈ F^ℓ` and `Φ_k : F → F` the F2-linear map with `Φ_k(x^u) = eq(r''_k, u)`:

~~~text
B_k(y) = Φ_k( eq(z_k, y) )          for y ∈ {0,1}^ℓ        (paper Claim 6)
b(y)   = Σ_k γ_k · B_k(y)
~~~

Ligerito (§13) consumes `b` only through its multilinear extension at points `(ρ, y')` with `ρ ∈ K^{ℓ−e}` (K = GF(2^256),
§2.4) and `y' ∈ {0,1}^e`, where `e` is the profile's clear-residual length (§15). The pairing is `ρ_j ↔ z_k[j]` and bit `j`
of `y'` ↔ `z_k[ℓ − e + j]`. Any correct algorithm may compute it. The reference form is the paper's width-128 branching
program (B.4), lifted to K:

~~~text
B̂_k(q) = Σ_{u<128} eq(r''_k, u) · S_u,  where S ∈ K^128 starts at e_0 and for j = 0 … ℓ−1:
    S'_l = Σ_{u} bit_l( (1 + z_k[j]) · x^u ) · S_u  +  q_j · S_l
~~~

(For Boolean `q_j` the step is multiplication of the F factor by `z_k[j]` or `1 + z_k[j]`.) Upstream computes the same value
with the DP24 tensor algebra over `K ⊗_{F2} F`; the results are equal.

### 12.5 Hand-off

Ligerito must decide `Σ_{y ∈ {0,1}^ℓ} f(y)·b(y) = T` for the committed `f` (§13). `T ∈ F` enters Ligerito's transcript as
an F element and its running claim as `T ∈ K`.

### 12.6 Rejections

- `|ring_switches| ≠ n`; `batching_nonces ≠ []`; `|s_k| ≠ 128`; a claim check fails; `grinding_nonce_k ≠ 0`.
- `|x_k| ≠ m − 6` or `|skip_k| ≠ 64` for any claim (upstream checks the extras against `c`'s length; the spec checks all).
- Ligerito rejects (§13).

## 13. Ligerito (dense opening)

Input: `T ∈ F`, the basis oracle `b̂` (§12.4), the commitment cap, `proof.pcs_open.ligerito`, and the schedule for `n = m − 7`
(§15). The committed table is `t : {0,1}^n → F` (`t(y) = f(y)`, §12.1). The verifier accepts iff (up to soundness)
`Σ_y t(y)·b(y) = T`. This is the paper's Appendix C with four changes, each in §19 (A11): claims and folding challenges in
`K = GF(2^256)`, so each recursive level commits the *split* of a K-table (`f = f_0 + u·f_1` becomes the F-table
`g(b, x) = f_b(x)` with a new low variable `b`); stratified queries; a Merkle cap instead of a root; and an explicit
out-of-domain sample at level 0 during the opening.

### 13.1 Commitment and code

- Level 0: `k0 = initial_k` lanes (rows) and `cols_0 = n − k0` message columns; lane `ℓ` is the low `k0` bits of the table
  index and column `c` the high bits: `M[ℓ][c] = t[c · 2^{k0} + ℓ]`. The first `k0` variables are folded first.
- Reed–Solomon over F, additive (LCH novel basis): basis `v_i = x^i`, domain `ω_q = x`-polynomial whose coefficient bits are
  `q` (`ω_q = (lo = q, hi = 0)`), `q < 2^d`, `d = cols + log_inv_rate`, natural order.
  `s_0(X) = X`, `s_{i+1}(X) = s_i(X) · (s_i(X) + s_i(v_i))`, `Ŵ_i = s_i / s_i(v_i)`, `X̂_c = Π_{i ∈ bits(c)} Ŵ_i`.
  Codeword `C[ℓ][q] = Σ_c M[ℓ][c] · X̂_c(ω_q)`. The verifier never encodes; it evaluates `X̂` at opened positions (§13.4).
- Each level's commitment is a Merkle cap (§14): leaf `q` hashes the column `(C[0][q], …, C[L−1][q])`. Level `ℓ ≥ 1` commits
  its split F-table with `k_ℓ = 4` lanes (16) and the rate of the schedule.

### 13.2 Round algebra

A sumcheck message is `(u0, u2) ∈ K²`, absorbed as `observe_f256(u0)`, `observe_f256(u2)` (two frames). With running claim
`s ∈ K` it defines `q(X) = u0 + (s + u2)·X + u2·X²`, so `q(0) + q(1) = s` holds by construction and no per-round check
exists. A challenge `r ← sample_f256()` sets `s ← q(r)`. Introducing a new claim `v` with its own message `(u0', u2')`:
`q ← q + β·q_intro` coefficient-wise (`q_intro` built from the message with claim `v`), `s ← s + β·v`, `β ∈ F`.

### 13.3 Transcript

Counters start at 0; "the next X" reads list X of the proof and rejects if it is exhausted. `PoW(list)` is
`observe_bytes(le64(next nonce of list))` (a message; never checked, R4). Preconditions: `R ≥ 1`;
`initial_cap = commitment.cap`; `sumcheck_transcript = []`; `sumcheck_transcript_f256 ≠ []`; `fold_grinding_nonces = []`;
every recursive `k ≥ 2`.

Level 0:

1. `observe_label("flock-ligerito-basis-f256-split-v0")`; `observe_f128(T)`; `observe_bytes(flatten(initial_cap))`.
   `s ← T` (in K).
2. Out-of-domain, once: `z ← sample_f128_vec(n)`; `y ← next ood_value`; `observe_f128(y)`; `PoW(claim)`;
   `β ← sample_f128()`; `s ← s + β·y`. Record context `(z, β, split = none)`. No message.
3. Read and absorb the next message; `q ← from(msg, s)`.
4. `k0` times: `r ← sample_f256()`; `s ← q(r)`; read and absorb the next message; `q ← from(msg, s)`. The `r`'s are the
   initial challenges. The last message is the round polynomial of the level-1 split coordinate.
5. `split_dim ← n − k0 + 1`; `cap_1 ← recursive_caps[0]`; `observe_bytes(flatten(cap_1))`.
6. Out-of-domain, twice: `z ← sample_f128_vec(split_dim)`; `y ← next ood_value`; `observe_f128(y)`; read and absorb the
   next message; `intro ← from(msg, y)`; `PoW(claim)`; `β ← sample_f128()`; `q ← q + β·intro`; `s ← s + β·y`. Context
   `(z, β, split = 0)`.
7. Queries on level 0: `PoW(query)`; `w ← sample_f128_vec(Q_0)`; positions by §13.5 over `2^{d_0}`.
8. `PoW(consistency)`; `α ← sample_f128_vec(⌈log2 Q_0⌉)`.
9. Open the rows against `initial_cap` (§14): `2^{k0}` elements per row.
10. `e ← Σ_{j<Q_0} eq(α)[j] · Σ_ℓ eq_K(initial challenges)[ℓ] · row_j[ℓ]`.
11. Read and absorb the next message; `intro ← from(msg, e)`; `PoW(claim)`; `β ← sample_f128()`; `q ← q + β·intro`;
    `s ← s + β·e`. Consistency context `(cols = n − k0, positions, α, β, start = 0)`.
12. `prev ← (cap_1, lanes = k_1, cols = split_dim − k_1, rate_1)`; reject if `split_dim < k_1`.

Recursive round `i = 0 … R−1` (level `i + 1`, `k = recursive_ks[i]`): reject if `split_dim < k`; `k` times
`r ← sample_f256()`, `s ← q(r)`, read and absorb the next message, `q ← from(msg, s)` (these are `level[i]`; `level[i][0]`
folds the split coordinate); `ext ← split_dim − k`. Then:

- Not last (`i + 1 < R`): `split_dim ← ext + 1`; the next cap, absorbed; two out-of-domain samples as step 6 (context
  `split = i + 1`); queries on the previous commitment (`PoW(query)`, `Q_{i+1}` coins); `PoW(consistency)`,
  `α ← sample_f128_vec(⌈log2 Q_{i+1}⌉)`; open the next `recursive_proofs` entry against the previous cap (16 lanes);
  `e ← Σ_j eq(α)[j] · Σ_ℓ eq_K(level[i])[ℓ] · row_j[ℓ]`; read and absorb a message; `PoW(claim)`; `β`; batch as step 11
  (context start `i + 1`); `prev ←` the new cap with its lanes, cols and rate; reject if `split_dim < lanes`.
- Last (`i + 1 = R`): reject unless every message was consumed and `|yr| = 2^{ext+1}`; `observe_f128(yr[t])` for each word
  in order (one scalar frame each); queries on `cap_R`; `PoW(consistency)`, `α`; open `final_proof` against `cap_R`;
  `e` as above; `PoW(claim)`; `β ← sample_f128()`; `s ← s + β·e` (no message); the final check (§13.6).

The message read after the last challenge is absorbed and required to exist but enters no equation; the claim used is the
one before it.

### 13.4 Consistency weights

For an opened level with `2^k` lanes and `L` columns, the verifier needs the multilinear extension of `c ↦ X̂_c(ω_q)` at a
point `p ∈ K^L`:

~~~text
MLE(X̂_·(ω_q))(p) = Π_{i<L} (1 + p_i · (1 + Ŵ_i(ω_q)))
Ŵ_i(ω_q) = s_i(ω_q) / s_i(v_i),   s_0(ω) = ω,   s_{i+1}(ω) = s_i(ω)² + s_i(v_i) · s_i(ω)
~~~

### 13.5 Query positions (the only coins read partially)

The schedule decomposes `Q` greedily into powers of two: `c_1 ≥ c_2 ≥ …` with `Σ 2^{c_i} = Q`, `c_i = min(⌊log2 rem⌋, d)`.
Queries are listed summand by summand, stratum `s = 0 … 2^{c_i} − 1` within each summand. Query `j` in stratum `s_j` of a
summand of depth `c_j` has position

~~~text
pos_j = (s_j << (d − c_j)) | (coin_j.lo mod 2^{d − c_j})
~~~

It reads only the low `d − c_j` bits of the coin's `lo` word; bits `d − c_j … 127` are unused (the recorded negative
`unused_high_bits_of_query_coin` must be accepted). Sampling is with replacement: no deduplication, no rejection.

### 13.6 Final check

With `cff(c) = 1 + c·(1 + u)` and `Π_{≥a} = Π_{s ≥ a} cff(level[s][0])`:

1. `orig ← initial challenges ‖ level[0][1..] ‖ … ‖ level[R−1][1..]`; reject unless `|orig| + ext = n`.
2. `res[y] ← b̂(orig ‖ bits(y)) · Π_{≥0}` for `y < 2^{ext}` (§12.4).
3. Each out-of-domain context adds `β · scale · Π_{t<|fixed|}(1 + z_t + fixed_t) · Π_{j<ext} eq(z_{|fixed|+j}, bit_j(y))`,
   with `(fixed, scale) = (orig, Π_{≥0})` for the level-0 sample, and `(level[s] ‖ level[s+1][1..] ‖ … ‖ level[R−1][1..],
   Π_{≥s+1})` for `split = s`.
4. Each consistency context `(L, positions, α, β, start a)` adds
   `β · Π_{≥a} · Σ_j eq(α)[j] · P_j · T_j(y)`, where `fixed = level[a][1..] ‖ … ‖ level[R−1][1..]` (empty for `a = R`),
   `P_j = Π_{i<|fixed|}(1 + fixed_i · (1 + Ŵ_i(ω_{pos_j})))`, `T_j(y) = Π_{i<ext} (Ŵ_{|fixed|+i}(ω_{pos_j}) if bit_i(y) else 1)`.
5. Accept iff `Σ_y res[y] · (yr[2y] + u · yr[2y+1]) = s` and every list is exhausted: `ood_values`, every nonce list,
   `recursive_caps`, `recursive_proofs`.

## 14. Merkle trees

The scheme is a parameter (`Flock.MerkleScheme`), named by the circuit file's `leaf_scheme.id`. It fixes the digest width
`h`, the leaf, the node, and the `PcsParams.merkle_hash` index the proof must carry (§8.3):

| id | h | leaf | node | `merkle_hash` |
|---|---|---|---|---|
| `flock-leaf/sha256-unsalted/v0`, `flock-leaf/sha256-unsalted` | 32 | `SHA-256(e_0 ‖ … ‖ e_{L−1})` | `SHA-256(left ‖ right)` | 0 (`Sha256`) |
| `flock-leaf/sha512-unsalted` (PR #83 from `b3baabd8`) | 64 | `SHA-512(e_0 ‖ … ‖ e_{L−1})` | `SHA-512(left ‖ right)` | 2 (`Sha512`) |

Each element is 16 bytes (§2.1), with no prefix, index or length, and no leaf/node domain separation (Flock's SHA-256
format, widened). A 64-byte digest is serialized as its 64 bytes (Rust `[[u8; 32]; 2]`), and the transcript observes each
cap's 64-byte nodes. The target's hiding leaf is `hm96-sha512/v1` (PR #93, pinned in META `leaf_scheme.target`): each opened
row then carries its 192-byte salt, in a layout PR #83's lane will publish with that change.
- The commitment is the cap: the `2^c` nodes at depth `c` below the (never formed) root, left to right, with
  `c = ⌊log2 Q⌋` for the level (7 at level 0: 128 hashes, 4096 bytes), clamped to `d`.
- A level's opening: `opened_rows` in query order (duplicates repeated), and `merkle_proof` = `Q · (d − c)` sibling hashes,
  flat, in query order. Checks: `|rows| = |positions| = Q`, `|cap| = 2^c`, `|siblings| = Q·(d − c)`, each row has the level's
  lane count, and every cap entry and sibling is `h` bytes. For each query, `acc ← leaf(row)`; then for each of the `d − c`
  siblings, `acc ← node(acc, sib)` if the current index is even, else `node(sib, acc)`, and the index is halved. Require
  `acc = cap[pos >> (d − c)]`.

## 15. Parameters: `fast100`

Schedules exist for `m = 22 … 35` only; the verifier rejects any other `m`. Every level has query and fold grinding 0,
`η = 0.02`, the Johnson regime, 1 out-of-domain sample at level 0 and 2 at every other level, `k = 4` at levels ≥ 1.

| level | rate | Q | strata `c_i` | cap | claim PoW bits | consistency PoW bits |
|---|---|---|---|---|---|---|
| 0 | 1/2 | 218 | 7, 6, 4, 3, 1 | 7 | 6 | 9 |
| 1 | 1/4 | 106 | 6, 5, 3, 1 | 6 | 6 | 9 |
| 2 | 1/8 | 71 | 6, 2, 1, 0 | 6 | 7 | 9 |
| 3 | 1/16 | 53 | 5, 4, 2, 0 | 5 | 7 | 10 |
| 4 | 1/32 | 43 | 5, 3, 1, 0 | 5 | 8 | 10 |
| 5 | 1/64 | 36 | 5, 2 | 5 | 8 | 11 |
| 6 | 1/128 | 32 | 5 | 5 | 9 | 11 |

Exception: at `m = 22` level 2 has rate 1/16 (Q 53, strata 5, 4, 2, 0).

| m | k0 | levels | cols per level | d per level | `ext` (`|yr| = 2^{ext+1}`) |
|---|---|---|---|---|---|
| 22 | 6 | 3 | 9, 6, 3 | 10, 8, 7 | 3 |
| 23 | 6 | 3 | 10, 7, 4 | 11, 9, 7 | 4 |
| 24 | 6 | 3 | 11, 8, 5 | 12, 10, 8 | 5 |
| 25 | 6 | 4 | 12, 9, 6, 3 | 13, 11, 9, 7 | 3 |
| 26 | 6 | 4 | 13, 10, 7, 4 | 14, 12, 10, 8 | 4 |
| 27 | 6 | 4 | 14, 11, 8, 5 | 15, 13, 11, 9 | 5 |
| 28 | 4 | 5 | 17, 14, 11, 8, 5 | 18, 16, 14, 12, 10 | 5 |
| 29 | 5 | 5 | 17, 14, 11, 8, 5 | 18, 16, 14, 12, 10 | 5 |
| 30 | 6 | 5 | 17, 14, 11, 8, 5 | 18, 16, 14, 12, 10 | 5 |
| 31 | 6 | 6 | 18, 15, 12, 9, 6, 3 | 19, 17, 15, 13, 11, 9 | 3 |
| 32 | 6 | 6 | 19, 16, 13, 10, 7, 4 | 20, 18, 16, 14, 12, 10 | 4 |
| 33 | 6 | 6 | 20, 17, 14, 11, 8, 5 | 21, 19, 17, 15, 13, 11 | 5 |
| 34 | 6 | 7 | 21, 18, 15, 12, 9, 6, 3 | 22, 20, 18, 16, 14, 12, 10 | 3 |
| 35 | 6 | 7 | 22, 19, 16, 13, 10, 7, 4 | 23, 21, 19, 17, 15, 13, 11 | 4 |

Shape identity: `k0 + 3·R + ext = n`. `log_batch_size = k0` in `PcsParams`. The table is data (`Flock.Schedule`); the TOML
files' `expected_eps_*` fields, `target_security_bits`, `analysis_version` and `hash` are diagnostics the verifier never reads,
and the Merkle hash is SHA-256 whatever the TOML says.

**Soundness per table.** Per run, the query terms dominate: `Q · log2(1/(√ρ + η))` exceeds 100 bits at every level (100.23,
100.00, 100.86, 100.12, 100.85, 100.29, 102.58), and the union over the levels and every other round-by-round term
(zerocheck, lincheck, ring switch, out-of-domain, claim and consistency batching, sumcheck) is 2^-97.8. No grinding credit is taken: under live coins a nonce buys nothing. Two
reps on one root with independent live coins give **2^-195.5 per table**, round by round, whatever the interleaving of the
reps.

## 16. Statements

Each statement supplies `setup(Config, PublicInput)`:

~~~text
Statement = { table, domain : Nat → Bytes, Σ : Bytes32, digest : Bytes32, hello : String,
              r1cs : { m, k_log, k_skip = 6, pin }, circuit : BlockMatrices, m_pts,
              regions : List Region, publicsCheck : Bytes → Except Reason Unit,
              extraClaims : Points × Publics → List ExtraClaim }
~~~

All three statements share the session shape: `Commit(root_F = Σ, root_B, publics)`, then two link points of `m_pts`
coordinates, then `Link(y = 0, 0)`, then Flock's coins. Nothing links to a second field, so `y` is pinned to zero (S10) and
the points serve only the region claims. Byte strings below are quoted exactly as the implementation spells them; some
still carry the retired word "netlist", and renaming them changes Σ and the statement digests.

### 16.1 Block matrices

Every statement proves one block R1CS: `A·z ∘ B·z = z` over F2 with `A = I_{2^nbl} ⊗ A_0`, `B = I_{2^nbl} ⊗ B_0`
(`C = I`). A block is `2^{k_log}` witness bits; the witness has `2^nbl` blocks, `m = k_log + nbl`.

`A_0` and `B_0` are `2^{k_log} × 2^{k_log}` over F2, assembled from:

1. **Slots.** The block is cut into aligned slots. Each slot holds one copy of a sub-circuit's R1CS: a BLAKE3 compression
   (`2^14` rows, upstream's `blake3::build_block_r1cs(3)`), a SHA-256 compression (`2^15` rows, `sha2::build_block_r1cs(3)`),
   a unit (an expanded circuit in `flock-unit-io/v1` or `flock-ir-unit/v2` form, `2^{unit_log}` rows), a lookup circuit
   (§16.5), or a mask slot (`A = B = I`). Rows no slot covers are forced zero (`A = B = 0`, so `z = 0`).
2. **Δ.** A list of `(row, col)` pairs for A and one for B, XORed into the slot matrices. The convention: pushing `(i, i)`
   removes the base input row `i` (which was `A = [i]`), pushing `(i, src)` makes it read `z_src`. "Both" pushes into A and
   B (so `z_i = z_src · z_src = z_src`); "A only" pushes into A with B left at the pinned constant column (so
   `z_i = z_src · 1`). Constants are wired as `A = [bits · pin]`.
3. **The pin.** One column fixed to 1 (`const_pin`); every slot's own constant column is wired to it. The verifier refuses a
   pin outside the block (`pin ≥ 2^{k_log}`).

The verifier needs `Â_0` and `B̂_0` at one point per rep (§10). Any algorithm that computes the multilinear extension of
the assembled sparse matrices is correct. Upstream's evaluates each slot type once on the first slot's eq tensor and
rescales it per slot by a ratio of eq entries, then adds Δ's entries; the result is the same.

The compression circuits are upstream data. The verifier holds them as pinned files (their `A` and `B` row lists; `C = I`)
and recomputes the digest the statement digests embed (`D_b3`, `D_sha`), upstream's circuit digest:

~~~text
D = BLAKE3("flock-r1cs-stmt-v2" ‖ le64 m ‖ le64 k_log ‖ le64 k_skip ‖ le64 useful_bits ‖ u8 [pin present] ‖ le64 (pin or 0)
           ‖ u8 layout (RowMajor = 0) ‖ Mat(A) ‖ Mat(B) ‖ Mat(C))
Mat(X) = le64 rows ‖ le64 cols ‖ for each row: le64 (number of entries) ‖ le64 column × entries
~~~

The BLAKE3 compression circuit has `m = 17`, `k_log = 14`, `k_skip = 6`, pin 11706. Level 3 proves each pinned expanded
circuit correct against its reference function (the lowering theorem).

### 16.2 Region claims

A region is a set of witness bits in every block whose values the verifier knows. It is given by `free` (up to `PT_LOCAL`
distinct in-block bit positions, LSB first) and a fixed assignment of the other in-block bits (`mask`, `fixed`). The
verifier refuses a statement one of whose regions repeats a free bit: the pinned bound assumes them distinct
(`Statement.LinkLayout`), and a region built from two bit ranges that overlap would otherwise pass. For a link point
`r ∈ F^{m_pts}` the claim is:

~~~text
point p ∈ F^m:   p[j] = 1 if bit j of mask and of fixed are set, else 0   (j < k_log)
                 p[free[t]] = r[t]
                 p[k_log + i] = r[PT_LOCAL + i]                            (i < nbl)
value:           Σ_{b < 2^nbl} eq(r[PT_LOCAL ..], b) · Σ_i eq(r[.. |free|], i) · bit_i(bytes(region, b))
ExtraClaim:      (value, skip = eq(p[0..6], ·) over 64 entries, x = p[6 .. m])
~~~

`bytes(region, b)` is the region's known bit string in block `b` (bit `8i + t` is bit `t` of byte `i`), padding blocks
included. Claims are listed region-major, then point 0 before point 1.

### 16.3 `verity/flock-pure-block/v2` (table `pure`)

- Domain `"flock-pure-block-v2/fast100x2/rep" ‖ decimal(r)`; Σ tag `"verity/flock-pure-block/sigma/v2"`; `PT_LOCAL = 16`;
  `m_pts = 16 + nbl`.
- Layout from `(row_bytes, units, sha)`:

| layout | rows | k_log | compression slots | units | blocks per VU | slot log |
|---|---|---|---|---|---|---|
| `Chunk(n)`, 2 ≤ n ≤ 64 | `1024 n` bytes, BLAKE3 | 20 | 32 | 32 | n | 14 |
| `ChunkTail(n)` | `1024 n + 512` | 20 | 32 | 32 | n + 1 | 14 |
| `Fp8` | 1536 | 21 | 48 | 48 | 1 | 14 |
| `ShaBf16` | 3072, SHA-256 | 22 | 98 | 96 | 1 | 15 |
| `ShaFp8` | 1536, SHA-256 | 21 | 50 | 48 | 1 | 15 |
| `Fp4` | 864, BLAKE3 | 20 | 28 | 24 | 1 | 14 |
| `ShaFp4` | 864, SHA-256 | 21 | 28 | 24 | 1 | 15 |

  `unit_pos0 = slots · 2^{slot_log − 13}` (units are `2^13`-bit slots), `blocks_real = blocks_per_vu · vus`,
  `nbl = max(3, ⌈log2 blocks_real⌉)`, `m = k_log + nbl`. Admission (before any coin): CN3 (no `Chunk(1)`/`ChunkTail(0)`),
  CN2 (`m ≤ 35`), NV2 (a 608-input unit only with the fp4 layouts), NV3 (row schemas), NV5 (y leaf width), NV1
  (`out = y << 10` for `fp8-*` relations with `y < 2^22`, else `out = y`).
- Statement digest: `SHA-256(TAG ‖ D_b3 ‖ sha256(unit file) ‖ le64 of [vus, blocks_real, nbl, m, k_log, 13, slots, units,
  unit_pos0] ‖ layout text ‖ Δ_A ‖ Δ_B)`, `TAG = "verity/flock-pure-block/v2"`, `D_b3` the BLAKE3 compression circuit's
  digest; layout text is present for `ChunkTail(n)`, for `Chunk(n)` with `n ≠ 3`, for `Fp8`, for the fp4 layouts (row
  schema, then text) and for the SHA layouts (text, then the SHA-256 compression digest), each the exact sentence in the
  implementation; each Δ is `le64(|Δ|) ‖ (le32 row ‖ le32 col)*`.
- Σ: `SHA-256(SIGMA_TAG ‖ relation ‖ 00 ‖ le64 [vus, units, row_bytes, nbl, m] ‖ sha256(unit file) ‖ digest ‖
  compact-sorted-JSON(header.instances) ‖ root_a ‖ root_b ‖ root_y)`.
- Public input: the header, the row digests of x and W per VU, the y words; the verifier recomputes the three frame-v3
  roots (§16.6) from them and compares with the header. The rows are not public input: the current instance file carries
  them, and the vector tooling replaces them by their digests once.
- Commit publics, per real block: the chunk chaining values (x chunks then W chunks, 8 le32 words each), `y16` (le32),
  `acc_out` (le32), and for `ChunkTail` unit 15's `(y16, acc)`. `publicsCheck` (C4, legacy, §16.9): for `Chunk(n)` without
  the epilogue, the last block's `acc_out` of each VU equals its output word; for every layout with chaining values, each
  VU's chunk values fold through BLAKE3's keyed parent tree to its row digest.
- Regions: `Key, Counter, Cv, AccIn, AccOut, [Y], [ChunkTail: Flags, CvMid, YMid | AccMid]` for the chunked layouts;
  `CvSlot(k)…, AccOut` for `Fp8`; `Digest(0), Digest(1), Y | AccOut` for the SHA and fp4 layouts. Values come from the
  layout's constants, the Commit publics, the public output words and row digests, and fixed padding values.

### 16.4 IR frame statements (table `irf`)

- `verity/flock-ir-frame/v3`: domain `"flock-ir-frame-v3/fast100x2/rep" ‖ r`; Σ tag `"verity/flock-ir-frame/sigma/v3"`;
  Σ = `SHA-256(SIGMA_TAG ‖ ["class" ‖ class pin] ‖ unit ‖ 00 ‖ sha256(unit circuit) ‖ digest ‖ public_sha256)` with
  `public_sha256 = SHA-256(header line ‖ every byte after the rows)`. Statement digest: `SHA-256(TAG ‖ D_b3 ‖ sha256(unit
  circuit) ‖ key ‖ le64 [blocks_real, nbl, m, k_log, slots, nb, chunks, units, unit_log, unit_pos0, ret_col, ret_words,
  cut_words, |regions|] ‖ per input port (le64 words ‖ u8 hashed) ‖ Δ_A ‖ Δ_B)`. `k_log ∈ [20, 22]`, `m_pts = 16 + nbl`,
  `m = k_log + nbl` (the implementation has no `m ≤ 35` refusal; the spec rejects `m > 35`).
- Public input: the frame-v3 file without its rows section (digests, unit output words, instance outputs, cut words).
- Commit publics: each real block's run output chaining values. `publicsCheck` (C4, legacy): each row's last-run chaining
  values fold through the keyed parent tree (one chunk: its ROOT output) to the row digest.
- Regions: `Params, CvIn, Cv, [Out], [CutIn, CutOut]`.
- Admission (legacy, §16.9): `check_cut_words` evaluates the pinned tail on the file's cut words; `check_public_ports`
  hashes public ports.
- `verity/flock-ir-sampling/v1` shares table `irf` and the v2 frame lineage; its admission evaluates `DivFullRcp` and the
  Gumbel noise. `verity/flock-ir-block/v1` is an older statement with public unit IO. Both are legacy-only.

### 16.5 The circuit statement (PR #83, table `nl`)

- Tags are a parameter (`Flock.Tags`): every string below is hashed or matched byte for byte, so each version of PR #83's
  statement is one value. `PT_LOCAL = 24` and `m_pts = 24 + nbl` in both versions.

  | tag | `verity/flock-netlist/v1` (PR #83 at `9294e161`) | `verity/flock-circuit` (PR #83 at `19c7269a`, `fd02e847`) | `verity/flock-circuit` on SHA-512 (PR #83 at `631567f7`) |
  |---|---|---|---|
  | `TAG`, META `statement` | `verity/flock-netlist/v1` | `verity/flock-circuit` | as `fd02e847` |
  | file format (line 1, META `format`) | `flock-netlist/v1` | `flock-circuit` | as `fd02e847` |
  | block keyword | `NET` | `CIRCUIT` | as `fd02e847` |
  | public file format; circuit hash key | `flock-netlist-inst/v1`; `netlist_sha256` | `flock-circuit-inputs`; `circuit_sha256` | as `fd02e847` |
  | Σ tag | `verity/flock-netlist/sigma/v1` | `verity/flock-circuit/sigma` | as `fd02e847` |
  | domain | `flock-netlist-v1/fast100x2/rep` ‖ r | `flock-circuit/fast100x2/rep` ‖ r | as `fd02e847` |
  | table | `nl` | `circuit` | `circuit` |
  | Merkle leaf scheme (§14) | `flock-leaf/sha256-unsalted/v0` | `flock-leaf/sha256-unsalted` | `flock-leaf/sha512-unsalted` |
  | `Hello` coin scheme (seed mode) | `coin-commit/sha256/v1` | `coin-commit/sha256` | `coin-commit/sha512` |
  | `Hello` `rounds` | absent | absent | `"bytes retained"`: every round carries `msg` (S12) |
  | identity | `merkle_hash: sha256` | a `hashes` object naming every commitment hash (`row_leaf_in_circuit: blake3-keyed/row/v2`) | the `hashes` object with `merkle: "sha512 (leaves and nodes, every Ligerito level)"`, `coin_commitment: "sha512"` and the retained-bytes `round_digest` |

  The verifier names these `verity/flock-netlist/v1`, `verity/flock-circuit@19c7269a`, `verity/flock-circuit@fd02e847` and
  `verity/flock-circuit` (the SHA-512 statement). From `fd02e847`, META `leaf_scheme` is a pinned object: the unsalted id, and
  a `target` reserving the hiding leaf with its pinned key named by a digest (`hm96-sha256/v1` and `key_sha256` at
  `fd02e847`; `hm96-sha512/v1`, 192-byte salts and `key_sha512` = SHA-512 of `default_key` at `631567f7`, where
  `default_key = SHA-512("verity/hm96-sha512/key/v1\0" ‖ be32 i)` for `i < 4`, bit 2047 cleared). The verifier compares it in
  canonical form, and the identity's `leaf_scheme` is that object.

- The in-circuit row leaf is a parameter too (`Flock.RowLeaf`): its slot circuit, its layout check, its share of Δ, its
  regions with their known bytes, and the check of the input roots. Both versions use the keyed-BLAKE3 row leaf
  (`blake3-keyed/row/v2`) described below.
- Σ = `SHA-256(SIGMA_TAG ‖ sha256(circuit file) ‖ digest ‖ public_sha256)`, `public_sha256 = SHA-256(compact-sorted-JSON(header
  with rows := false) ‖ digests ‖ outputs)`.
- Statement digest: `SHA-256(TAG ‖ D_b3 ‖ sha256(circuit file) ‖ row key (32) ‖ compact-sorted-JSON(identity) ‖ le64 [n,
  blocks_real, nbl, m, k_log, pin, |regions|, G] ‖ Δ_A ‖ Δ_B)`; `identity` is the backend identity object (statement,
  backend `"C-interactive"`, proof class, profile, reps, commitments, flavour, Merkle hash, leaf scheme, ZK hooks).
- The circuit file: line 1 `flock-netlist/v1 <name>`, line 2 `META <json>`, then `NET <name> <lines>` blocks, each an
  expanded circuit in `flock-ir-unit/v2` form (the unit, tail stages). `sha256(file)` is the pin. The parser refuses a `CUT`
  line and the META keys `tail, cut, cut_words, public_ports, public, native, outputs_by_tail`: that is where the rule
  "the verifier evaluates nothing" is enforced.
  - Each expanded circuit's rows: the input rows `A = B = [i]` (an input column) or empty (input padding, a forced zero);
    the constant `A = B = [const]`, the last useful column. Every computed row, between them, reads in A and in B only
    earlier columns that are not input padding, or the constant (topological order: the soundness package's `Rows.topo`).
    So every column a row reads is below `useful`.
  - Each port group's ports fit its words, and each input group lies within the input rows. The input groups ascend and
    are disjoint, and every input port bit's row is `[i]` (only padding past a group's ports is empty): one copy each.
  - The input rows end before the constant (`128 · inWords ≤ constPos`, 128 rows per input word). Otherwise an input
    port could cover the constant's column, and Δ's input copies would rewrite the slot's constant row, leaving the
    unit's constant free.
  - Each output group ends inside the slot (`col + words ≤ 2^{unit_log} / 128`, D11). Otherwise a region at its column
    (`Circuit.outRegion`) would reach past the slot into the next range's columns.
  - Upstream's composite verifier checks none of these.
  - Lookup slots (META `lookups`, in key order after the expanded circuits, each named `lookup:<table>`) are built from
    SHA-256-pinned tables (`rsq, sqrt, rcp, ex2`) as two-level one-hot circuits.
    - Rows: the `n` index bits in one input word; the low and high decoders, each minterm the AND of two half minterms,
      recursively; one product row per output bit `j` and high minterm `h`, with B the XOR of the low minterms `l` whose
      `table[h·2^lo + l]` has bit `j`; padding to a word; the output word; the constant.
    - `lo` is META's `lo_bits`, and the table's size must be `2^n` words.
    - A fold may read the product rows' B side from the table, since it is about `bits · 2^{n−1}` entries.
    - The built rows are held to the same order. The product rows' B side, from the table, reads the low decoder's
      minterms, which precede them.
- Layout: whole VUs per block (`G` per block); slots: BLAKE3 compression runs per input port and chunk, one parent tree
  per multi-chunk port (heap order, root first), the unit range, tail stages, lookup slots, one mask slot, forced-zero
  padding. Δ: const wiring to the pin; A-only chaining-value copies within runs and from children into parents; "both"
  copies of unit leaf bits from message bits; "both" copies for every META wire. No key, counter or flag is wired: they are
  region claims.
- Public input: the public file (no rows; `serve` refuses a file with rows): header, row digests per instance and port,
  output words. The verifier recomputes the input roots over the digests and the output roots over the output words
  (u16 big-endian leaves), and checks the pinned padding digest `keyed_hash(key, zero row)` and padding chaining value.
- Commit publics: empty. Regions: `Params`, `CvIn` and (single-chunk ports) `Digest` per run range; `Params`, `CvIn`,
  `Digest` at the parent root; `Out` on the unit range. Values: the layout's BLAKE3 parameters, the key, the public
  digests and outputs, and META's pinned padding values.
- Coins: `seed` mode carries `"coins":"coin-commit/sha256/v1"` in `Hello`.
- Shape: `blocks_real = ⌈n / G⌉`, `nbl = max(3, ⌈log2 blocks_real⌉)`, `k_log ≤ 27`, `m = k_log + nbl`; the spec rejects
  `m > 35` (no Fast100 schedule exists above it).
- **The proved range** (`checkInRange`, in `setup` and `setupH` for every statement). The statement must lie in the
  soundness package's `Stmt.InRange`: `k_log ≤ 27`, at most 1024 regions, and `m_pts = 24 + nbl ≤ 64`. So the verifier,
  and not only the writer, accepts only statements the proved bound covers. The pinned statement is
  `flock_verify_sound_205` (`Security/Proofs/Flock/Soundness/Discharge/FailClosed/Numeric.lean`): every statement
  accepted without `--zk` carries the 2^-205 per-table bound, with no range hypothesis, since acceptance gives the range.
  `checkInRange_ok` is the lemma that says the check accepts exactly that range.

### 16.6 frame-v3 roots

~~~text
H(tag, parts) = SHA-256("veritor/protocol/merkle/frame/v3\0" ‖ be32(|tag|) ‖ tag ‖ Π_p (be64(|p|) ‖ p))
uint(x)       = big-endian bytes of x without leading zeros (0 → 00)
leaf(r)       = H("leaf", [dom, uint(r), uint(r), schema, payload])
pad(r)        = H("pad",  [dom, uint(r)])              for r in [n, next_pow2(n))
node(d, i)    = H("node", [dom, uint(d), uint(i), L, R])   d = 0 at the first merge, i = index in the new level
~~~

A single leaf is its own root. Row-digest leaves carry the 32-byte digest; output leaves carry the word big-endian in its
schema's width. Row digests themselves are `blake3::keyed_hash(row_key(role), row)` with `row_key(role) =
"verity/blake3-leaf/v1/" ‖ role`, zero-padded to 32 bytes, or `SHA-256(prefix_block(role) ‖ row)` for the SHA schemas.
Recomputing a root over public digests is a check of the commitment scheme on public data, not an evaluation of the
computation.

### 16.7 The partition invariant

The verifier checks the partition invariant itself, on its own copy of each circuit's cut (`Flock.Partition`,
`flock-verify unit-cut`). A cut is `total_gates`, `structure_gates` (constants and wiring: circuit structure, never
committed), `owner[g]` (`"input"` for an input gate, a computing unit `u ≥ 0` otherwise), the `reads` `(producer,
consumer)`, the returned `outputs`, the `committed` gates, the `input_gates`, and the `recomputed` pairs `(gate, earlier
gate)`. The failure codes, in order, are those of `verity.ml.circuits.partition.validate_unit_cut`:

| code | rule |
|---|---|
| `input-outside-input-unit` | every input gate is in the input unit |
| `gate-not-certified-once` | every computed gate is computed in exactly one computing unit |
| `gate-recomputed` | a recomputed value stays in the unit that computes it |
| `gate-count` | certified gates + input gates + structure = `total_gates` |
| `read-uncommitted` | a cross-unit read goes only through a committed value |
| `output-uncommitted` | a returned value is committed |
| `committed-unread` | a committed value is read across units or returned |

With no failure, each gate is certified by exactly one unit and the committed set is the boundary set. A read outside the
circuit is malformed. Under `Q_word` v2 (a cut's `"recompute": "report"`, the reference's `recompute="report"`) a recomputed
pair across units is no failure: it is counted (`recomputed_across`), and every other code is v1's.

**One Call's cut against its graph** (`Partition.checkCut`, `flock-verify cut-check`; reference `verity.ml.circuits.cut.check_cut`). The invariant runs on the Call's computing gates, with its parameters as the input unit and, unless served values
are given, the committed set the cut implies: a value is committed exactly when another unit reads it or the Call returns
it. Then the width rule `Q_word_v1{X, W, EXTRAS}`, today `(16, 32, 0)`: each unit's boundary (its gates another unit reads
or the Call returns) is at most `X + EXTRAS` bits, or one value of at most `W + EXTRAS` bits; else `unit-too-wide`, with
`(unit, bits, values)`.

**`Q_word` v1 over the verifier's own copy of the program** (`Flock.Qword`, `Flock.Program`, `Flock.Extract`;
`flock-verify evaluate`, `qword-program`; reference `verity.ml.circuits.cut` and `verity.ml.circuits.partition_object`):
- the program is decoded from its `verity-ir/descriptor/v1` bytes by #120's rules, and each Call's graph is built as core's
  `CallGraph` builds it;
- a program that is not topological, or whose root batches or scans, is inapplicable to every query;
- the cut is core's: a flat cut packs the returned gates into runs of at most `X` bits and promotes committed units until no
  edge crosses units that head nothing, and a separable body is its members' cuts;
- `Extract.checkCalls` checks each Definition's cut once, on its Call's whole graph with the derived committed set, for the
  partition invariant and the width rule `(X, W, 0)`, as `partition_object.verify` does.
- `Q_word` v2 (`qword-program --query-version 2`) has v1's units; its cut check reports a value computed in two units (each
  Definition's `recomputed_across`) instead of refusing it with `gate-recomputed`.

The algorithm step by step is in `note:20260929T1052Z-report-flock-verifier-qword-cut-check`.

An input node is `Input<w>_v1` (bound per run) or `RegisteredInput<w>_v1` (bound at registration; core's PROTOCOL.md §5):
`Program.inputId?` records the binding time; both are inputs to every query, never Calls.

**`Q_call` v1** (`Flock.Qcall`, `flock-verify qcall-program`; core's PROTOCOL.md §11) is core's evaluation, refusals and cut
check, as `Flock.Qcall` documents them.

### 16.8 Objects by content

The verifier loads each statement's objects from its own archive, keyed by content, and never from the prover's session
directory: `verify --archive DIR --circuit H --public H --comp H` reads `DIR/sha512/<H>` for each (`H` a SHA-512), and
rejects an object that is not held or does not hash to its key. The lookup tables are those `MUFU_TABLES` pins, held
under their SHA-512; `Circuit.parse` checks each table a lookup uses against that SHA-512, and the META's `sha256` against
the SHA-256 pin. A digest is only a lookup key into what the verifier holds: the statement it verifies is
the one those objects define, and the record must bind that statement (S1). `archive-put DIR FILE...` stores files under
their SHA-512. A session directory supplies only the record and the openings. Without `--archive`, the flags are paths
(the development and agreement tools).

### 16.9 Strict and legacy conformance

Daniel's rules: the verifier checks proofs and never evaluates any part of the computation; circuits are total; the
circuit is public and every value is private. The verifier of record implements them as two modules:

- **Strict** (the default): accepts only statements whose checks are the Flock proof, the session, the frame-v3 roots over
  public digests and outputs, and hashes of pinned padding constants. Today that is the circuit statement (§16.5).
- **Legacy** (to replay recorded cells for agreement with upstream, labelled as such): adds exactly these native checks,
  each named in the verdict:
  - L-CV: BLAKE3 keyed parent trees over committed chunk chaining values (pure-block, IR frame C4);
  - L-ACC: the final accumulator equals the output word (pure-block `Chunk(n)`);
  - L-TAIL: the pinned tail evaluated on cut words (IR frame cut templates, IR block);
  - L-SAMPLE: `DivFullRcp` and the Gumbel noise (IR sampling).

  Legacy statements also make intermediate values public (chunk chaining values, accumulators, cut words), which the rule
  "every value is private" forbids; the legacy module accepts that as recorded.

A legacy verdict is never a strict verdict. The circuits' totality is a property of the pinned files, checked by level 3's
lowering theorem, not at run time.

### 16.10 The hm96-sha512 row statement (PR #83 from `eb90718f`; with the e2e bindings from `e51e2b86`, `verity/flock-circuit@e51e2b86`)

M0's format changes three things from §16.5: serving rows are
`hm96-sha512/row/v1` leaves proved in the circuit (no BLAKE3, no `comp.rows`), every identity and binding digest is
SHA-512, and the unit draw is live (§7.3). `Flock.HmRow` is its statement; the lincheck fold, PIOP and PCS are §10–§14.

- **Tags.** META `scheme` is `frame-v3-sha512/hm96-sha512` and `row_schema` is `hm96-sha512/row/v1`. The public file's
  circuit key is `circuit_sha512`. `Hello` adds `"digests":"sha512"`. The identity carries META `leaf_scheme`
  (`hm96-sha512/v1`, `key_sha512`, and `salt_source`: the OS's own bytes, every leaf's 192 read with its tree's; the
  statements pinned `@b3727bd9` and earlier name a ChaCha20 expansion of a per-proof OS key) and `seed_injection` (true
  only in the selftest harness's build: the statement's name with `+seed-injection`).
- **Rows.** For a port of `words` u16 words (`2·words` a multiple of 128): `x = SHA-512(prefix ‖ row)` with `prefix =
  pad128("verity/sha512-row/v1\0" ‖ 01 ‖ 10 ‖ be32 words)`, `b = x ⊕ M(key)·y`, `c = SHA-512(salt_prefix ‖ y)` for the
  row's 192-byte salt `y` under `hm96-sha512/v1`'s default key. The verifier recomputes each row's `prefix`, the midstate
  after it (one SHA-512 compression from the IV), and the padding block `0x80 ‖ 0 ‖ be128(8·(128 + 2·words))`, and refuses
  a META `rows` entry that differs, and a `row_key_sha512` other than SHA-512 of the default key.
- **Bit rows (`hm96-sha512/row/v2`).** META may carry `row_schemas`, one `["hm96-sha512/row/v2", ℓ]` per port, for
  `row_schema` (refused: both, neither, or typed); the identity's `row_leaf_in_circuit` then names `hm96-sha512/row/v2`.
  Port `p` is core's bit row of `ℓ < 2^64` bits (`sha512/row/v2`): `⌈ℓ/16⌉` u16 leaves, bits low first, `x = SHA-512(P(ℓ)
  ‖ bytes)` over `B = ⌈ℓ/8⌉` bytes. The verifier derives from `ℓ` the words, `⌈(B + 17)/128⌉ − 1 ≤ 2^16` compressions,
  `P(ℓ)`'s midstate and the padding `0x80 ‖ 0* ‖ be128(8·(128 + B))`, refusing a differing META `ports` or `rows` entry and
  a statement with no compression. Message bit `b` of byte `j` is a row bit if `8j + b < ℓ`, else a zero fill bit (`j <
  B`) or padding; so the hm96 `pad` may hold row bytes, and a row of ≤ 111 bytes has no compression.
- **Slots.** `sha512x3` (inputs `(cv 512, msg 1024) × 3`, outputs `out 512 × 3`) and `hm96` (inputs `cv 512, pad 1024, y
  1536`, output `bc 1024`) are `CIRCUIT` sections. Per VU, port `p`'s compressions are numbered on from the previous
  ports', three per slot (slot `k / 3`, sub `k % 3`), `per_vu[sha512x3] = ⌈total / 3⌉`; `per_vu[hm96] = next_pow2(ports)`,
  slot `p` for port `p`. The `sha512x3` range is `packed` (aligned to one slot, spanning its count); every other range is
  aligned to its power-of-two span. Every range's slots are at least a word, as on the BLAKE3 path, and at most the block
  (`7 ≤ slot_log ≤ k_log`): a slot below 64 bits makes the fold no `αA₀ᵀe + B₀ᵀe` of any matrices, and a packed range of
  no slots would escape the block bound.
- **The nets the verifier builds.** The `sha512x3` and `hm96` nets must equal the verifier's own, in every field
  (`Flock.HmNets`, D7). So must a unit net whose name starts with `rec-open/` (V*'s opening, `rec_outer.lowering`):
  - its name must be `rec-open/l<LANES>-h<H>`, with `LANES` a positive multiple of 8 and `0 ≤ H ≤ 32` (the side bits
    are `dirs`' words 0 and 1, and at `H = 0` there is no climb);
  - its input ports must total `384·LANES + 1024·max(1, ⌈H/2⌉) + 2048` bits and its output ports 2048, compared before
    the verifier builds its net (about 4,374 rows a lane), so a name can't make it build a net larger than the file's
    rows allow;
  - the statement's first five rows must be `row, aux, acc, coef, dirs` at the input widths below (`out` and `acc_out`
    follow), and every copy's input word `i` must read leaf `i` (`leaves_in[u] = [0, 1, …]`), so each copy reads its
    opening's rows in that order; both are compared before the net is built (`RecOpen.rowsOk`, `RecOpen.leavesOk`);
  - the net must be `Flock.RecOpen`'s builder of `rec_open._steps` at those parameters, laid out as `ir_lower.layout`
    lays it out: inputs `row, aux, acc, coef, dirs` of `8·LANES`, `64·max(1, ⌈H/2⌉)` (at `H = 0` one row the net does
    not read), `64`, `16·LANES` and `64` u16 words, and outputs `out` then `acc_out` of 64 words each, one 16-bit port
    per word.

  So must a unit net whose name starts with `rec-open-stacked/` (one of R's levels, V*'s `recOpenStackedName`): its name
  must be `rec-open-stacked/l<LANES>-h<H>`, and the same three rules hold at those parameters, the net being the same
  builder's under the unit's own name.

  So must a unit net whose name starts with `rec-residuals/` (a part of V*'s algebra, `rec_residuals.lowering`):
  - its name must be `rec-residuals/m<m>-k<k_log>-c<claims>-p<j>`, with a `fast100` schedule at `m`, `k_log ≥ 6` and
    `claims ≥ 2`, and neither `k_log` nor `claims` above the unit's input bits, compared before the verifier computes
    the structure;
  - META `holo` (`RecResiduals.HOLO_KEY`: `true`, or absent for false; any other value refuses) says whether the parts
    are a private circuit's (`rec_algebra.Shape.holo`: each comb operand also reads its `h`). The shape is at that
    `holo`, so exactly one net is accepted: a holo statement on plain parts refuses, and so does a plain statement on
    holo parts. With `--rec-inner`, V*'s statements are built at the verifier's own `--hide` (`holo` in
    `verity/rec-inner/v1`, `RecVStar.Inner.holo`), so a file's META `holo` must equal it, and every `VERDICT` line
    carries it as `private_circuit`. The verifier refuses an inner shape with `holo` true until V*'s circuit writer
    writes holo parts;
  - the verifier computes `GfResiduals`' S at that shape (`Flock.RecAlgebra.structureOf`, `rec_algebra.structure`) and
    its row ports (`rec_residuals.spec`: one per round that absorbs elements, then `acc0` and `acc1` of 2), groups its
    residuals as `rec_residuals.parts` does, each residual in the first part of at most 64 regions, 160 `sha512x3`
    slots and 16 residuals it fits, and refuses unless the parts hold every residual once; part `j` must exist;
  - part `j`'s S is the whole S's residuals and the operands their products read, each term renumbered into the part's
    `w` (its row ports' elements, in order) and `v` (the elements it reads, ascending); a term outside them, or a
    transposed block not in one run of the part's `w`, is refused, never read as element 0;
  - its input ports must total the part's (`16·P(n)` bits a row port of `n` elements and `v`, `P(n) = 64·⌈8n/64⌉`) and
    its output ports 2048, compared before the verifier builds its net;
  - the net must be `Flock.RecResiduals`' builder of `rec_residuals.unit_circuit` on part `j`'s S, laid out as
    `ir_lower.layout` lays it out, one 16-bit port per word.

  So must a unit net whose name starts with `rec-opening/` (a part of R's opening, `rec_residuals.stage_opening`):
  - its name must be `rec-opening/m<m>-c<claims>-p<j>`, with a `fast100` schedule at `m` and `claims ≥ 1`;
  - the circuit must have one unit per VU (`units_per_vu = 1`), so each part is one copy of its unit, refused before
    the structure is computed;
  - the rest is a `rec-residuals/` part's (the claims bounded by the input bits, part `j`'s bits, the net), at the
    opening's S (`Flock.RecOpening.structureOf`, `rec_algebra.structure` at an opening-only shape) and its row ports.

  Any other unit is not checked against a builder (D13, §17.1).
- **Units.** `units_per_vu ≥ 1` (D8, §17.1). The unit range's `vus_per_block × units_per_vu` slots are a power of two only
  when the unit returns the outputs, so a tail that returns them would otherwise pass with no unit in a VU.
- **Blocks.** `vus_per_block ≥ 1` (D9, §17.1), and a power of two whatever returns the outputs. Every range's count is
  `vus_per_block` times its per-VU count, and the unit range's power-of-two bound binds only when the unit returns the
  outputs, so a tail that returns them would otherwise pass with no VU in a block, or with `G = 3`: the `Digest` regions'
  `log2 G` VU bits (Lean's `Nat.log2`, Rust's `trailing_zeros`, which agree only at a power of two) would pin VUs 0 and 1
  in one verifier, VU 0 in the other, and VU 2 in neither.
- **Header sizes.** A power of a number META chooses is computed only once the number is bounded. `parse` refuses
  `k_log > 27` where it reads it (`HmRow.kLogOf`), before the layout check computes `2^{k_log}`, or `2^{slot_log}` once
  `slot_log ≤ k_log`; and it compares a lookup's `in_bits` with its pinned table's `log2` before `2^{in_bits}`. `setupH`
  refuses `k_log > 27` too, and a pinned table's words are a power of two, so the verifier accepts and refuses what it did
  before (`kLogOf_ok`, `lookup_index_test`), refusing such a file before spending memory on it.
- **Wires.** META `wires` begins with the row wires, which the verifier derives and compares: compression `k`'s `cv` from
  compression `k − 1`'s `out`, and hm96 slot `p`'s `cv` from the row's last compression. No other wire touches a row slot.
- **The pin** is the constant column of the first slot of the first range that is not the mask (`HmRow.pin`). That range
  must have a slot: with none, the pin's row would be empty, or, in a packed range, could fall inside another range's rows.
- **Δ,** in this order: every slot's constant from the pin; per VU and port, the first compression's `cv` bits (the
  midstate) and the hm96 slot's `pad` bits (the padding block), each bit from the pin (1) or the unit range's first slot's
  forced-zero row (0); every unit's leaf bit `t` of the u16 word at row byte `B` from the message bit of byte `B + t/8`,
  bit `t % 8` (compression `B / 128`, word `o / 8`, bit `56 − 8·(o % 8) + t % 8` for `o = B % 128`; a zero leaf from its own
  forced-zero row); every wire. For a bit row: the first `cv` (with no compression, the unwired hm96
  slot's) from the midstate; every constant message bit as `pad`'s are; a leaf or cut bit at row bit `16·leaf + t ≥ ℓ`
  from the forced-zero row (no leaf reads a fill bit), one in the last block from the hm96 `pad`.
- **Regions.** `Digest(p)` per port: the hm96 range's `bc` group, free bits `[0, 10)` and the `log2 G` VU bits above slot
  `p`, valued at each VU's `b ‖ c` (a padding VU's the pinned `dummy.bc`); then `Out` (§16.5).
- **The region-word check** (`HmRow.checkRegionWords`).
  - A region claim's ring-switch message carries every bit of each word the region covers in the clear.
  - So `setupH` refuses a statement in which a column shares a 128-bit word with a *partial* region's columns without
    being one of them, unless its rows in the combined `A` and `B` are both empty.
  - A region is partial when some in-word bit 0–6 is not free. Its covered words are its fixed bits above 6, with each
    combination of its free bits above 6.
  - Row `q` of the combined `A` is the slot type's row placed at `q`, in block columns (a mask slot's is `[q]`), plus every
    `Δ_A` entry `(q, col)`, summed over GF(2) as `addDelta` sums them. An entry twice cancels. `B` is the same.
  - Today only a tail stage's `Out` region is partial. On the GEMM coordinate it is 16 bits of a word, and the other 112
    columns are checked.
- **Public file.** `flock-circuit-inputs`: header `units = {vus_per_block, units_per_vu, indices (ascending, one per
  instance), blocks (indices in chunks of G)}`, `frame_v3.frame = "frame-v3-sha512"`, `frame_v3.row_key_sha512`; body
  `b ‖ c` (128 bytes per instance and port, instance-major), then the output words. The public digest is
  `SHA-512(canon(header, rows = false) ‖ body)`. The frame-v3-sha512 roots (§16.6 with SHA-512) are over the population:
  row leaves `leaf(dom, i, "hm96-sha512/row/v1", SHA-512(leaf_prefix ‖ b ‖ c))` (a bit row's under
  `hm96-sha512/row/v2`), output leaves `u16` words. Each port's `dummy.bc` must be the zero row's: `SHA-512(prefix ‖
  0^(2·words)) ‖ SHA-512(salt_prefix ‖ 0^192)` (a bit row's with `P(ℓ)` and `0^B`).
- **The drawn statement.** For a record with a `unit_draw` over the population's `N` instances, instance `i` is population
  position `units[i]`; the header is the population's with `unit_draw`, `instances = K`, `population = N` and the drawn
  units' indices in `units`; the public digest covers that header and the population's body, so the roots stay the
  population's. A null `unit_draw` is no draw. U1–U3 then hold of the record and this statement.
- **Shared-row files (statement `verity/flock-circuit@967b8d06`).** The header may carry
  `shared_rows = {port: R_p}`, naming exactly the circuit's input ports, each `R_p ≥ 1` (a value that is not a non-negative
  integer counts as 0). Input port `p`'s rows are then a table of `R_p` rows, committed once:
  - the body is every table row's `b ‖ c` (port-major, `Σ R_p` of them), then every instance's refs (u32 LE, instance-major,
    ports in order, each below its table's `R_p`), then the output words; the public digest covers all of it, refs included;
  - port `p`'s frame-v3 tree has `R_p` row leaves, leaf `r` over table row `r`'s `b ‖ c`;
  - instance `i`'s `Digest(p)` value is the `b ‖ c` of table `p`'s row `refs[i · ports + p]`, so the row the circuit hashes
    for that instance is the committed row its ref names;
  - a drawn statement keeps the population's tables, its instances carry their refs, and its public digest covers the
    population's tables, every refs and every output word.
  A file without `shared_rows` is read exactly as before, and the earlier statement names refuse a file that has it (their
  upstream builds reject its length). `@967b8d06` is also the identity text of `68ae79f2`: `round_digest`, `statement_digest`
  and `sigma` read `sha512`.
- **Digests.** The circuit pin is `SHA-512(circuit file)`. The statement digest is `SHA-512(TAG ‖ circuit SHA-512 ‖
  canon(identity) ‖ le64[n, blocks_real, nbl, m, k_log, pin, |regions|, G] ‖ le64|Δ_A| ‖ Δ_A (le32, le32) ‖ le64|Δ_B| ‖
  Δ_B)`; Flock's 32-byte statement slot (round 0's R1CS digest) is its first 32 bytes. `Σ = SHA-512("verity/flock-circuit/sigma"
  ‖ circuit SHA-512 ‖ statement digest ‖ public digest)`, 64 bytes, is `Hello`'s `link.sigma` and root_F.
- **Record.** Every digest is SHA-512 of the same bytes as before under a `_sha512` key: `link.root_f_sha512`,
  `link.root_b_sha512.<t>`, `link.publics_sha512.<t>`, `link_sha512`, `streams[i].root_sha512`, `rounds[k].msg_sha512`,
  `proof_sha512[j] = {stream, sha512, bytes}`. The coin commitment is `coin-commit/sha512`; the derivation stays SHA-256.
- **The e2e bindings (from `e51e2b86`, and every later statement; `@eb90718f` without them).** META carries
  `program_sha512` (a SHA-512 hex digest) and `partition = {rule, digest}`, the rule `verity/partition/v1` or the
  provisional `flock-circuit/unit-cover/v0` and the digest SHA-512 hex. Both are part of the circuit file, so the circuit
  pin binds them into the statement digest and Σ. The public header's `units` adds `classes`, one per instance: each must be
  the verifier's own class of the circuit, `SHA-512(head ‖ "\nMETA " ‖ canon(META without partition, program_sha512,
  program_digests) ‖ "\n" ‖ the file after its META line)`. For a `verity/partition/v1` binding the verifier needs its own copy
  of the partition object (`--partition`, by path or archive digest). Its form is #111's canonical one with open query names:
  exactly `format`, `program`, `query`; a SHA-512 program; a query of exactly `name` (a non-empty string), `version` (a
  natural number) and `params` (an object), and for a query the verifier evaluates, its parameters too (`Q_word` v1 and v2:
  exactly `X`, `W`, positive integers; `Q_call` v1: exactly `X`, an integer in 1..2^16). Its tagged digest `SHA-512("verity/partition/v1\0" ‖ canon(object))` must be META's, and its
  `program` must be META's `program_sha512`. The provisional unit cover takes no object.
- **Tail outputs, packed ranges and leaf cuts (GEMM).** META may name `out_net`, the circuit whose
  output group holds the returned outputs (default `unit`), with `out_net_ports`, and `leaf_cuts`, `[unit, u, port, leaf]`.
  - When a tail stage returns the outputs: the unit range need not be a power of two and the units return none; that stage's
    only output group is one 16-bit port per output word (`16 · words` a power of two); and the `Out` region is that group,
    free bits `[0, log2(16 · words))` and the stage range's slot bits, valued at each VU's output words (u16 LE).
  - Any range may be packed except those a region reads (`hm96`, `out_net`, the mask).
  - A leaf cut wires a unit's cut input from a row word: bit `t < 16` copies the word's message bit (as a unit leaf does),
    the higher bits the forced-zero row. Each wires its port once, and Δ lists them after the unit leaf wiring, before the
    wires.
- **The template queries** (`Flock.TemplateQuery`; core's `TemplatePartition`): `Q_template_instance` v0 (`{template}`, a
  string) and `Q_template_instances` v0 (`{templates}`, a non-empty list of distinct strings). With the verifier's own copy of
  the program (`--program`, its `verity-ir/descriptor/v1` bytes, by path or archive digest), whose SHA-512 over the canonical
  descriptor without `annotations` must be the object's `program`, the verifier decodes the program's callees, definitions
  and root nodes, and applies the query. Every non-Input root node must be a `call` or `batch` of a listed template, matched
  by the descriptor id in the program's bytes. Each call is one unit and each batch member one unit, in node then member
  order, and a unit's gates are its instance's interval. Every listed template has an instance, or the query refuses. The
  circuit's template (the one key of META `program_digests`) must be listed, and every unit the header's `units.indices`
  claims must be an instance of it: then the statement's units are derived. A unit class's circuit (META `template`
  `unit-class`, its one key `unit-class/…`) names the Definition its units are instances of in META `definition`, and that
  descriptor id is then the template the query must list and the claimed units must be instances of. That the class circuit
  lowers the Definition is the class's own claim, which the verifier does not check (the note says so); the class digest
  covers META `definition`.
- **The partition finding.** Every verdict of such a statement carries `partition = {rule, query, digest, units, note}`.
  `units` is `derived` when the verifier derived the statement's units itself, given `--program`: for the template
  queries (above), and for `Q_word` v1 and v2 (§16.7) when the program is the object's, the query applies, every claimed unit is
  one whole activation of the circuit's template (the innermost Definition on its `locate` path, whose cut is one unit),
  and every Call's cut holds the invariant and the width rule. Likewise for `Q_call` v1 (§16.7): every claimed unit is one
  whole activation of the template (the Definition its path ends at, and not a batch's or scan's members taken together),
  and every Call's cut holds the invariant and `X`, a value computed in two units counted in `note`, not refused. A
  failing cut refuses the partition, naming the Call and its codes. For any other query, or without `--program`, `units`
  is `as stated`, with the reason in `note`, which an audit reports as a gap.
- **Registered reads**. A shared-row header may name `registered`, `{port: {value, positions, paths}}`: registered rows.
  The verifier needs its own `--registered F` (`verity/registered-reads/v0`: per port the value, root, leaves, domain,
  read positions, `words`, `word_bits` 16 or 32, absent for the port's own 16-bit rows); the port's rows must be
  `sha512/row/v1` rows of their u16 words, padded. A port is any row of the circuit: an input port's, or (§16.13) an
  output port's row, whose table is the instances' output rows, instance `i`'s ref `i`, so instance `i`'s output is the
  registered row at `positions[i]`. Each table row's `hm96-sha512/row/v1` leaf must climb its path to the root at its
  position, each instance's ref name its unit's read position in the verifier's program (units `derived`). A port on
  one side only refuses; verdicts report `registered`. A recursive statement reads the inner session's round commitments
  as registered values, round `i`'s as `rec-c<i>` and level `l`'s committed tops as `rec-top-L<l>`; a drawn statement
  must draw in full every unit the verifier's reads of such a value name (U4, `Registered.checkFull`, D14), so that no
  stage's message is read off a sample. The registered opening (`RecOpening`) reads its rounds as registered values too:
  `rec-opening-c<i>` is round `i` of `RecOpening.rounds` (each claim's ring switch, then Ligerito's rounds in
  `ligRounds` order), and its ports are the `c<i>`, then `acc0`, `acc1`, then the claims. U4 names these as well
  (`Registered.roundCommitment`), with the stacked table R's level-`l` tops `rec-top-stacked-L<l>` and the sparse phase's
  messages `rec-sparse-<p>`, `p` one of `pre<j>`, `v2`, `sc<j>`, `u` and `vt` (`rec_sparse.message_ports` without `h`,
  which reads the rep's `rec-c<i>`, and `claim`, which reads `rec-claim`). The sparse phase's `rec-sparse-v`,
  `rec-sparse-zero` and `rec-sparse-tau` are no round's commitment.
- **Identifier.** The header may carry `identifier`, a JSON object C-Flock does not read (an application's name for what the
  statement is about, such as PoUW's). The public digest covers the header, so Σ binds it. The verifier needs its own
  `--identifier F` (a JSON object, by path or archive digest), and the header's must equal `F`'s in RFC 8785 form: both are
  `canon`'s on core's RFC 8785 domain (`Program.domainError`, `verity.ml.circuits.codec.jcs_domain_error`: strings and keys ASCII
  without DEL, integers of magnitude at most 2^53 − 1), and anything outside it refuses. A header identifier that is not an object, one with no
  `F`, and an `F` with a header that names none refuse. Verdicts and `statement` report `identifier`. The prover writes it
  with `circuit.write(..., identifier=)`.
- **Public input ports** (`Flock.HmIn`, §16.13's statements). META may name `public_inputs` (`[{port, words, seed,
  derivation}]`) and `leaves_public` (per unit, each leaf's public word or −1): unit inputs the verifier recomputes from
  public data, not rows. The words of an unpacked unit range's leaf group holding them hold no row leaf; each run of
  them is cut into the largest aligned blocks of `2^e` words, and Δ copies none of their bits. Regions `in r` follow the
  digests, block by block: `2^s` words each, `s = min(e, PT_LOCAL − 7 − ⌈log2 count⌉)`, the slot bits free, valued at
  each instance's public words. The header repeats `public_inputs`; the body ends with every instance's public words
  (u16 LE). The verifier needs its own `--public-inputs F` (`verity/flock-public-inputs/v1`: `{format, instances,
  ports}`, then the population's words), refusing a differing word, a missing file and a port named on one side only.
- **Proof.** The Merkle trees are `hm96-sha512/v1` (`PcsParams.merkle_hash = 3`, 64-byte digests): every level's openings
  carry `opened_salts` (192 bytes per opened row, after the siblings), and an opened row's leaf is
  `SHA-512(leaf_prefix ‖ (x ⊕ M(key)·y) ‖ SHA-512(salt_prefix ‖ y))` for `x = SHA-512(row)`.

### 16.11 The typed statement (`verity/flock-circuit/types`, item 1e)

`verity_flock.typed_statement` writes §16.10's statement with its unit as circuit types and layouts instead
of rows. The first line is `flock-circuit/types NAME`, and META `format` and `statement` name it: `Tags.circuitTypes210d32e1`
(`verity/flock-circuit/types@210d32e1`), and `+seed-injection` for the selftest build; §16.13's `verity/flock-circuit/types`
hides its outputs. Its identity is `967b8d06`'s text, naming the typed statement. The statement
digest's leading tag is the statement id, `verity/flock-circuit/types`, as the prover hashes it; its Σ tag is
`verity/flock-circuit/types/sigma` and its domain `flock-circuit-types/fast100x2/rep` ‖ r (a new id gets new tags).

- **No statement carries rows.** A `CIRCUIT unit` or `CIRCUIT root` section is refused. `Typed.read` checks META and
  derives every row once, through `DeriveAll.deriveChecked` only:
  - `types`: `{digest: content}`. Each key must be its content's digest, and each type passes `CircuitType.check`.
  - `layouts`: the layout records. Each record's `sha512` must be its digest, and the list must pass `Layout.check`, with no
    tables held.
  - `unit_layout`: the unit's layout, which must be derived (`own` is `derive`).
  - `unit_type` must be that layout's type, `policy` must be `calls/v1`, and `unit_in` must be empty: this version binds
    no inputs to constants.
- **A flat class** (a unit with no placed parts, reads or Δ beyond its constant's copy of `one`) is §16.10's statement.
  - `HmRow.parseTyped` inserts the unit's rows as the file's `CIRCUIT unit` section, first, where the old statement has
    it, and parses the result with `HmRow.parse`.
  - The circuit digest and the class are the typed file's own: its SHA-512, and `classOf` over its bytes.
  - The unit's computed columns (its order after its inputs) must lie past its input rows, and its input columns must be
    its net's input port bits, in order (`Typed.flatNet`).
- **A template** (GEMM) is parsed with its block (`HmRow.parse`'s `tmpl`).
  - **Circuits:** `root`, the unit's own region with its entries at part columns left out, one slot per VU. Then each
    placed layout's rows, named by its digest, one slot of its range per part. A part's circuit is its own layout's, by
    position among the placed layouts (`HmRow.blockOf`), never by digest: two layouts with one digest stay apart. A part
    naming a layout the file lacks is refused.
  - **Derived, never stated:** `units_per_vu` (1), the leaf maps (none), the output circuit (the root) and the wires (the
    rows' chains). META stating any of them, or `lookups`, `leaf_cuts` or `unit_log`, is refused.
  - **Checks** (`check_typed`'s): the root's ports are the rows' words and the output words, 16 bits each, in one
    output group. Each placed layout has one slot per part, of its range's size. The root's slot has a forced-zero row.
    The rows' circuits (`sha512x3`, `hm96`, found by name) come after the root and the placed layouts, so no part is
    one of them.
  - **Δ:** first the flat statement's entries. Then, per VU:
    1. each input bit `w` a copy of its message bit (bit `w % 16` of row word `w / 16`);
    2. each part's bound input rows, `(i, i)` and each source, in item order;
    3. the root rows' cross entries: A's in `Δ_A` and B's in `Δ_B`.
  - **Regions:** the digests, then `Out` over the root's output group.
- **Reads** (attention): a placed read is a part whose layout `table/v2` generates (`own: {gen: "table/v2", …}`), placed
  and bound as a call is.
  - **Tables:** the statement names each table by the SHA-512 of its raw `u32` LE words and never carries them. The
    verifier holds its own copy (`Typed.hold`): a library table (`LIBRARY_TABLES`, by that SHA-512), read from the
    verifier's file under the library's name (`--tables DIR/<name>.u32`), of `2^index_bits` words whose SHA-512 is the
    name, checked before any use. Anything else is refused.
  - **A read's slot:** its generated rows. Its product rows' B side is read from the held table in the fold
    (`Lookup.foldB`), at its `READ` record's `prod0`, `lo_top` and `k` (not the table's value bits).
  - A read inside the root's own region, or inside a placed call, is refused.
  - **Rows only through `deriveChecked`,** whose check covers reads: a placed read is a segment of a generated
    layout for its table, and a generated layout's rows are `table/v2`'s (`genOk`).

### 16.12 Sessions of several tables (`--session-tables J`)

A session may prove its units in `J` tables (`Stmt.setupTables`, upstream's `session_tables`). `J` is the verifier's
setting (`verify --session-tables J`, default 1), never the record's. Only `verity/flock-circuit@967b8d06` and §16.13's
statements define such a session; any other statement refuses `J > 1`.

- **The tables.** Take the session's units: the record's draw's (§7.3), or without a draw the population's `0 … N − 1`.
  - The draw must be a `subset` draw of the registered population. A Bernoulli or stratified draw is refused, since its
    count is not fixed.
  - Refused as well: `J > 64`, no units, or a count `J` doesn't divide.
  - The units split into `J` equal consecutive parts of `k` units. Table `j` is `circuit-{j:02}`. Its statement is §16.10's
    drawn statement of `{"law": "subset", "population": N, "k": k, "units": part j}`, under the identity plus
    `session: {tables: J, table: j, units: "the draw's units (without a draw, the population's) in equal consecutive
    parts; table j proves part j as a subset draw of the population"}`.
  - Nothing in the record describes a table's units.
- **The session.**
  - Every table must have the same `m_pts`.
  - Σ = `SHA-512("verity/flock-circuit/sigma-tables" ‖ le64(J) ‖ Σ_0 ‖ … ‖ Σ_{J−1})`, and `root_F = Σ` (S6).
  - `Hello` names the tables in order, with this Σ.
  - A table's rep domain is `flock-circuit/fast100x2/<t>/rep<r>`.
  - U1–U3 compare the record's `unit_draw` with the session's draw, never a table's part.
- **The checks.**
  - §5.3 runs over every table: S5, S7, S8, S13 and S16 per table. S10's `y` has `2J` zeros. S11 and S17 cover exactly the
    `2J` rep streams. S14's roots are in table order, which is lexicographic.
  - Table `t`'s reps are the files `<t>.rep<r>.bin`. Each is verified against its table's statement and domain, and S18
    applies per table.
- **Live rules the record can't show.** One `Commit` carries every root. Every proof comes after the session's last
  coin (the barrier, which serves zero-knowledge; a verifier accepts without it).

### 16.13 Hidden outputs (the default `verity/flock-circuit` and `verity/flock-circuit/types`)

`@967b8d06` (§16.10) and `types@210d32e1` (§16.11) with their outputs as rows (Daniel, Oct 1); `+seed-injection` is the
selftest build. Statements composed up to verity `b3727bd9` are `@b3727bd9` and `types@b3727bd9`: the same, but META
`leaf_scheme.salt_source` names a ChaCha20 expansion of a per-proof OS key where the default's names the OS's bytes
(§16.10's tags), and the identity carries that object. Statements composed up to verity `cec3e347` are `@cec3e347`
and `types@cec3e347`: the default's, but their `--zk` identity names the pads and every table's draws ChaCha20 streams
of the prover's seed where the default's names the OS's bytes (§16.14). `Flock.HmOut` (`Stmt.setupHidden`; its soundness is `FailClosed.flock_headline_exec_hidden`, and under
`--zk` `zk_session_soundH` and `zk_session_soundHJ`, within the one theorem of §16.15) applies §16.10's definitions to every
row. The public file holds roots, hiding leaves and structure only, under M0 and `--zk`. M0's transcript is not hiding
(`NON_ZK_PROOF`, zero mask slots); only `--zk` targets that.

- **Rows.** META `ports`: the `input_ports` input rows, then one `hm96-sha512/row/v1` row (`v2` of all its bits under
  `row_schemas`) per `out_ports` entry (its name, its `w` words zero-padded to `⌈w / 64⌉ · 64`), each with its own salt,
  `rows` entry and compressions after the input rows'. `per_vu[hm96] = next_pow2(rows)`.
- **Δ** then copies, per VU, output word `k`'s bit `t` (of the returning unit, the last tail stage's port or a typed
  root's output group) to its row's message bit `t % 8` of byte `2k + t / 8`; past `w` words, the forced-zero row.
  Unit leaves and leaf cuts read input rows only; `HmRow.checkTyped` puts a template's output rows after its inputs.
- **Regions.** `Digest(p)` per row and no `Out`, so the region-word check has no shared word.
- **Public file.** §16.10's header without the content digest; the body is every row's `b ‖ c` (instance-major, inputs
  then outputs) and nothing else; one frame-v3-sha512 tree per row. Under `shared_rows` an output row's table is the
  instances' rows (`R_p = n`, never deduplicated), instance `i`'s ref `i`. A padding instance's output rows are the zero
  instance's (`dummy.outputs`) under zero salts, each row's `dummy.bc` checked.
- **Identity** adds `outputs`: `"hidden: each output port an hm96-sha512/row/v1 row hashed in the circuit; the public file
  has no output word"` (`v2` under `row_schemas`).
- **Linking** is a check (`Flock.HmLinks`). It binds a public topology; it does not hide the topology: both verifiers
  read the wiring from their own program (the descriptor, the links file), which is the circuit's, so publishing it is a
  known leak beyond the size class that later work removes. An output row is committed as an input row of the same words, so a reader
  committing it under the producer's salt has the producer's `b ‖ c` as its input leaf. META `links`
  (`[{port, leaves: [lo, hi]}]`, at least one) names the link rows: each an input row (by name) that holds template parameter
  leaves `[lo, hi)` as the one output row holds outputs, a `hm96-sha512/row/v1` row of that row's words, at most that many
  leaves, a row once, the ranges disjoint. Staging (`class_statement.Wiring`) gives each reading instance its own link-row
  table row under its writer's output-row salt. The check holds for a hidden-output statement under a template query with
  the verifier's own program (`--program`, `--partition`), over every unit of the registered population, before any draw, so
  a drawn reader is held to its writer's committed row whether the writer is drawn or not. Per unit, from the program
  (`Program.operands` of its root node and member): every parameter leaf that is another unit's output (its gate inside a
  unit's interval) lies in a link's range; each link's leaves are either all circuit inputs (an ordinary input row there) or
  exactly one writer unit's outputs, all of them, in order; that writer is a unit of the public file; and the reader's
  link-row `b ‖ c` equals the writer's output-row `b ‖ c`. Otherwise the verifier refuses, naming the leaves, the row, the
  writer and the reader (`links: cut wire [lo, hi) from unit v to unit u: …`). A template-query statement with no META
  `links` is held to the same check, so any cut wire between its units refuses it. META `links` without the program and
  partition, under any other query, or on a statement with public outputs (`setupH`) is refused. The live verifier
  (`flock-circuit serve`, `replay`, `statement`) makes the same comparison on the population's file against the
  verifier's own `--links F` (`verity/flock-links/v1`: `circuit_sha512`, the partition digest, META's link ports, and every
  `[reader, port, writer]` instance triple, derived from its program by `class_statement.links_file`); a circuit with
  `links` needs the file and one without refuses it. Not checked: a link across statements (refused), that the class
  circuit reads leaf `lo + k` from word `k` of the link row (the class lowering's claim, like the Definition it lowers), and
  public-output statements' links (their outputs are words, not rows). A salt commits one row value: reused for another
  row value, it reveals `b₁ ⊕ b₂ = x₁ ⊕ x₂`; a link row holds its writer's value, so the shared salt reveals `b₁ = b₂`,
  that the two rows are equal: the wire itself, which a hidden topology would have to hide too. A link is as good
  as both proofs (under a draw, an undrawn writer's output is unproven: the check binds the reader's input to the writer's
  committed output, not to a correct one), and a deduplicated reader table (`shared_rows`) links at most one producer
  instance per row.
- **Registered reads** are §16.10's, of `v1` rows. A registered port is an input row or an output row: a registered output
  row opens the registered root at its unit's read position, under the registered row's salt, so the instance's output is
  that row (a `row-seg/v1` or `v2` row is refused: its leaf is not the registration's `row/v1` leaf).
- **Agreement:** README.

### 16.14 `--zk` sessions

`verify --zk` (the verifier's setting) checks `flock-circuit --zk` sessions (any `J`, draws, hidden outputs) by the live
PROTOCOL.md's masked protocol on §7.4's coin tree (`Flock.Zk*`). `hooks` is upstream's `zk_identity`; `proof_class`
stays `NON_ZK_PROOF`. Its `pads` (and `tables`, for `J > 1`) name the prover's randomness: the OS's bytes for the default
statements (`Tags.osMasks`, Daniel's decision 4), ChaCha20 streams of the prover's seed for those pinned `@cec3e347` and
earlier. **Agreement:** README.

Each table's statement must have upstream's mask shape (`Zk.shapeOk`), and, which upstream does not check, its Δ and its
regions avoid the mask slot (`Zk.maskFree`, D12): no pair of `da` or `db` has a row or a column in the mask range's
word-aligned span, and every region fixes, at each of the span's words, an in-block word bit (`i + 7 < k_log`, not
free) that differs from the word's bit `i`. Otherwise the verifier refuses: `zk: a pair of Δ or a region meets the mask
slot`. The zero-knowledge proof covers no such statement: a region or a wire there would read the masks.

### 16.15 The forms the soundness proof covers

`verify` fails closed (Daniel, Oct 4): soundness is one theorem over every session it accepts, with no condition on the
circuit, the public file or the statement's form, so it refuses every statement outside the forms the proof covers. A
refusal is a completeness limit; accepting an uncovered form is a soundness gap. `Flock.ProvedScope.check` is the one
rule; widening it to another form is one line of it plus the proof that the headlines cover that form. Every session of
a statement outside them is refused once every check has passed, so a session that fails a check keeps that check's
reason, and one whose verdict is `refused: <why>: a statement form the soundness proof does not cover` passed every check,
`<why>` the first rule its statement fails:

1. an hm96-sha512 row statement (§16.10 on): `<name>, a statement before hm96-sha512 rows`;
2. untyped (§16.11): `<name>, a typed statement`;
3. retained rounds (§6.2) and `hm96-sha512/v1` leaves (`<name>, a statement without retained rounds`, `... with <leaf>
   leaves`);
4. at most one table without `--zk` (§16.12): `<J> tables in a session without --zk`;
5. no public input ports: `public input ports (--public-inputs)`;
6. no shared-row public file, except in a `--zk` session of a statement with hidden outputs whose verifier holds its
   own registered reads (`--registered F`, `ProvedScope.sharedRowsOk`; with it, §16.10's registered check opens every
   table row): `a shared-row public file (header shared_rows)`;
7. `Layout.scopeOk` of the circuit: a unit reading one input group (`a unit reading <g> input groups`) that returns the
   outputs (`a circuit whose outputs a tail stage computes (META out_net)`), of at most `2^32` rows, whose computed rows
   read only port bits among its inputs (`a unit row reading an input column that is not a port bit`), and every row a
   `sha512/row/v1` row of `16 · words` bits in at least one chunk (`a row that is not a sha512/row/v1 row of 16 · words
   bits in at least one chunk`). Under hidden outputs (§16.13) the check is `Layout.scopeOkH`: the same unit, and every
   row a `sha512/row/v1` row as above, or every row, in any mix, either an unsegmented `sha512/row/v2` row (§16.14) of
   `16 · words` bits whose last bytes and SHA-512's 17 bytes of padding fit its `hm96` slot's block
   (`128 · chunks ≤ 2 · words` and `2 · words + 17 ≤ 128 · (chunks + 1)`, `ProvedScope.v2Row`) or a `sha512/row-seg/v1`
   row of `16 · words` bits segmented as `HmRow.segShape` cuts it, in its segments' and its outer message's compressions,
   each segment's bytes and SHA-512's 17 bytes of padding ending in its last block (`ProvedScope.segRow`: a segment of
   `b` bytes in `c` compressions has `128 · (c - 1) ≤ b`), the output rows unsegmented and each holding one output port
   (`ProvedScope.bitRows`; refused as `rows that are neither all sha512/row/v1 rows of 16 · words bits in at least one
   chunk nor all unsegmented sha512/row/v2 rows of 16 · words bits ending in their hm96 block or sha512/row-seg/v1 rows
   of 16 · words bits whose segments end in their last block, the output rows unsegmented, one output port each`). A
   PoUW tile's rows, row-seg A rows and v2 B rows, are such a mix (`ZkHidden.witnessMix_scope`).

At setup, a drawn session whose statement sets up is refused when no soundness claim draws by its draw's sampling
design (`Flock.ProvedScope.drawLaw`), whatever flags the verifier holds: the claims draw by the stratified sampling
design, with or without `--zk`, and under `--zk` by the subset one (`FailClosed.SoundZJ`, `SoundZHJ`). The verdict is
`refused: <why>: no soundness claim draws by this law`, `<why>` one of `a bernoulli draw`, `a work draw`,
`a work draw with a closure` and `a subset draw without --zk`. A session without a draw is refused too, since every
claim is about a drawn session (each claim's record hypothesis reads the record's draw): it is verified against the
population's statement, and its verdict is
`refused: a session without a draw: every soundness claim is about a drawn session (D12)` only once every check has
passed and its statement's form is covered, so a session without a draw that fails a check keeps that check's reason. A
claim for the draw of every unit would cover it; none is proved.

The soundness package proves `scopeOk` and `scopeOkH` the computable checks (`FailClosed.scopeOk_eq`, `scopeOkH_eq`), and `FailClosed.*`'s headlines
are over `Verifies` (accepted and in scope) with no hypothesis on the statement's form. `FailClosed.flock_verify_sound` is
the one theorem: at every statement, with or without `--zk` and `--registered F`, it states the headline of the
statement's case (public or hidden outputs, `setupH` or `Stmt.setupHidden`; under `--zk`, one table or several; under
`--zk --registered F` at hidden outputs, `FailClosed.SoundZHJR`, `ZkReg.zk_session_soundHJR` over `VerifiesZKR`, whose
rows read no `pub.tables = none` and so cover rule 6's shared-row exception), and elsewhere `--registered F` accepts no
more than the verifier does without it. The live server refuses the same statements with the same reason
(`verity/ml/flock/live`'s `proved_scope`), also after the session's checks: its verdict is the refusal, and
`without_refusal` keeps the session's verdict otherwise. Its rule 6 reads the registered reads off the public header
(`registered`): the Lean verifier holds `--registered F` exactly when the header names them, refusing either without the
other at setup.
`flock-firewall` is a separate executable that accepts no proof, and an accept from `contract` means what
`Contract.holds_*` states, outside `flock_verify_sound`.

### 16.16 Plain leaves (`verity/flock-circuit+plain-leaves`, a recursive session's inner statement)

§16.13's statement with plain SHA-512 Merkle leaves (`flock-leaf/sha512-unsalted`, §14): a recursive session's inner
statement, staged under `FLOCK_PLAIN_LEAVES=1`, whose firewall commits every tree top under a salt of its own, so a leaf
needs none. `Tags.withPlainLeaves` changes three things.

- META `leaf_scheme` is pinned to upstream's `leaf_scheme_plain` object (`verity_flock.circuit.LEAF_SCHEME_PLAIN`):
  `{"id": "flock-leaf/sha512-unsalted", "inner": "sha512 column digest", "salt_bytes": 0,
  "use": "a recursive session's inner statement: the gateway commits every tree top under a salt, ZK off"}`.
- The identity's `leaf_scheme` is that object, and its `hashes.merkle` is `"sha512 (every Ligerito level)"`, as upstream's
  `identity` writes them for such a statement. Everything else is §16.13's identity, and on `--coins os` its live-OS form.
- Every opening carries an empty `opened_salts`, as every opening of PR #83's prover does from `2908d078` on
  (`Setup.openedSalts`, set on every hm96-row statement); `631567f7`'s unsalted openings carry none. Every level's leaf and
  node are SHA-512 (`merkle_hash` 2).

The verifier identifies the statement (`statement` prints its identity and digest), but `verify` refuses any
statement whose leaf scheme is that object (`Tags.plainLeaves`), with or without `--zk`, until a soundness headline
covers it: every hidden-output headline needs `merkleLeaf = "hm96-sha512/v1"`. The statement is ZK off, and upstream's
`--zk` refuses it too.

**A registered description.** When the inner statement is a private circuit's (its description, the circuit's `op`,
`src` and `out` bits, one row port every instance reads), the developer registers the description's rows once (§16.10's
registration: `hm96-sha512/row/v1` rows under salts drawn once, a frame-v3-sha512 root under the registration's domain),
and the statement opens them against that root: the header's `registered` names the description port, and each of its
table rows' position and path (`verity_flock.inner_registered`). The verifier that holds the root (`statement
--registered F` on a plain-leaves statement; V*'s native side, `InnerFold.lean --registered F`) refuses the statement
unless, for each port of `F` (`Registered.checkDescription`):

1. the header's `registered` names exactly `F`'s ports, each a row of the circuit, `hm96-sha512/row/v1` rows of the
   registered width (`Port.fits`), and a table of a shared-row file;
2. the port's table is the registration in order: as many rows as the registration's leaves, table row `r` at position
   `r`, its `b ‖ c` opening the registered root there through its path;
3. every instance's ref names a row of the table; and
4. every table row is some instance's ref: the statement reads every registered row (D14).

Each refusal names the port and the value it compared (the table's rows and the registration's leaves, the row, its
stated position, the instance and its ref). No position depends on which unit an instance is, so the units need not be
derived from the verifier's program, as §16.10's reads need them. Without `F`, a statement naming registered reads is
refused as §16.10 refuses one. The description's registered value is its packed row (16 description bits to a u16 word,
as the class statement lays out the parameter), so its `words` are the port's.

## 17. Acceptance, assumptions and trust

### 17.1 How the Lean verifier becomes the verifier of record

Upstream's Rust verifier (below, with `verity/ml/flock/live`'s `ReplayChallenger`) is the reference in a cross-validation test,
run in CI on every change to the Lean verifier, the statements or the pinned files:

1. **Agreement on the corpus.** For every entry of `vectors.json` (§18), both verifiers produce the same verdict:
   - every honest session: both accept, except a session of a set marked `lean_refuses`, a drawn session of one marked
     `lean_refuses_drawn` and a session without a draw of one marked `lean_refuses_drawless` (D12 below);
   - every record-level negative: both reject, except the recipes marked `accept` (both accept), `info` (both return
     the same verdict) and the divergence recipes D1, D2 below (the Lean verifier rejects, upstream accepts);
   - every red-team forgery, materialised as a record and proofs: both reject.
2. **Fuzzing.** Mutations of honest vectors (a byte of a proof, a coin, a round digest, a record field, the public input)
   are generated with a fixed seed per CI run and a rolling seed per nightly run. Both verifiers must return the same
   verdict on each, and no mutation of a proof byte or of a round digest may be accepted by either.
3. **Coverage.** The corpus covers every statement and layout in §16 but `--zk` (§16.14), `m` from 22 to 35 where a
   statement reaches it, `os` and `seed` coins (§7), every check id of §5.3 and every rejection in §8–§14 at least once.

A disagreement fails CI and is resolved against the paper and this spec, never by copying upstream's behaviour without a
spec change. After the gate passes, production runs only the Lean verifier; upstream's stays in CI as a regression oracle.

"Upstream's verifier" means `Server::from_record` then `Server::finish` with the statement's `StatementVerifier` (and,
for the pre-G2 records of §5.1, the statement binary's `replay` subcommand). The spec is deliberately stricter in these
places, and the corpus marks the vectors that exercise them with upstream's expected verdict (`upstream_accepted` in an
expectation):

- **D1:** `root_F = Σ` (S6). Upstream's server skips it for statements with link points; the replay tools check it.
- **D2:** no stream other than the configured rep streams (S11). Upstream ignores zero-round child streams and streams
  outside the rep prefixes.
- **D3:** `link.sigma = Σ` (S4). Closed upstream at PR #83 `b3baabd8`.
- **D4:** retained round bytes hash to the recorded digest (S12, §6.2). Closed upstream at `631567f7`: `from_record`
  requires `msg` on every round when `Hello` says `"rounds": "bytes retained"`, and checks its length and digest.
- **D5:** registered reads (§16.10). Upstream ignores the header's `registered` (the public digest binds it); the spec
  refuses a row that does not open its registered root.
- **D6:** a `--zk` record's `coin_tree` (§7.4); upstream ignores it.
- **D7:** the `sha512x3` and `hm96` nets are the verifier's (`Flock.HmNets`); upstream checks only their port widths.
- **D8:** at least one unit per VU (§16.10). Upstream's `Composite::check` in the pinned builds that have tails (`25519ba1`,
  `967b8d06`) lets `units_per_vu = 0` through when a tail returns the outputs; the spec refuses it, as
  `verity/ml/flock/live`'s now does. No vector exercises it.
- **D9:** at least one VU per block, and a power of two of them (§16.10). Upstream's `Composite::check` in the same builds
  lets `vus_per_block = 0` through when a tail returns the outputs; the spec refuses it, and every `vus_per_block` not a
  power of two, as `verity/ml/flock/live`'s now does. No vector exercises either.
- **D10:** the header's `identifier` (§16.10). Upstream ignores it (the public digest binds it); the spec refuses one that
  is not the verifier's own `--identifier F`. No vector exercises it; `verity/ml/flock/tests/test_lean_identifier.py` runs
  it on the Lean verifier.
- **D11:** each expanded circuit's output groups end inside its slot, `col + words ≤ 2^{unit_log} / 128` (§16.5,
  `Net.ofRows`). Upstream's `IrUnitNet::parse` in every pinned build checks only that a group is an aligned power of two of
  words, and its `Composite::check` doesn't bound an output column, so upstream accepts a statement that moves an output
  group past the slot; the spec refuses it, as `verity/ml/flock/live`'s parser now does. Honest statements pass both: every
  writer (`ir_lower.layout`, `derive`, `tail.stage_layout`) closes its output groups before the constant row, so
  `128 · (col + words) ≤ constPos < useful ≤ 2^{unit_log}`. No vector exercises it;
  `verity/ml/flock/tests/test_lean_out_group_slot.py` runs it on the Lean verifier.
- **D12:** a statement form the soundness proof does not cover (§16.15). The pinned upstream builds verify it; the spec
  refuses every session once every check has passed, as `verity/ml/flock/live`'s now does. A set marked `lean_refuses`
  in `vectors.json` expects that refusal of each session that upstream accepts, and of every other session its own reason
  (`agree.py --lean-refuses`): a session refused for its form alone passed every check, and is compared as accepted. So,
  too, a draw by a sampling design no soundness claim draws by (§16.15: a bernoulli draw, a work draw with or without a
  closure, and a subset draw without `--zk`), which the Lean verifier alone refuses, at setup (the live server and
  upstream verify it): a set marked `lean_refuses_drawn` expects that refusal of every drawn session
  (`agree.py --lean-refuses-drawn`). `verity/ml/flock/tests/test_lean_draw_laws.py` runs each sampling design on the
  Lean verifier. So, too, a session without a draw (§16.15), which the Lean verifier alone refuses, and only once every
  check has passed: a set marked `lean_refuses_drawless` expects that refusal of each session without a draw that
  upstream accepts, and of every other session without a draw its own reason (`agree.py --lean-refuses-drawless`).
  `agree.py`'s `passed_every_check` counts the sessions refused last and for nothing else. So, too, under `--zk`, a
  statement whose Δ or regions meet the mask slot (§16.14, `Zk.maskFree`), which the Lean verifier alone refuses (the
  live server and upstream verify it); honest `--zk` statements avoid it, and no vector exercises it;
  `verity/ml/flock/tests/test_lean_zk_mask.py` runs it on the Lean verifier.
- **D13:** a unit net named `rec-open/…` or `rec-open-stacked/…` is the verifier's V* opening net (`Flock.RecOpen`,
  §16.10). Upstream takes the unit's net as the file gives it. The spec refuses such a unit whose name is not
  `rec-open/l<LANES>-h<H>` (or `rec-open-stacked/l<LANES>-h<H>`) at valid parameters, one whose port bits are not the
  opening's at them, one whose rows or copies' leaves are not the opening's inputs in order, and one whose net differs
  from the verifier's. No vector exercises it; `test_rec_open_unit_of_other_bits_is_refused_before_its_net_is_built` runs
  the bits' refusal on the Lean verifier, `verity/core/service/flock/Flock/Test/RecOpenLeaves.lean` the rows' and
  leaves', and `test_the_lean_verifier_reads_rs_level_under_its_own_name_as_the_opening` a stacked unit's acceptance and
  its net's.
- **D14:** U4 (§7.3, §16.10): a drawn statement must draw in full each unit that reads a round's commitment. Upstream
  ignores the header's `registered` (D5) and so cannot tell which units read one; the spec refuses the draw at setup. No
  vector exercises it: no set's draw meets a port of a round's commitment (`rec-c<i>`, `rec-top-L<l>`, R's and the sparse
  phase's). `Flock.Test.Full` runs it on synthetic ports and draws.
- **D15:** an inner statement's registered description (§16.16). Upstream ignores the header's `registered`; the spec
  refuses a description table that is not the registration in order, a row that does not open the registered root, and a
  registered row no instance reads. No vector exercises it; `verity/ml/flock/tests/test_inner_registered.py` runs each
  refusal on the Lean verifier.

D3 and D4 are closed by `631567f7`; the corpus's `upstream_checks` names them per set (`--upstream-checks` in
`redigest.py` and `fuzz.py`).

Elsewhere the spec follows upstream, even where upstream is lenient without soundness cost: record keys for unconfigured
tables, `matrix_evals` and bytes after the proof pair are ignored, nonces at proof-of-work sites are accepted as
messages, and `r_eq = 1` in a zerocheck round gives `G(0) = 0`. Where upstream panics (`m` without a schedule,
`|s_hat_v| ≠ 128`), its statement verifier catches the panic as a rejection; the spec rejects explicitly.

### 17.2 Assumptions (for Table 1 and the soundness lane)

Soundness of a verdict rests on:

- **Statistical:** the Flock protocol's round-by-round errors for `fast100` (§15), squared over two reps on one root with
  live coins: 2^-195.5 per table by §15's ledger. The soundness lane proves 2^-205 per table in Lean, at the analysis
  slack `η = 1/200`, with no named assumption: mutual correlated agreement up to the Johnson bound (BCHKS25 Theorem
  4.6) is proved from DKT26 in ArkLib, and ring switching and Ligerito's list-decoding analysis are proved there too
  (`lean/soundness/README.md` §1.1 and §1.6, and its `ASSUMPTIONS.md`).
- **Coins:** with `os` coins, each coin is uniform and unknown to the prover before it sent the round the coin answers.
  With `seed` coins, additionally: the seed commitment hides the seed, and SHA-256 in counter mode is a PRF (§7.2).
- **Hashes:** SHA-256 collision resistance for the Merkle trees, the round digests and the statement binding (Table 1:
  `cr/sha-256`); the tight Merkle bound is in the random-oracle model. For the target statement, the Merkle trees and the
  Halevi–Micali leaves rest on SHA-512 collision resistance instead (`cr/sha-512`). With retained round bytes (§6.2), R2
  needs no hash assumption. BLAKE3 collision resistance for the row-digest leaves
  (the commitment scheme's assumption, not the verifier's).
- **Circuits:** the pinned circuits mean what the statement says they mean. Level 3 proves the lowering correct for every
  pinned circuit; that the circuit matches silicon is empirical and outside the verifier.
- **Custody:** the session record and proofs are the ones the verifier's server wrote (the record is evidence only under
  the verifier's custody; nothing in it authenticates its author).

### 17.3 Trusted components

- **Lean's kernel,** for the level-3 theorems (re-checked with an independent checker).
- **Lean's compiler, runtime and the C compiler, for the running binary.** Proofs are about Lean definitions, not the
  machine code that runs. This trust point is new once only the Lean verifier runs. It is mitigated by the agreement tests
  of §17.1: an independent implementation (upstream's Rust verifier, compiled by rustc) must agree on the whole corpus and
  on fuzzed mutations in CI, so a miscompilation that changes a verdict on any tested input fails CI. It is not mitigated
  for inputs no test reaches; the fuzzing and the coverage requirement narrow that gap, they do not close it.
- **No `@[extern]` or `@[implemented_by]` substitution** in verification logic; if one is ever added it joins this list.
- **The Rust cross-check's own trust base** (rustc, upstream's code) matters only as a test oracle: a bug shared by both
  implementations would pass CI, which is why the Lean verifier is written from the paper and this spec, not from
  upstream's code.

### 17.4 Assurance stages

Assurance is staged and reported like a16z's zkVM security stages (Thaler, 2025). Stage 1 covers the protocol, stage 2
the verifier implementation, stage 3 the prover. Level 3 is this lane's; level 4a is the soundness lane's. A stage is
complete for a statement only when every constraint system in it (row leaf, units, lookups, tail stages) is covered. That
is the article's recursion caveat, applied to the slots of one block.

| stage | here | level |
|---|---|---|
| 1a PIOP soundness | zerocheck with univariate skip, lincheck with the verifier-computed fold, ring switching, and the region claims, for F2 block R1CS | 4a |
| 1b PCS binding | Ligerito, with the paper's proximity and correlated-agreement results as named axioms; Merkle binding under the scheme's collision resistance (SHA-256, or SHA-512 on the target statement; an explicit collision extractor) | 4a; Merkle 3 |
| 1c Fiat–Shamir | not used (interactive coins). Replaced by **session binding**: every coin is issued only after the framed bytes it answers are fixed by their SHA-256 digest (R2); both reps bind one root (S13, S16, S18); Σ, the digest and the link points precede every Flock coin (S4, S14, §9.1) | 3 |
| 1d constraint system = semantics | the lowering theorem for every pinned circuit against its reference function (soundness and totality), and the assembly: the fold equals the multilinear extension of `I ⊗ A_0 + Δ`, the region values equal the extension of the known bytes, and Δ makes the stated copies | 3 |
| 1e glue | one theorem, `FailClosed.flock_verify_sound`, over every statement `verify` accepts, `setupH`'s and `Stmt.setupHidden`'s (§16.13, §16.15): `verify` accepts ⇒ every instance's outputs are its reference function of its inputs, except with probability δ. That the public file's `frame_v3` roots, domains and schemas are those of its `b ‖ c` rows and output words is the verifier's native check (`checkPublic`, §16.10), stated deterministically by `FailClosed.flock_verify_public_ties`, not by `flock_verify_sound`. Without `--zk`, no zero-knowledge claim (`NON_ZK_PROOF`) | 4a |
| 2 implementation = protocol | the executable is the definition the theorems are about, so no second implementation has to be matched. The field code is proved to be GF(2^128) and K (level 3, below). What remains: the decoders implement §8 and §5.1; the hash code meets FIPS 180-4 and the BLAKE3 spec (tested against vectors, used as assumptions); compiler and runtime trust (§17.3) | 3 |
| 3 prover completeness | out of scope (upstream's prover). Every honest session in the corpus is accepted | tests |

The lowering theorems are proved about a certified checker, over all inputs. The verifier runs that checker on the pinned
files at setup and refuses a circuit it cannot certify. So no proof evaluates a pinned file in the kernel, and no proof
uses `native_decide`.

Level-3 status (`lean/FlockProofs`, kernel-checked; the Lean audit, `tools/verity/lean/check.py` with `../../../core/service/flock/lean-audit.json`,
allows only `propext`, `Classical.choice` and `Quot.sound`):
- **1b, Merkle: an explicit collision extractor.** A fixed hash with 32- or 64-byte digests has collisions by counting, so
  a proposition "a collision exists" holds outright and says nothing about the verifier. The statements therefore carry the
  reduction as computable functions from the verifier's accepted checks to an explicit pair of inputs, with theorems that
  the pair collides:
  - `merklePair` (through `opensPair` and `climbPair`) maps two Merkle checks the verifier accepts against one cap, whose
    rows differ at a query, to a `MerklePair`. That is either the first node whose two parents are one while its children
    differ, or the two rows when their leaves are one.
  - `merkle_binding` proves the pair collides (`MerklePair.Collides`), for any `MerkleScheme` whose nodes and leaves have its
    digest width (`Sized`; `MerkleScheme.all_sized` for every known scheme, from `Sha256.hash_size` and `Sha512.hash_size`,
    proved from the definitions). The collision is either two different pairs of digest-width children with one node, or
    two different rows with one leaf, under any salts.
  - `MerklePair.shaInputs` and `MerklePair.hm96Inputs` map the pair to the hash's own inputs. `sha256_inputs`,
    `sha512_inputs` and `hm96_inputs` prove those two inputs differ and have one digest:
    - a node's input is its two children, concatenated at a fixed split, so different pairs are different inputs;
    - the row encoding is injective (`rowBytes_inj`);
    - for `hm96-sha512/v1`, `Hm96.leafPair` gives the outer inputs, or the salt digest's (`leafPair_collides`), or the rows'
      when their row hashes are one.
  - The propositions `merkle_binding_collision` and `MerkleScheme.collision_hash` are corollaries.
  - The executable only ever hashes digest-width children: `merkleCheck` refuses a sibling or cap entry of another width,
    and the running digests are hash outputs.
  - Nodes and leaves are not tag-separated in the unsalted schemes, and binding doesn't need it. Both openings climb from
    a leaf at one position through `d − c` nodes, so a leaf is only ever compared with a leaf. As a backstop, under
    `fast100` an opened row has 16, 32 or 64 elements (256, 512 or 1,024 bytes), while a node's input is 64 or 128 bytes.
- **1c, session binding:** `squeeze_answers` and `squeeze_retained` are proved. A squeeze that succeeds returns exactly
  round k's recorded coins, and only if the framed bytes are that round's retained bytes, or hash to its digest.
- **2, the fields** (`lean/level3`, a second package: Mathlib at the soundness package's pin, the executable by path; its
  own `lean-audit.json` under the same audit):
  - `F128.poly_mul`, `F128.poly_add`: the executable's product (carry-less multiply, Karatsuba, the GHASH reduction) and sum
    are polynomial multiplication and addition over GF(2) modulo `G = X^128 + X^7 + X^2 + X + 1`. `toAdj` is a bijection
    `F128 ≅ GF(2)[X]/(G)` that preserves both.
  - `G_irreducible`: Rabin's test, with the kernel checking `x^(2^128) = x` and a Bézout witness for `x^(2^64) + x` on the
    executable's own multiplication (`decide +kernel`, no `native_decide`). So `GF128 := F128` is a `Field` whose `+`, `*`
    and inverse are the executable's `add`, `mul` and `inv` (`GF128.inv_eq`, including `inv 0 = 0`).
  - `Q_irreducible`: `Y^2 + Y + x⁻¹` is irreducible over GF128, because the trace of `x⁻¹` is 1 (computed the same way) and
    the trace vanishes on every `y^2 + y`. So `GF256 := F256` is a `Field` whose `+` and `*` are the executable's three-
    product ladder (`GF256.mul_eq`), and `F256.xinv` is `x⁻¹` (`GF128.xinv_mul_x`).
- **The seam** (`FlockLevel3/Seam.lean`): `embed : GF128 →+* GF256` (`c ↦ ⟨c, 0⟩`), `u = ⟨0, 1⟩`, `ofLimbs` (an executable
  `⟨c0, c1⟩` is `embed c0 + embed c1 · u`, the soundness model's `Arith.ofLimbs`) and `u_root` (`u² + u + x⁻¹ = 0`).
- **1d, assembly, first lemma** (`FlockLevel3/Assembly.lean`): `eqTable_get`: `eqTable r` has `2^|r|` entries and entry
  `b` is `Π_j (r_j if bit j of b, else 1 + r_j)`, the soundness model's `eqAt`.
- **The soundness lane's `Arith.Correct`, every field** (`ArithFacts`, `Pinned`, `Nodes`, `NovelBasis`), for
  `F = GF128`, `K = GF256` and the executable's functions:
  - `card_F`, `card_K`, the characteristics;
  - packing (`unpack_pack`, `pack_unpack`, `unpack_add`), with bit `v` as the coefficient of `x^v`;
  - `pinned_indep`, with an inverse bit matrix checked by the kernel on the executable's `eqTable pinned`, and
    `pinned_ne_one`;
  - `nodes`: `lagS`/`lagΛ` are the Lagrange weights on `φ8(0…63)` and its coset, and `combW` recovers `Pcomb` (φ8 is
    additive, and the barycentric denominators are `den 6`, `den 7` by translation);
  - `omega_injective` (`d ≤ 64`), `xhat_poly` (`L ≤ 40`: the subspace polynomials, their normalisers checked nonzero),
    `ofLimbs_bijective`, `lo_balanced`.
- **1d, the fold, first lemmas** (`Fold.lean`): `slot_factor`, so a slot's in-block tensor is its local tensor times
  `eq(ρ_hi, q)`, which is the reason `Stmt.fold` folds each slot type once; and `foldT_get`, so the sparse fold is `Xᵀe`.
- **A lookup slot's rows compute `table[index]`** (`LookupRows.lean`, `build_computes`, over any field of characteristic
  2, for every table and index):
  - `build` encodes rows `RA`, `RB` into its net (`Sparse.ofRows`). A witness satisfying every row, with the product rows'
    B side the table's as `Lookup.foldB` folds it (`tableB`), and the constant 1, has:
    - index bits that are bits (`z = z·z`);
    - output bit `j < bits` equal to bit `j` of the table word the index bits name.
  - The decoders are one-hot (`decode_spec`), each product row picks one table bit, and each output row sums them.
  - `build_spec` walks `build`'s loops.
  - The one hypothesis on the net is `useful ≤ KONST`, which every table meets: the pinned tables' slots have about 50,000
    rows, and `KONST` is 2^48 − 1.
  - `net.a.row r = RA[r]` is `Sparse.ofRows`'s round trip (`ofRows_row`, in the soundness package, #177).

## 18. Test vectors

`verity/ml/flock/verifier/vectors.json` (`flock-verifier-vectors/v1`) lists them; the bytes stay in the evidence store
(`research data fetch <record> --path <path>`).

- **Honest sessions** (`honest`, `expect: accept`): pure-block, IR frame v2 and v3, IR sampling (legacy L-SAMPLE) and
  vllm-block cells, and the PR #83 circuit statement, each with its record, verifier run and commit, public input files,
  `Hello` and statement digest. The earliest pure-block sets are pre-G2 records (§5.1).
- **Record-level negatives** (`file_negatives`): recipes applied to an honest session, each with its expected verdict. The
  required acceptance is `unused_high_bits_of_query_coin`: flipping bit 127 of a query coin must not change the verdict
  (§13.5). Two informational cases, accepted on honest proofs, are §19's A1 and A2.
- **Forgeries** (`forgeries`): the red-team runs, among them R-BREAK (two honest reps over different witnesses, which
  upstream's Fiat–Shamir r2 verifier accepts; this spec rejects it at S13/S16/S18), and the plan that materialises them.
- **Circuit statements** (`replayable`, `live`): staged circuits with sessions, mutants and selftest records (README).
- **Spec self-check:** `transcript_check.py` rebuilds every round of a session from this spec alone (§6, §8, §9–§13,
  including `T` from §12.3) and compares each round's SHA-256 and coin count with the record. It checks the transcript and
  the layout, not the verifier's equations.
- **Missing:** no stored session uses `seed` coins or `--zk`, and none exercises the legacy L-TAIL path with a forged cut
  word as bytes.

## 19. Ambiguities resolved from upstream's behaviour

Each item is a point where the paper is silent, or differs, and the spec follows upstream (`b684b12` plus our layer).

| id | topic | the paper | the spec (upstream) |
|---|---|---|---|
| A1 | zerocheck coins | `r ∈ F^{m−k_s}`, 7 fixed | 6 extra coins `r_skip` squeezed first and never read; kept so the transcript matches. Flipping any of their bits leaves any verdict unchanged |
| A2 | last coin | — | Ligerito's final claim-batching coin: read, but an honest proof satisfies the final equation for every value, so flipping its bits leaves an honest proof accepted (a dishonest proof's check depends on it) |
| A3 | vanishing on S | the verifier checks `P_ab − P_z` vanishes on S; degree ≤ 126 | imposed by interpolating over S ∪ Λ with S = 0; degree ≤ 127 (Schwartz–Zippel `127/|F|`) |
| A4 | `v†` | the verifier computes `P_z(λ)` | the prover sends `final_c_eval`; the verifier compares it with its interpolation; not absorbed |
| A5 | residual sumcheck encoding | unspecified | `(G(1), G(∞))` of the eq-stripped quadratic, `G(0)` derived, no eq factor in the final check; `inv(0) = 0` |
| A6 | pinned coordinates | `r_i = x^{2^{i−1}}/(1+x^{2^{i−1}})`, 3 in F8 and 4 in F, order unspecified | address bits 6–8 (F8 via φ8), 9–12 (F) |
| A7 | lincheck | `(m_0 − k_s + 1)`-round sumcheck, then claims on `Â_0`, `B̂_0` | the verifier computes `α·A_0ᵀe + B_0ᵀe`; `k_log − 6` rounds, top bit first; the prover sends 64 skip values; fresh φ8 coin; `matrix_evals` unused and ignored |
| A8 | constant wire | absent | extra coin `β`; `comb[pin] += β`, target `+ β`; needs the pin to be 1 in every block |
| A9 | ring switching | one claim, own `(m−7)`-round sumcheck, B̂ via MBP | each claim switched with its own 7 coins, then one independent coin per claim; no standalone sumcheck: Ligerito's F256 folding sumcheck replaces it; B̂ at K points |
| A10 | packing | `(r_hi, r_lo)` generic | the 7 low address bits; `b_v = x^v` |
| A11 | Ligerito | uniform queries; unique or list decoding; OOD after every commitment | as §13: stratified queries, GF(2^256) folding challenges, OOD at level 0 despite Remark 11 (see §13), clear residual of `2^e` values |
| A12 | proof of work | a small amount of grinding | Fast100 grinds only Ligerito's claim and consistency batching; under live coins every nonce is a message and is never checked (R4); structural zero-nonce checks remain where upstream has them (§8, §12, §13) |
| A13 | statement binding | absent | label, statement digest and Merkle cap absorbed before the first coin |
| A14 | extra claims | step 4, an auxiliary protocol | public-region claims at the verifier's link points, opened in the same batch as `ab` and `c`; their points and values are not absorbed (the session order fixes them before any Flock coin) |
| A15 | `root_F` | — | the live server skips it when `points = 2`; the spec requires it (S6, D1), as the replay tools do |
| A16 | `m > 35` | — | no Fast100 schedule; upstream panics (caught as a rejection); the spec rejects explicitly |

## 20. Mapping onto Lean

One module per section, each a set of pure functions with total signatures:

~~~text
Flock.Field.GF2_128   §2.1   add, mul, inv (inv 0 = 0), bytes ↔ (lo, hi)
Flock.Field.GF2_8     §2.2   φ8
Flock.Field.GF2_256   §2.4
Flock.Hash            §2.5   SHA-256, BLAKE3 compression, keyed tree
Flock.Record          §5     decode : Json → Except Reason Record;  sessionChecks : Config → Record → Except Reason Unit
Flock.Transcript      §6     State := { pending : ByteArray, k : Nat };  absorb, squeeze : State → Stream → Except Reason (State × List F)
Flock.Coins           §7     derive, stream, coins_q (seed audit only)
Flock.Proof           §8     decode : ByteArray → Except Reason Proof   (bincode, hand-written)
Flock.Zerocheck       §9     verify : Params → ZerocheckProof → State → Except Reason (ZcClaim × State)
Flock.Lincheck        §10    verify : Params → Matrices → … → Except Reason (LcClaim × State)
Flock.Opening         §12    ring switching, batching
Flock.Ligerito        §13
Flock.Merkle          §14    verifyPath / verifyBatch
Flock.Schedule        §15    fast100 : Nat → Option Schedule   (the table as data)
Flock.Statement.*     §16    PureBlock, IrFrame, Circuit; Legacy (L-CV, L-ACC, L-TAIL, L-SAMPLE)
Flock.Verify          §3     verify : Config → Record → Proofs → PublicInput → Verdict
~~~

The transcript state is threaded explicitly (a state monad over `Except`), never global. Level 3's theorems attach to
`Flock.Statement.*` (lowering correctness for every pinned circuit), `Flock.Record` and `Flock.Transcript` (session
binding: every coin follows the bytes it answers, both reps bind one root, the link context precedes every coin),
`Flock.Merkle` and `Flock.Field`. The soundness lane (level 4a) reasons about `Flock.Verify` with the paper's theorems as
named axioms.
