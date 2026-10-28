# Proof of useful space (POUS)

A GPU keeps an encoding `C` of model weights `W` in its memory and decodes `W` from it on every use. A timed audit must
certify that the GPU still holds more than a `ρ` fraction of `C`. In the ideal-permutation model, answering a random
block within the deadline requires the block itself, because rebuilding it from a smaller state takes more sequential
oracle rounds than the deadline allows. At the reference Feistel-SHAKE256 `Π₂` this does not hold: a prover rebuilds a
label from 1.53% of `C` in 481 Keccak-f (red team; "Main caveat"). This directory
holds the reference implementation, `verity.core.protocols.pous` ("Codecs" through "Tests" below), and the Lean formalization,
`lean/` (from "In Lean" on): the definitions, the guarantees and the proofs so far.

## Main caveat: H is a monolithic random oracle (narrow-state H)

**No current result gives security with a concrete hash.** Every result that models the hash H as a random oracle with
`m`-bit outputs holds only in that model: B3's chain, B5 and the stacked audits, `DenseMeets`, and P3's labels (the band
certificate is pure combinatorics, so it is unaffected). A real hash whose `m`-bit output comes from a narrower internal
state breaks the instantiation. The adversary stores each label's pre-squeeze state and regenerates the labels from
those states: with SHAKE's 1600-bit state, Dense in one round from about 20% of `C`, and P3 in three rounds from about
39%. This is the Ristenpart–Shacham–Shrimpton limit: indifferentiability does not carry over to storage games, so no
ideal-hash result transfers to a concrete hash by citation. So "no named assumption" for `DenseMeets` means within the
ideal-hash model only, and so do the dense scheme's deployment results below. The chosen fix is H as the overwrite chain
over an ideal permutation `Π₂` of width `2m + 512`, proved in the ideal-permutation model. Its pebbling side and its ex
post facto bounds are proved in Lean (`Security/Proofs/Pous/Chain/Flat.lean`): `ChainExPostFacto` (B5 for `Π₂`)
for dense, and its DAG-generic form `ChainExPostFactoG` for the band, with the guarantees below (P3 with the concrete
`H`, the band, and the dense chain scheme). They hold in the ideal-permutation model only. `Π₂` itself (Feistel-SHAKE)
was taken as a heuristic ideal permutation. Sources: the POUS store's `docs/p3-cryptanalysis.md`, "Narrow-state H", and
`docs/band-vs-dense.md`.

**The reference `Π₂` breaks the same way, as the red team showed.** The overwrite chain fixes the
narrow-state attack only over an ideal `Π₂`; the reference Feistel-SHAKE256 `Π₂` has the weakness inside it. Each
round function `F_i` is a long SHAKE256 squeeze from a 200-byte post-absorb state, and a chain key, the first `m`
bits of the pad call's output, is the first `m` bits of `pad ⊕ F_1 ⊕ F_3 ⊕ F_5 ⊕ F_7 ⊕ F_9` (each `F_i` at its
round's input), with `pad` a known constant. A
prover that keeps the five odd rounds' post-absorb states of each block's pad call (1000 B per 64 KiB block, 1.53% of
`C`) rebuilds any whole label by squeezing them in parallel: 481 sequential Keccak-f (24 µs at 50 ns), with no `Π₂`
call and no chain walk, 20 to 130 times inside every deployed deadline. The attack replays bit-exactly against the
band (`band-chain/d12/v1`) and dense (`dense-chain/v1`) encoders, and P3's key floor breaks identically. The Lean
statements (`Meets`, `GradedMeets`, `ContinuousAudit`, `BandPub*`, `P3Concrete*`) stay sound in their models; what
fails is the `feistel-indifferentiability` claim, now refuted, and every sentence below that converts a wall-clock
deadline into a count of whole `Π₂` calls (`call_ns`, the effective `D`, the cliff, the cap): each holds in the
ideal-permutation model only, and none bounds an adversary at the reference `Π₂`.

## The problem: timed audit, help accounted

- **Syntax.** A trusted encoder `Enc(W) → (pp, C, vk)` runs once and erases its secrets. `W` is arbitrary and fixed
  before any setup randomness (all zeros included). Decoding `Dec(pp, C) = W` is public and bit-exact. `C` is `N`
  blocks of `b` bits.
- **Parameters.** `λ = 128`; `|C| + |pp| ≤ 1.05·|W|`; retention `ρ = 18/19`; soundness `δ = 1/100` plus the
  construction's `ε_crypto`.
- **Time.** A responder is `(D, Q)`-bounded: `D` sequential steps and `Q` operations in total. In ideal-model proofs a
  step is one round of parallel queries to the primitive, and `Q` counts queries.
- **Game.** The adversary picks `W`; then `Enc(W)`. Preprocessing, with no time limit, leaves a state `σ` and nothing
  else. The memory challenger issues `k` uniform block indices, all at once or one at a time. Each answer is due within
  the deadline, so the responder is `(D, Q)`-bounded, with no outside help. Whenever a challenge arrives the adversary's
  state is at most `ρ|C|` bits; `W` and `pp` are free.
- **Deployment: help is accounted, not excluded.** No guarantee assumes trusted prover hardware,
  so nothing attests the prover's machine and nothing asks that its responder be isolated. The game's "no outside
  help" is read through the help-accounted row of console's actual-circuit model, the model a certificate cites under
  the id `perfect-isolation` (kept from the earlier model, since certificates pin it): on PoUS's challenge schedule,
  some challenge tick `t_j` has `mem_X(t_j) + 2^e·h ≥ S` for the audited location `X`, where `h` is the help `X`
  received. It is proved at `e = 0` for a location that does no arithmetic modulo `p`, for challenges on a
  fixed schedule. PoUS keeps sending each challenge when the last answer arrives, and the model's bridge is to cover
  that schedule; that bridge is not written yet. An accepted audit bounds what `X` held plus what it
  was helped, never `X`'s memory alone.
- **Requirement.** `Pr[every answer correct and on time] ≤ δ + ε_crypto`. The honest prover stores `C` and answers by
  lookup.
- **Cost** (not modeled in Lean): encoded matmuls at most 2× the tuned plaintext sequence, decoding on every use.

The floor is `k = 86` challenges (`ρ^k ≤ δ`). Locality (bandwidth-limited off-device storage) is a later relaxation.

## Codecs

Three codecs share everything above the scheme (setup, layout, `vk`, audits):

- **The band scheme (`band.BAND12`), the secure-first default** (POUS `docs/band-vs-dense.md`). One layer on the band
  DAG with in-degree 12, Lean's `chainBand` (`lean/submissions/band-chain/BandChainModel.lean`).
- **The dense scheme (`dense.DENSE`), selectable.** One layer on the complete DAG, Lean's `chainDense`
  (`lean/submissions/sponge-dense/SpongeModel.lean` §9). It is the same construction with every earlier block a parent.
- **P3 (`scheme.P3`), the fast research codec.** Two layers with a transpose, Lean's `p3Scheme`
  (`lean/submissions/p3-meets/P3Meets.lean`) over `p3Model'` (`pebbling/ColumnAwareDraft.lean`) and
  `Definitions.Pous.Model.Invertible`.

The concrete primitives are those of POUS `docs/p3-instantiation.md`. Each set of parameters names its codec
(`params.codec`), and every function that takes a codec defaults to it. P2 (`schemes/p2.py`, experimental) is a
`protocol.Scheme` and not a codec: see "The generic protocol".

The secure-first scheme is the band graph with in-degree `d = 12` (POUS
`docs/band-vs-dense.md`), superseding the dense choice of `docs/scheme-choice-e2e.md`. Dense stays selectable
(`dense.POINT_64K`).

## The band scheme (secure-first)

`band.SECURE_FIRST` is `BandParams(512, 524288, d = 12)`: 512 blocks of 65,536 B per segment (32 MiB), audited with
`k = 111` challenges and a certificate for `D = 5` calls of `Π₂`, each at least 241.5 µs
(`Pous.SecurityProofs.BandMultiMeetsFamily`, for every `d ≥ 5` and every segment count, with no hypothesis, in the
ideal-permutation model; the reference `Π₂` does not meet it: "Main caveat"). Segment `s`
is exactly `chainBand B d ℓ 512 (dom tag_s)`. It is the dense scheme
below with one change: block `v` absorbs only its `d` newest parents, `C_{v-1}, ..., C_{max(0, v-d)}` (Lean
`parentsDesc (bandDAG B d) v`), then the pad.

**The per-call floor** (`primitives.pi2_keccak_f`, `pi2_call_ns`). A 64 KiB `Π₂` call is ten Feistel rounds, and
each round function absorbs a 2-byte prefix and a 65,568-byte half: 65,570 bytes, 483 blocks at SHAKE256's rate of
136. Round `i + 1`'s absorb block `k` needs only round `i`'s squeeze block `k`, so the rounds stream, and in the
overwrite chain a call boundary is a round boundary. A call in a chain costs 10 × 483 = 4,830 sequential Keccak-f,
241.5 µs at 50 ns per Keccak-f; counting each round's 482 further squeeze blocks in sequence too would give 9,650
(482.5 µs). The band's and dense's `call_ns` is 241,500 ns, so in the ideal-permutation model `D = 5` covers
`Δ + RTT < 6 × 241.5 µs = 1.449` ms. At the reference Feistel-SHAKE256 `Π₂` this does not hold: a prover answers
without completing a call, rebuilding a label from 1.53% of `C` in 481 Keccak-f ("Main caveat").

- Block `v` costs `min(v, d) + 1` calls: 6,578 per segment at `d = 12`, 20 times fewer than dense's 131,328. Decoding
  is `d + 1 = 13` rounds deep, and a segment decodes in 24.6 s on one core in Python (dense: 493.9 s).
- **Deployment condition, the cliff.** The effective `D`, the sequential `Π₂` calls an adversary completes within
  `Δ + RTT`, must not exceed `d`. With `d ≥ D′` the band is exactly as hard as dense at `D′` (`band_chain_hard`). At
  `D′ = d + 1` the gap attack (`band_gap_attack`, `band_cliff_d12`) holds every block but each `(d + 1)`-th: every
  dropped row has all its parents held, and the audit accepts with probability 1. At 241.5 µs per call, `d = 12`
  holds while `Δ + RTT < 13 × 241.5 µs ≈ 3.14` ms, in the ideal-permutation model (at the reference `Π₂` a prover
  answers in 24 µs from 1.53% of `C`, inside every deployed deadline: "Main caveat"). `audit.UnsignedRounds` refuses
  a deadline past it (`band.check_deployment`), and `test_gap_attack_at_the_cliff` replays the attack on real labels.
- `band.codec_for(d)` gives the codec for any in-degree; with `d ≥ B − 1` the band is the complete DAG, dense.

## The dense scheme (selectable)

`dense.POINT_64K` has `B = 512` blocks of `ℓ = 65,536` B (524,288 bits) per segment, so a segment is 32 MiB: the
point `dense_meets_64` certifies. P3 keeps 65,560 B, the multiple of `n = 220` it needs. The audit runs with
`k = 111` challenges and a certificate for `D = 5` calls of `Π₂`, each at least 241.5 µs. The 111 is what
`Pous.SecurityProofs.ChainDenseMeets64` certifies for this overwrite-chain `H` over an ideal `Π₂` (not the reference
one: "Main caveat"), for one segment, with no hypothesis; a
random-oracle `H` needs only 106 (`dense_meets_64`). Segment `s` is exactly
`chainDense B ℓ 512 (dom tag_s)`, where `tag_s = salt || u64(s)` plays the role of Lean's 256-bit salt:

~~~text
dom(tag_s, i)  = tag_s (32 B) || u256(i) (32 B)                  Lean `dom`
h_0            = 0^ℓ || dom(tag_s, i)            (ℓ + 64 B)      Lean `iv`: the domain is the capacity, no call
for B in (C_{i-1}, ..., C_0, pad):  (r, h) <- Pi2(B || h)        Lean `ovStep`, `absorbed` (newest first, then pad)
    r = the first ℓ B (dropped), h = the last ℓ + 64 B (chained)
key_i          = r of the pad call                              Lean `ovKey`
C_i            = key_i xor W_i                                  Lean `labelOv`
pad            = 0x01 || 0..0 || 0x80            (ℓ B)          Lean `padBlk`: the first and last bit
Pi2            = Feistel10 on 2ℓ + 64 B, F_i(x) = SHAKE256(0x48 || u8(i) || x)[0 : ℓ + 32]
~~~

- There is no `P` and no transpose. Block `i` costs `i + 1` calls, so a segment costs `B(B + 1)/2` (131,328), about 8
  minutes to encode in Python.
- Decoding recomputes every key from `C`: `W_i = C_i xor key_i`, the rows in parallel, `B` rounds deep (Lean
  `chainDense.dec`).
- `vk`, setup and the audit are the ones below. The leaf index `v` is `u16`, so `B ≤ 65,536`.

## P3 (research)

`W` is `N` segments of `n` lower inputs, each a label of `m = n · wc` bits (`n` components of `wc` bits). For segment
`s`, with `tag_s = salt || u64(s)` and the band graph `B(n, k)` (the `max(k, 1)` nodes before `v`):

~~~text
c1_v = P_(tag_s,0,v)(W_v xor H(tag_s, 0, v, c1_par(v)))
c2_v = P_(tag_s,1,v)(transpose(c1)_v xor H(tag_s, 1, v, c2_par(v)))        transpose(C)_v[u] = C_u[v]
~~~

`C` is every segment's `c2`: `B = N · n` blocks of `m` bits, block `s · n + v`. Decoding reverses both layers. Each
label is `P⁻¹` beside its key, and the transpose sits between the layers (Lean `decodeW`). Segments decode
independently.

The operating point is `params.OPERATING_POINT`: `n = 220`, `k = 12`, `wc = 2384` bits, so `m = 65,560` B and a segment
is 14.4 MB. `params.LEAN_P3_MEETS` (`wc = 37`) is the instance Lean's `p3_meets` is about.

P3 has no certificate. `Pous.SecurityProofs.P3ConcreteMeets64` (one segment), `P3ConcreteMeets14GB` (971 segments) and
`P3ConcreteMeets64D12` (`B(220, 16)` at `D = 12`) are proved, with no hypothesis, for P3 over the overwrite chain in
the ideal two-permutation model, at the operating point's parameters, `D = 10`, `Q = 2^20` and `k = 106`. But Lean's
chain starts from `iv(domP3 tag L v)`: zeros, then `tag_s ‖ u256(2v + L)` in the 512-bit capacity part. The code's
`h_0` is the instantiation's 48-byte header ("Backends") in the low bytes, with the capacity part zero. The guarantees
are about Lean's `H`, so they do not certify the code's (`DISCREPANCIES.md` P3 item 16).

## Lifecycle

1. **Commit.** The server commits to `W` before any salt exists (`setup.commit`: tagged SHA-512 over the parameters and
   `W`, tag `verity-pous/w-commitment/v2`).
2. **Salt.** The verifier draws the 24-byte salt after the commitment (`setup.draw_salt`, through `verity.ml.randomness`
   from its own secret or a later beacon round).
3. **Encode.** The server computes `C* = Enc(W, salt)` and sends it segment by segment. The verifier builds `vk` by
   `setup.verifier_setup`, one of two ways.
   - **Public encoder, the default** (`setup.PUBLIC_ENCODER = True`; POUS `docs/public-encoder/protocol.md`, milestone
     1). `setup.verifier_key_from_encoding` checks each segment's size and block count against the verifier's own
     `params`, decodes it (`params.codec.decode_segment`), streams the result into `commit`'s framing byte for byte,
     and keeps only the segment's leaf digests. It accepts iff the digest equals the commitment, and then builds `vk`
     from those leaves alone. Any mismatch, in any byte, block count or segment count, raises `ValueError`: there is
     no `vk` and no audit. It calls no encoder, reads no root or leaf from the server, and holds one segment at a
     time, never `W`. With the band codec (the default), the decode check costs 6,578 `Π₂` calls, 13 deep, per 32 MiB
     segment: 24.6 s on one core in Python and 6.8 s on four processes, against 493.9 s and 121.9 s for dense.
   - **Private encoder** (`public_encoder=False`). The verifier encodes `W` itself (`setup.verifier_key`). Comparing
     roots (`setup.check_server_root`) catches a wrong server encoding at setup.

   Lean proves that acceptance forces `C* = Enc(W, salt)` for both chain codecs: `chainDense_encDec` (POUS
   `lean/submissions/public-encoder/`) and `chainBand_encDec` (`band-chain/BandEncDec.lean`, through the DAG-generic
   `chainScheme_encDec`, pending red-team §40). It also proves that the public game is equivalent to the private one
   for any scheme with that property (`pubAuditSecure_exact_iff`, `pubMeets_exact`), so the certificates carry over
   at `k = 111`:
   - the band: `Pous.SecurityProofs.BandPubMeets64`, `BandPubMeets14` and `BandPubMeets14Global`, for every `d ≥ 5`,
     and `BandPubMeetsFamily` at every segment count from 1 to `2^64`, with no hypothesis, under the deployment
     conditions in their docstrings, and `BandPubMeetsFamilyD2` and `BandPubMeetsFamilyD1` the same family at
     `D = 2`, `k = 97` and at `D = 1`, `k = 93`;
   - dense: `dense_pub_meets_64` and `seg_pub_meets_14` (a random-oracle `H`), and `chain_pub_meets_64` (the overwrite
     chain), given `ChainExPostFacto`, which `chainExPostFacto_v6` proves (`Security/Proofs/Pous/Chain/Flat.lean`;
     not a guarantee).

   An accept-all check provably fails (`acceptAll_fails_dense`). P3 goes through the same code, but its
   encode-after-decode lemma is not proved.

   The encoder is deterministic and public, so the server running it changes nothing in the timed game. A
   server-chosen or predictable salt would break the model's "`W` before coins" order.
4. **Serve.** `layout.decode_tensor` decodes one tensor from the segments it overlaps.
5. **Audit.** The memory challenger reveals `k` indices one at a time (`audit.UnsignedRounds`, uniform with replacement,
   keyed by `audit.challenge_key` and bound to `vk`'s root). The challenge key is the memory challenger's own secret,
   kept secret until the audit starts (never a public beacon): a public key makes every challenged index computable in
   advance, so a prover holding little of `C` could rebuild only the challenged blocks within the deadline and pass.
   Each block must arrive within `Δ = 0.5 ms` plus
   `rtt_allowance_ns`, and must match `vk`. The audit accepts only if every answer is correct and on time. At 300 µs
   about 1.5% of honest 132-answer audits fail on the host clock (POUS `docs/p3-gpu-measurements.md`). The network
   allowance defaults to 0 until the memory challenger's RTT is measured.

   On the deployed path the memory challenger (`memory_challenger`, `challenger_party`) runs the session in this
   order. Nothing that commits to a coin no round has revealed leaves the device before the last answer is in: the
   server's commitment, `W` and `C*` under the setup's published root, is in before the first round, and the coin root
   comes after the last.

   | Step | From, to | What |
   |---|---|---|
   | draw | challenger | the session id, claimed in its `SessionLog`; the `k` indices from its own secret, committed in its coin tree (root and hm96 key), all kept on the device |
   | warm-up | challenger, server | `WARMUPS` frames on the public `WARMUP_INDEX`, untimed and unrecorded |
   | round `j` | challenger, server | index `j` alone (`u64be`); the send time is read |
   | answer `j` | server, challenger | the whole block; the receipt time is read before it is hashed; the first late answer ends the session |
   | end | challenger, server | `u64be 2^64 − 1` |
   | sign | challenger | one LMS signature over the header (the `SessionCommitment`'s fields and how the session ended) and the round records' root |
   | release | challenger, verifier | the `SessionCommitment` (the coin tree's root and key, the session id and parameters, the setup's root), the signed session and every revealed coin's opening; `commitment` and `openings()` refuse before the signature |

   The verifier checks each round's index against its opening under the released root (`verdict`). The challenger is
   bound to its coins by being an auditor-trusted device that runs its stated code (`pous-ring-device`,
   `trusted-host`), not by an early commitment, so no guarantee reads the coin tree's hiding.

## Public outputs

Under the proof system, an
audit makes public its verdict and nothing about a single round:

- the verdict;
- the salted roots of `W` and of the encoding `C`;
- the salt;
- the setup draw and its profile;
- the memory challenger's session commitment (its coin tree's root and key) and the openings of its coin tree, both
  released once the session is signed.

Everything else is witness. The per-round on-time bits stay in the memory challenger's signed record, so a server can't
steer what the auditor sees round by round. The memory challenger reveals each index from its coin tree, receives the
whole 2,056-byte answer, hashes those bytes itself, and signs the index, the send and receipt times and the hash. The
hash is the memory challenger's, not a commitment the server made, so the audit needs no possession claim. The
reference here runs the rounds in the clear: `audit.AuditResult` keeps every answer and its latency, which is the
memory challenger's record, never a public output.

## The generic protocol (`protocol`, `schemes`)

The lifecycle above is written once, in `protocol`, for any `protocol.Scheme`.

- **What a scheme supplies.** It implements `encode_segment(salt, s, w_seg)` and `decode_segment(salt, s, blocks)`, and
  overrides `respond` only if an honest answer is more than the stored block. It also sets `name` (its byte-format
  string), `backend` (its reference primitives) and `certificate`. As with `verity.ml.commitments.scheme.CommitmentScheme`,
  a scheme is named and versioned, looked up by name, and has conformance vectors pinned under its name.
- **The registry.** `schemes.get(name, segments)` returns a registered scheme at its own parameters, or at
  `params=` of the same name. Each scheme is one module, `schemes/<name>.py`:
  - `band-chain/d12/v1`, the default;
  - `dense-chain/v1`;
  - `p3/v1`;
  - `p2-16448/v4` and `p2-16448/v2`, **experimental** (one module, two key expanders).
- **`protocol.Certificate`** is what an audit at the scheme's parameters establishes, Lean's
  `Meets S .sequential D Q k ε`. It holds:
  - `k`, `rounds` (D), `work` (Q), `call_ns`, `rho`, `delta`, `eps_log2`;
  - `cliff`: the most rounds the structure tolerates at any `k`;
  - `segments`: how many segments of `W` the theorem is about;
  - `theorem`, and its open `hypotheses`;
  - `claims`: ids in `verity.core.claims`' format.

  Each `theorem` is the recorded proof `Pous.SecurityProofs.X` it rests on, of the guarantee `Pous.Guarantees.X`. At
  the frozen point, the band's certificate is `Pous.SecurityProofs.BandMultiMeetsFamily`, with no hypothesis: k 111,
  D 5, Q 2^20, 241.5 µs, cliff at `d`, for every segment count N from 1 to `2^64` (all that `BandParams` allows), so no
  segment count is a gap. It is one `Meets` for the whole
  store: challenges are uniform over all its blocks, the state bound is the whole store's, and one permutation serves
  every segment. A 70 GiB store, 2,240 segments, is certified whole. At one segment it is also `BandChainMeets64`, and
  at 448 (14 GB) `BandMultiMeets14`.

  Dense is `Pous.SecurityProofs.ChainDenseMeets64`, with no hypothesis and no cliff, for one segment; other segment
  counts are gaps in the storage profile. P2 (`schemes/p2.py`) is **experimental**:
  - It is the squaring-based scheme: two big-integer squarings per 2,056-byte block to decode.
  - It follows the P2 spec (`p2/PROTOCOL.md`, v4: the POUS store's `docs/efficient-crypto/p2-spec-draft.md`, frozen
    v1, with SHA-512 for every hash) in its maps and values:
    - the payload is 2,054 bytes per block, which is the spec's profile `p2-2054B`, and the manifest names it;
    - every block, payload and key is a fixed-width, unsigned, big-endian integer;
    - the tweak is rejection-sampled below p, never reduced.
  - Its keys are fresh per block and expanded where they are used, so nothing key-sized is stored, which keeps the
    1.05× space bound. Two codecs differ only in the expander:
    - `p2-16448/v4`: the spec's §4 counter-mode SHA-512 streams with its v4 framing (`frame` is
      `verity.ml.randomness.frame`), one for K and one for t's candidates, per `(segment, block)`, each 64-byte
      chunk one compression after the zero-padded prefix's midstate. This is the expander the spec's setup assumption
      P2-EXP-IO names (decision F4). The keys are the spec's vectors' given the spec's prefix; the prefix frames the
      protocol's 24-byte salt (derived from the commitment to W by `setup.draw_salt`) where the spec frames the model
      commitment and a 32-byte setup salt. That goes with the rest of the spec's setup framing (its leaves and
      challenge derivation), which the generic protocol does not yet follow;
    - `p2-16448/v2`: ChaCha8 keyed by a 32-byte seed per segment (`SHAKE256("verity/pous/p2/seed/v1" ‖ tag_s)`), with
      nonce = the block index. It is **nonconforming** to F4, and its `experimental` reason says so. Its keys cost
      about 1% of the device's decode work.
  - Both certificates are the pinned `Pous.SecurityProofs.P2SlackFamilyUncond` at the spec's k = 105 (decision F6):
    its `λ = 128` row, for every block count from 2^8 to 2^23 and every payload width in range, `p2-2054B`'s among
    them, with no hypothesis; the certified store is 2^19 blocks (8 segments of 2^16). It is an abstract-M1 result, and
    the SMS-to-M1 instantiation assumption is not accepted (the spec's decision 8).
  - Above 2^23 blocks the run protocol takes `p2.wide_certificate`, the per-period guarantee
    `Pous.Period.SecurityProofs.periodHeldAt`: every block count from 2^19 to 2^27 at the registered width and prime,
    the same `⌊ρ·blocks·w⌋` at `1% + 2^-128`, so Demo B's 1038 segments of 2^16 blocks certify 1,060,008,348,186 bits
    (`PeriodHeldDemo`). It is in the ring model and rests on the open conjecture `P2RingRoundSecureWide`, the device
    model `RingDeviceAt` and the coin commitments' hiding at that count (`p2.WIDE_CLAIMS`).
  - SeqRoot sets `call_ns` (`ROOT_CALL_NS` = 854,029 ns), half the fastest 16,448-bit root with 8 cores of one chiplet
    cooperating, which the spec allows. No root may fit within twice `Δ + RTT`. v1's 1 ms RTT allowance
    (`Δ + RTT = 1.5 ms`) kept only 1.14× of that margin, so spec v1.1 (`p2/PROTOCOL.md` §12) keeps w = 16,448 and caps the RTT allowance at `RTT_ALLOWANCE_CAP_NS` = 354 µs, so
    `Δ + RTT ≤ 854 µs`.
    - The single-core floor is the timing red team's Zen 5 core at 5.7 GHz with full-rate IFMA and its best
      Toom-4/Karatsuba plan: τ_F = 361.55 ns per squaring (`regate.py` at w = 16,448).
    - A root is 16,446 squarings plus about 20 multiplications (40 squaring-equivalents), so
      16,486 × 361.55 ns = 5,960.5 µs. The red team's 5.97 ms rounds τ_F to 362 ns first.
    - The cooperation speedup is the measured 8-core one, 2,716.7 / 778.5 = 3.490× per squaring on one Zen 4 chiplet.
    - The cooperating root is 5,960.5 / 3.490 = 1,708.1 µs, and half of it is 854,029.18 ns, rounded down to
      `ROOT_CALL_NS`. The cap is the largest whole µs with `Δ + cap < ROOT_CALL_NS`: 500 + 354 = 854 µs. The
      memory challenger accepts it and refuses 355 µs and v1's 1 ms, and the margin at the cap is 2.0001×.
    - **Caveats** (§12): the 3.490× was measured on an untuned Zen 4 kernel and is applied to a modelled floor. The red
      team's sync model (t_sync = 20–40 ns per squaring, `coop.log`) gives 4.24× and 5.55× at 8 cores, and under it
      the margin at the cap is 1.64× and 1.26×. The cap rests on the measured speedup.
    - **Sources:** the timing red team's scripts and logs (`regate.py`, `coop.log`), and the P2 assumption review,
      which found v1's cap unsafe.
  - `Scheme.experimental` carries the remaining reasons: the SMS-to-M1 instantiation is not accepted, and the
    XOR-then-reduce compression attack on it is open. The memory challenger audits an experimental scheme only with
    `experimental=True`, and its profile's statement begins with it. No experimental scheme is ever the default.

  P3 has no certificate: the concrete-`H` guarantees have its parameters, but their chain starts from a different `h_0`
  ("P3 (research)").
- **`protocol.TimedAudit`** is `audit.UnsignedRounds` at the certificate's `k`. It refuses:
  - a `k` below the certificate's;
  - a deadline whose effective D (calls of `call_ns` within `Δ + RTT`) exceeds the certificate's `rounds`: for the
    band and dense, `Δ + RTT` must stay under 1.449 ms (a model-level bound: at the reference `Π₂` a prover answers
    from 1.53% of `C` well inside it, "Main caveat"). `UnsignedRounds` alone refuses only past the band's cliff, at
    about 3.14 ms (3,139,500 ns at `d = 12`);
  - a scheme with no certificate at its parameters.

  A scheme may have several certificates (`Scheme.certificates`), and the memory challenger audits under the first
  that covers its `k` and deadline. The band's are `BandMultiMeetsFamily` (`k = 111` at `D = 5`, the default), then
  `BandMultiMeetsFamilyD2` (`k = 97` at `D = 2`, `Δ + RTT` under 724.5 µs, `d ≥ 2`), `BandMultiMeetsFamilyD1`
  (`k = 93` at `D = 1`, under 483 µs, `d ≥ 1`) and `BandMultiMeetsFamilyD0` (`k = 89` at `D = 0`, under 241.5 µs, every
  `d`). An unset `k` stays at the first certificate's, so the default audit is unchanged; the three smaller `k`s are
  used only when asked for. A band with `d < 5` has only the certificates with `D ≤ d`.

  `research=True` runs a refused audit anyway, except past the cliff, with the first certificate's refusals as gaps.
- **`protocol.StorageProfile`** is what an accepted audit establishes: a server whose state held fewer than
  `⌊rho · |C|⌋` bits passes with probability at most `delta + 2^eps_log2`, by the certificate. It also lists the
  `gaps`, where the certificate does not cover the audit: a refusal run with `research=True`, or a segment count of `W`
  other than the theorem's. The band's certificate covers every segment count from 1 to 2^64, and dense's one segment.
  `to_json` writes `verity-pous/storage-profile/v1`.
- **The rest:**
  - `protocol.verifier_key(scheme, commitment, salt, mode=PRIVATE or PUBLIC, w=..., encoded=...)` builds `vk`
    either way;
  - `commit`, `draw_salt`, `encode`, `decode`, `decode_tensor`, `challenge_key` and `responder` wrap the modules
    below;
  - every coin comes from `verity.ml.randomness` through `setup` and `audit`, and a scheme draws none.
- **Segment tags must be injective across all segments.** This is a hard requirement of every implementation:
  segment `s` is keyed by `tag_s = salt ‖ u64 s` (`labels.tag`), or by one salt with global indices.
  - With a shared domain, identical segments get identical labels. An adversary storing one of them then answers every
    challenge, and the audit accepts with probability 1: Lean's `shared_domain_breaks` makes `Meets` false for every D,
    Q, k and ε.
  - `segDom_inj` is the certificates' hypothesis.
  - `tests/test_pous_protocol.py` checks it for every scheme, and replays the attack on real labels.
  - `benchmarks/pous` gates every implementation's tags against `salt ‖ u64 s` over 448 segments.
- **Claims.** `protocol.claim(id)` resolves an id from `verity.core.claims`, or from `protocol.LOCAL_CLAIMS` until core
  holds it: `ideal-permutation`, `perfect-isolation` (the help-accounted row: "The problem"), `sequential-step-floor`,
  `feistel-indifferentiability` (refuted at the reference `Π₂`: "Main caveat"), and P2's `m1-ideal-permutations`,
  `sms-m1-instantiation`, `seq-root`.
- **Tests.** `tests/test_pous_protocol.py` runs one conformance suite over every scheme, plus a toy that implements
  only the two methods. Each registered scheme's vectors are pinned in `tests/fixtures/schemes/<name>.json`, written
  by `tests/pous_scheme_vectors.py`: the certificate at the registered point, and a small instance's segment digests
  and `vk` root. `VERITY_POUS_SLOW=1` also encodes a whole band12 segment through the interface against
  `band/64kb/segment`.

## The continuous audit's memory challenger (`continuous`)

`continuous.ContinuousMemoryChallenger(vk, key, n, late_deadline_ns=Δ_late, max_misses=f₀)` is the memory challenger
of the continuous audit's model, `Security/Definitions/Pous/Model/Continuous.lean`, with the guarantee
`Pous.SecurityProofs.ContinuousAudit`: `n` challenges over time, drawn as the timed audit's indices, accepted
when at most `f₀` answers miss. A miss is an answer that is wrong or later than `Δ + RTT`. `n` is fixed before the
stream, so the stream is never stopped or extended on misses, though it may reject early (`exceeded`). It is one
`audit.UnsignedRounds` over the stream with a hard deadline, and it enforces the three requirements that
`Security/Proofs/Pous/Continuous/Verifier.lean` proves of the model's memory challenger:

1. **No pipelining.** `challenge()` reveals index `j + 1` only once answer `j` is in or `Δ_late + RTT` has passed since
   index `j` was revealed, and in that case it first closes challenge `j` as rejected; the close stands even when `j`
   was the last and `challenge()` then refuses (`run_continuous` closes it with `expire()`). Each deadline runs from the
   memory challenger's own clock reading at the reveal; a latency the responder reports (`ChallengeReply.report`) is
   never read. An answer names its challenge (`respond(j, block)`), and an answer to any challenge but the open one is
   refused, whether that challenge is closed or not yet revealed. So a stale answer is never credited to the next
   challenge, even when both ask for the same block.
2. **A hard deadline.** An answer past `Δ_late + RTT` is rejected without its block being checked
   (`ChallengeReply.rejected`). So is a challenge still unanswered by then, closed by `expire()` or by the next
   `challenge()`. `UnsignedRounds` refuses a `Δ_late` below `Δ`, and one whose `D_late` exceeds `audit.LATE_ROUNDS = 12`
   or the band's `d`, where
   `D_late` is the number of 241.5 µs calls that fit in `Δ_late + RTT`; at `d = 12`, `Δ_late + RTT` must be under
   3,139,500 ns. That cap is the graded audit's; `ContinuousAudit` holds at every `D_late`, so for it the cap is only
   conservative. `ContinuousMemoryChallenger` refuses to run without a hard deadline.
3. **Every late answer counted.** Every challenge closes with exactly one `ChallengeReply`, whether it was answered,
   rejected or expired. `AuditResult.misses` counts every wrong or late answer, and `AuditResult.late` every answer
   past `Δ + RTT`, rejected ones included.

Run the continuous audit only on `ContinuousMemoryChallenger`, never on a bare `UnsignedRounds` with `late_deadline_ns`. Its
`respond(block)` names no challenge, so once challenge `j` expires, an answer to `j` still in flight is credited to
`j + 1`, a hit when both ask for the same block. That block was on the wire, not in the memory the model measures at
the send time, so the audit would credit storage the responder does not hold, as the red team found.

**The audit's duration cap.** `audit.AUDIT_CAP = 1,505` (`Pous.Rebuild.auditCap`) is the cap on a graded audit's
window that `Pous.SecurityProofs.BackgroundRebuildExcluded` requires the memory challenger to enforce.
`UnsignedRounds(..., audit_cap=...)` and `ContinuousMemoryChallenger(..., audit_cap=...)` enforce it on the memory
challenger's clock, counted in calls of the parameters' `call_ns` as `D` and `D_late` are, so the limit is 1,505 × 241.5
µs ≈ 363.5 ms.
- An answer received more than `audit_cap · call_ns` after the first challenge's send is rejected unchecked and aborts
  the audit (`ends_at`). So does a challenge closed after that time by `expire()` or the next `challenge()`.
- `closes_at` is never past `ends_at`, so `run_continuous` closes the open challenge there.
- An aborted audit (`aborted`, `AuditResult.aborted`) never accepts, whatever its misses, and reveals no other index.
  It is rejected and never retried with fresh challenges, since a retry would let a prover that stalls near its
  deadlines escape a failing audit.
- A cap from 1 to 1,505 is accepted.

The cap is off by default. The continuous audit's guarantee measures the memory at each send time and needs no cap,
and its model's memory challenger has none. Under the cap, a stream whose arrivals spread over more than 363.5 ms always
aborts.

`run_continuous(challenger, send, receive, wait=...)` drives the stream over a transport. `wait(j)` returns at arrival
`j`, `send(j, i)` sends challenge `j`, and `receive(until)` returns the next `Reply(j, block, path, report)`, or None
once the memory challenger's clock is past `until`, the open challenge's `closes_at` (its send time plus
`Δ_late + RTT`). A reply naming any challenge but the open one is dropped. Once the hard deadline has passed, the driver
closes the open challenge with `expire()`, the last one too, so a silent or slow responder cannot hold the stream open.
A transport that returns None before the deadline makes `expire()` refuse, and the audit stops with the challenge open.
The caller chooses the arrival times and `n`, and nothing here draws or checks them. The game needs both to be the
memory challenger's own randomness, independent of the setup, the indices and the answers. Without `late_deadline_ns`,
`UnsignedRounds` is unchanged, the single audit's memory challenger. `tests/test_pous_continuous.py` tests each
requirement.

## The streamed-label wipe (experimental, `streamed_wipe`)

A proof of secure erasure for a memory wipe, in place of the permutation-based retention
for wipes: C8, the ideal-permutation premise, is false at the reference Feistel10-SHAKE256 `Π₂`, where 1.53% of the fill
rebuilds any label ("Main caveat"). Here the labels need no permutation and no graph. They are the
auditor's own PRF outputs, so a label the server didn't keep is a label it can't answer, and a server that kept its
prior content through the wipe had no room for the labels. The reference is `streamed_wipe.py`, its tests
`tests/test_pous_streamed_wipe.py`, its verdict's Lean port `MemoryChallenger.StreamedWipe`
(`verity/core/devices/memory_challenger/lean/`), its timing `benchmarks/pous/streamed_wipe/`, and its guarantees
`Pous.Guarantees.StreamedWipe` and `StreamedWipeMeets2600GB` (`Security/Definitions/Pous/Guarantees/StreamedWipe.lean`, the
game in `Security/Definitions/Pous/Model/StreamedWipe.lean`), both proved (`Security/Proofs/Pous/StreamedWipe/`), where
`PousStreamedWipe.honest_accepts` shows the server that keeps every label accepted on every run.

**Labels and coins.** The memory challenger, the auditor's device on a machine the prover doesn't control, derives every
label itself, under a key fresh for each server and each wipe:

~~~text
K     = derive(secret, "verity/pous/streamed-wipe-labels/v1", {server, session})        32 bytes
L_i   = SHA-512("verity-pous/streamed-wipe/label/v1\0" ‖ K ‖ u64be i)                   64 bytes, i in [0, N)
c_j   = derive(coin_secret, "verity/pous/streamed-wipe-coins/v1", {server, session, n}).uniform(N, j),  j in [0, k)
~~~

A label's message is 75 bytes, one SHA-512 compression, so a GPU derives labels faster than any link carries them.
SHA-512 is the PRF (C3, claim `prf/sha-512`). `coin_secret` is drawn from the OS once the fill is complete, and the
indices are uniform and independent, with replacement: live coins, never Fiat–Shamir.

**Messages.** One session per server and wipe, under an id the auditor designates (`session`, 32 bytes):

| Step | From, to | What |
|---|---|---|
| open | challenger, public | the `WipeCommitment` header: the protocol, the server, the session, `N`, the label size, `k`, the batch size, the deadline a batch (`limit_ns`), and every memory tier inside the boundary with its size, the sizes adding up to `N · 64` bytes; the id goes into the challenger's `SessionLog` first, which refuses it again |
| fill | challenger, server | `L_0 … L_{N−1}` in index order, a chunk at a time; the challenger reads its clock at the first and last chunk |
| challenge | challenger, server | once the fill is complete, the coins are drawn, then batch `t`'s indices, `c_{tB} … c_{tB+B−1}`; the send time is read |
| answer | server, challenger | 64 bytes an index, in order; the receipt time is read before anything is hashed, then each answer is compared with `L_{c_j}` |
| egress | network certifier, challenger | the bits that crossed the boundary over the session (C6), as the certifier counts them |
| sign | challenger | one LMS signature (`memory_challenger.SCHEMES`, n = 64) over `SHA-512("verity-pous/streamed-wipe/session/v1\0" ‖ u64 len ‖ header ‖ records root)`: the header is the commitment, how the session ended, the fill's times, the batch count and the egress; the records root is the frame-v3-sha512 root of the batch records, each `u32 b ‖ u32 first ‖ u32 count ‖ u64 send_ns ‖ u64 receipt_ns ‖ u32 wrong ‖ SHA-512(indices ‖ answers)` |

The first late or wrong batch ends the session (`late`, `wrong`). No batch is retried and no index is redrawn, so a
failed wipe is a visible failure. `verdict(signed, commitment, trusted=...)` accepts a session only when it is
signed under a trusted key and its header is the published commitment, its batch size is 1 (the reveal the guarantee
covers), every batch is present in order, none has a wrong answer, each is within `limit_ns` on the challenger's clock,
and the first was sent after the fill ended.

The verdict a caller records is the Lean's: `MemoryChallenger.StreamedWipe.verdict`, the executable
`streamed-wipe-verdict`, which `wipe_verdict.judge` (beside the Lean package) sends the signed session, the commitment,
the trusted keys and the accepted schemes' ids as JSON (`to_json`), and which refuses a record `to_json` can't have
written. `wipe_verdict.VERDICT` is the switch: `lean` (the Lean alone), `both` (the Python's beside it, and a session
the two differ on in any field rejected as a disagreement) or `python` (no Lean).
`verity/core/devices/memory_challenger/lean/tests/test_wipe_verdict.py` holds the Lean to `streamed_wipe.verdict` on drawn sessions at every batch size, one adversarial edit of them (a wrong
label, a late or early answer, an untrusted signer, a wrong count, a batch dropped or added, a field past its width),
JSON the Lean must refuse, and named negative controls.

**Batches and deadlines.** A batch's deadline runs from its send to its receipt on the challenger's monotonic clock, at
a resolution of at most `memory_challenger.CLOCK_RESOLUTION_NS`. It is to be set below the challenger's round trip to
the server plus the fastest fetch from outside the boundary, so that an answer fetched from a store outside, counted or
not, is late. The bound doesn't read the deadline: in the game, without `K`, nothing computes a label at any speed, and
what the outside holds is charged as egress. The deadline is what keeps C5 and C6 checkable during the challenges.
The guarantee covers the sequential reveal, batch size 1 (`WipeParams.covered`), where each answer reads only the
indices before it. Batch sizes above 1 are implemented and timed, but not covered (below), so the verdict, the Lean's
and the Python's alike, rejects such a session however it ran (`batch size B is uncovered: …`).

**What is public.** The commitment (protocol structure, with the server's declared tiers: the model's size), the
verdict and the certified egress in whole MiB, rounded up and saturating at `2^15 − 1` (`Verdict.public()`, the
protocol's outputs, 16 bits in all), and the coins once the session is signed (`Challenger.coins()`, the verifier's draws). The
signed session (its times, its batch records and the exact egress) is witness, as a memory-challenger session is: the
prover proves `verdict` of it through the proof system, and only the verdict is published. The labels and `K` are the
auditor's; nothing of the prover's is committed, opened or read.

**The bound.** For every prover of the sequential wipe, with `M` bits inside the boundary when the fill ends (every
tier), `E` bits of egress, `q` bits of uniform prior content `z`, and every `g`:

~~~text
Pr[accepted ∧ z survives] ≤ (g/N)^k + k · C(N, g+1) · 2^(M+E) / 2^(q+(g+1)·b) + Adv_PRF
~~~

where `b = 512`, and `Adv_PRF` is SHA-512's advantage, as a PRF, against the distinguisher that runs the wipe with this
prover. With `ρ = g/N`, the first term is `ρ^k`: a prover that can answer at most `ρ·N` labels after every history
passes with probability at most `ρ^k` (sampling). The second is the guessing term: answering more than `g` labels while
keeping `z` means `M + E` bits held `z` and `g + 1` labels, which happens for a `2^(M+E−q−(g+1)·b)` fraction of
`(z, L)`, times the `C(N, g+1)` sets and `k` rounds (compression). With `M = N·b`, an accepted wipe keeps at most about
`(N − g − 1)·b + E + log₂ C(N, g+1) + log₂ k + log₂(1/δ)` bits of the prior content: the labels it didn't hold, the
egress and a counting term. `z` survives when a function of what is held inside and outside returns it, so the bound
covers prior content moved out of the boundary too.

At the deployed fill, `N = 40,625,000,000` labels (2.6 TB) at `δ = 2^-40`, half of it for sampling:

- `k = 72,074` is the fewest challenges for which a server short of 1 GB of labels passes with probability at most
  `2^-40` (`challenges_for`, sampling alone);
- at `k = 74,000` (`DEPLOYED_K`), with `g = N − 15,598,671`, an accepted wipe keeps at most `E + 8,186,012,770` bits of
  the prior content, `E` plus 1.023 GB (`retained_bits`, with the exact `C(N, g+1)`);
- `StreamedWipeMeets2600GB` proves `E + 8,600,000,000` bits (`E` plus 1.075 GB) at `g = N − 15,610,000`, where the
  count `C(N, r) ≤ N^r ≤ 2^(36·r)` spends the difference: `k · (D/N) = 28.434 > 41 · ln 2` for sampling, and
  `17 + 548 · (D − 1) + 41 = 8,554,279,510` bits for compression, each `2^-41`.

At `k = 2152` reads and `δ = 2^-128`, `StreamedWipeMeetsFamily`
(`Security/Definitions/Pous/Guarantees/StreamedWipeFamily.lean`, with its instances `StreamedWipeMeets96GiB` and
`StreamedWipeMeets141GB`, proved in `Security/Proofs/Pous/StreamedWipe/Family.lean`) proves `⌈M/19⌉ + E` bits for every
fill of `N ≥ 2^19` labels (one 32 MiB segment or more, no upper limit) at `g = N − ⌈N/24⌉`: `2152/24 > 129 · ln 2` for
sampling, and with the count `C(N, r) ≤ 2^N` the compression term needs `q ≥ E + 512 · (⌈N/24⌉ − 1) + N + 141`, below
`⌈M/19⌉ + E` since `512/24 + 1 < 512/19` bits a label, each `2^-129`.
`(18/19)^2152 ≈ 2^-167.9`, but sampling can't take all of `f = 1/19`: the missing labels' bits and the count come out of
the same `M/19`. At `k = 2152` this split reaches about `f = 1/23`, and `f = 1/19` from about `k = 1,765`.

For any read count, `StreamedWipeMeetsReads` (`Security/Definitions/Pous/Guarantees/StreamedWipeReads.lean`, proved in
the same `Family.lean`, the family being its instance at `r = 24`) proves `512 · (⌈N/r⌉ − 1) + N + ⌊log₂ k⌋ + 130 + R + E`
bits at `2^-128` for every `k` and `r` with `(r − 1)^k · 2^129 ≤ r^k` (that is `(1 − 1/r)^k ≤ 2^-129`), every fill of
`N ≥ 1` labels and `R` bits of the server outside the fill, which count in full. So `f ≈ 1/r + 1/512` at
`k ≈ 129 · ln 2 · r ≈ 89.4 · r`. Its H200 instance, `StreamedWipeMeetsH200`, meets the onsite target of at most
5 GB a device: at 141 GiB (151,397,597,184 bytes) filled whole, `N = 2,365,587,456` and `R = 0`, the least ratio is
`r = 33` and its least read count `k = 2906`, for `39,068,035,213 + E` bits (about 4.884 GB). At `r = 32` (`k = 2817`)
it is 5.027 GB. NVML's total on the deployed card, 150,754,820,096 bytes, is 613 MiB less, and that card filled whole
is 38,902,167,181 bits at the same `r` and `k`. A fill that leaves memory out
counts it in `R` in full, so the same 5 GB needs a larger `r`: what an accepted session certifies is the threshold at
its own `N` and `R`.

`verity/wipe/v0` publishes this threshold at the session's `k`, which the memory challenger chooses (`wipe_party
--reads`, 2152 by default) and its commitment names, at `r = ratio(k)`, the largest `r` the condition allows, with `R`
every declared tier's bytes past the fill (`wipe_period.certify`, `MemoryChallenger.WipeProtocol.certify`; from 1 to
`MAX_READS = 2^17` reads). One 32 MiB segment at `k = 2152` (`r = 24`) is 11,709,069 bits plus the egress, under the
family's `⌈M/19⌉ = 14,128,182`; the H200 filled whole at `k = 2906` is `StreamedWipeMeetsH200`'s figure.

**What it rests on.** No named assumption. The PRF enters as the advantage term in the conclusion, which C3 bounds for
an efficient prover. The physical premises are the game's shape, and the deployment must make them true:

- *no unaccounted tier*: `M` is every bit the server can hold when the fill ends, every memory tier the commitment
  declares, and nothing else;
- C6, *egress certified over the whole wipe*: `E` is at least the bits that left the boundary from the fill's start to
  the last answer, as the network certifier counts them;
- C5, *no side channel*: no label leaves except through that egress, and nothing reaches the server during the
  challenges but the indices and what the outside sends.

The `+ N` in `StreamedWipeMeetsReads`'s threshold is the count `C(N, m) ≤ 2^N` of the sets of `m = ⌈N/r⌉ − 1`
missing labels, `N/512` bytes, 29.5 GB on a whole 8×H200 server. `StreamedWipeMeetsReadsTight`
(`Security/Definitions/Pous/Guarantees/StreamedWipeServer.lean`, proved in
`Security/Proofs/Pous/StreamedWipe/Server.lean`) replaces it with `m · (⌊log₂⌊3N/m⌋⌋ + 1)` (`C(N, m) ≤ (e·N/m)^m`), so the fill's share is about `(N/r) · (512 + log₂(e·r))`. Per server (the
wipe leaves at most 5 GB of the server's assumed memory capacity), `StreamedWipeMeetsServer` states it against the
assumed capacity `M_s = 512 · N + R` over every tier, `R` being every bit outside the fill (the boot disk and the GPU
contexts included): a server keeps at most `B(N, r, k, R) + E` bits, `B` the fill's share plus `R`.
`StreamedWipeMeetsServer5GB` is it at the 5 GB point `B ≤ 4·10^10`, and the servers that reach it, at the least `r`
and its least `k` (the H200 also at its deployed `k`), are:

| server | fill | `R` | `r` | `k` | fill's share | `B` | statement |
| --- | --- | --- | --- | --- | --- | --- | --- |
| one H200, measured | 148.48 GB | 2.276 GB | 56 | 4,963 | 2.693 GB | 4.969 GB | |
| one H200, deployed | 148.48 GB | 2.276 GB | 62 | 5,500 | 2.432 GB | 4.709 GB | `StreamedWipeMeetsServerH200` |
| 8×H200, HBM | 1,128 GB | 8 × 0.3 + 1.5 GB | 1,050 | 93,843 | 1.0995 GB | 4.9995 GB | `StreamedWipeMeetsServer8H200HBM` |
| 8×H200, HBM and 2 TB RAM | 3,128 GB | 8 × 0.3 + 1.5 GB | 2,922 | 261,229 | 1.0998 GB | 4.9998 GB | `StreamedWipeMeetsServer8H200HBMRAM` |
| 8×H200, and 12 TB disk | 15,128 GB | 8 × 0.3 + 1.5 GB | 14,183 | 1,268,143 | 1.1000 GB | 4.99996 GB | `StreamedWipeMeetsServer8H200` |

The H200 is vy-dc-h200-2's card as r20261026-003454-bfd2 measured it (`Pous.StreamedWipe.h200`): 143,771 MiB in all
and a fill of 4,425 segments of 32 MiB, so `R` is 18,211,667,968 bits; the deployed row is `wipe_party --reads 5500` at
its published `r = 62`. With today's fill-process contexts at zero reserve, `R = 8 × 1.2 GB = 9.6 GB` on its own, and no
`r` reaches 5 GB. `8 × 0.3 GB + 1.5 GB` is a minimal context per GPU and a host OS, unmeasured: the minimal-context
probe in Tuesday's window on vy-dc-h200-2 is to measure it, and the 8×H200 rows' honest lemmas wait on it.

Each of `StreamedWipeMeetsReadsTight`, `StreamedWipeMeetsServer`, `StreamedWipeMeetsServer5GB` and
`StreamedWipeMeetsServerH200` is its claim (`Security/Definitions/Pous/Claims/StreamedWipeServer.lean`) at the wipe's
verdict, with two lemmas beside it (`Security/Proofs/Pous/StreamedWipe/Witnesses.lean`). The honest one
(`Pous.Claims.<name>_meets`): its numeric premises, `M_s = 512·N + R`, `0 < N`, `(r − 1)^k · 2^129 ≤ r^k` and the
5 GB point, hold at the measured H200 at `k = 5500`. The nontrivial one (`Pous.Claims.<name>_fails`): against a verdict
that accepts everything, a prover that writes its prior content into its memory and reads it back is accepted and
keeps it with probability 1, while the PRF's advantage against the distinguisher built from it
(`wipeDistinguisher P`) is 0. A prover that keeps nothing stays within `2^-q` (`Pous.Claims.keepNothing_within`).

**Measured** (`r20261019-061002-ee6b`, one RTX PRO 6000 Blackwell Server Edition with 188 SMs;
`benchmarks/pous/streamed_wipe/run.sh`; every gate passed, 76 vectors against `streamed_wipe.label`):

| | |
|---|---|
| labels on one GPU | 7.05·10⁹ labels/s, 451 GB/s stored (451 GB/s compute alone): the whole 2.6 TB label set in 5.76 s |
| GPU to pinned host | 57.2 GB/s (a copy alone: 57.2 GB/s) |
| the node's NIC | 400 Gb/s (`eth0`; `mlx5_0` 4X NDR): 50 GB/s, so a 2.6 TB fill takes 52.0 s at line rate |
| `ib_write_bw`, HCA loopback, 1 MiB, 4 QPs | 350.7 Gb/s (43.8 GB/s): a fill in 59.4 s at that rate |
| TCP over loopback, Python | 4.5 GB/s on one connection, 28.9 GB/s on 16 |
| 74,000 challenges, loopback TCP, a 16 GiB table | batch 1: 2.60 s (32 µs a round at p50, 72 µs at p99); 16: 255 ms; 256: 103 ms; 4,096: 74.5 ms; 74,000: 71.7 ms (median 75.4 ms over 20) |
| one reference session | a `2^20`-label fill over TCP in 0.61 s, 74,000 sequential rounds in 5.0 s (49 µs at p50), signed and accepted |

So one GPU derives labels for 9.0 servers at line rate, but its PCIe link carries 57 GB/s, 1.14 servers' worth: a
challenger feeding nine servers needs nine host links' worth of PCIe, or labels sent from the GPU to the NIC directly,
which was not measured. The node's load average was about 930 during the run, which inflates the Python timings.

**Holes.**

- **The batched reveal.** With batch size `B > 1`, an answer reads its whole batch, so the set of labels answerable
  after a history is no longer defined index by index. The two direct arguments fail: a witness batch for each answered
  label costs `(B − 1)·log₂ N` bits for each of about `N` labels, far more than a label saves, and a single shared
  witness gives `(g/N)` a batch, `k/B` factors in place of `k`. Covering it needs a direct-product (multi-instance)
  compression bound. Until then the verdict accepts only the sequential wipe: 74,000 rounds took 2.6 s over loopback,
  against a fill of about a minute.
- **The physical premises** are the game's shape, not yet tied to `Definitions.Accounting.Physical`, whose `Memory` and
  `Links` constraints are their form there.
- **The inter-node stream** is not measured: between the nodes only TCP 22 passes. The fill time is the NIC's line rate
  (52 s) and its RDMA loopback (59 s), and the deadline needs the challenger's round trip to a server and the fastest
  fetch from outside the boundary, neither measured.
- **The guarantee isn't stated against the Lean verdict, and there is no Program.** `MemoryChallenger.StreamedWipe.verdict`
  decides the verdict, but the guarantees read the game's `Accepts`, not that function, so they stay frozen (no-function)
  until restated against it. `verdict` has no circuit for the proof system, so it is not yet proved in zero knowledge.

## API

| Module | What it holds |
|---|---|
| `params` | `Params(n, wc, k, segments)`, `OPERATING_POINT`, `LEAN_P3_MEETS` |
| `labels` | the bit layout: Lean's `codeEquiv` (bit `j` of block `i` at `i · width + j`), little-endian bytes; `transpose`; `tag` |
| `oracle` | queries (`HQuery`, `PQuery`/`PinvQuery` with a `Tweak`, `PiQuery`/`PiInvQuery`), programs (`Prog`), `run` with Lean's depth/work/width, `parallel` |
| `primitives` | `KeyDerivation` and `TweakablePermutation` interfaces; `OverwriteChain` (the default `H`), `RandomOracleKey`, and the broken `XorSponge` and `MiyaguchiPreneel`; the backends `reference`, `heuristic_arx`, `ideal` |
| `codec` | `Codec(name, encode_segment, decode_segment)`, the seam every module above the scheme uses; `Geometry`, the sizes they read; `split`, `encode_segments` |
| `band` | `BandParams(blocks_per_segment, m, d, segments)`, `SECURE_FIRST`; `codec_for(d)`, `BAND12`; `reference`, `ideal`; `effective_rounds`, `check_deployment` (the cliff) |
| `dense` | `DenseParams(blocks_per_segment, m, segments)`, `POINT_64K`; `encode_segment` and `decode_segment` on bytes (reading each row's parents from the parameters, so the band shares them); `DENSE`, the two as a codec; `DenseChain` (the chain `H` of both), `reference`, `ideal`; the `*_prog` oracle programs |
| `scheme` | P3's `encode_segment(params, prims, salt, s, w_seg) -> blocks` and `decode_segment(params, prims, salt, s, blocks) -> w_seg` on bytes; `P3`, the two as a codec; `encode` and `decode` over all of `W`, segment by segment; the `*_prog` oracle programs |
| `layout` | `Layout.of(params, [(name, size)], align)`, `Layout.of_ranges(params, [(name, offset, size)])`, `pack`, `decode_tensor(..., codec=params.codec)` |
| `setup` | `commit` (streamed over segments), `draw_salt`; `verifier_setup(params, prims, commitment, salt, encoded=C*, w=None, public_encoder=PUBLIC_ENCODER)`, which picks `verifier_key_from_encoding(params, prims, commitment, salt, C*, codec=params.codec)` (public encoder, the default) or `verifier_key(params, prims, W or its segments, commitment, salt, encoded=None, codec=params.codec)` (the verifier encodes); `server_root`, `check_server_root` |
| `verifier` | `VerifierKey` (`root` and leaf digests; `build`, `from_leaves`, `from_segments`, `from_row_digests`), `leaf_hash` (whole or chunked blocks), `row_leaf` (from a block's row digest), `segment_leaves`, `MerkleTree` (audit paths), `verify_path` |
| `audit` | `challenge_key`, `HonestResponder`, `UnsignedRounds` and `run_timed` (wall clock; `respond` may return `(block, report)`, kept in `Answer.report`), `simulate` (oracle rounds); `k` and the adversary's call time default to the parameters' own (band and dense: 111 and 241.5 µs); `UnsignedRounds` refuses a band deadline past the cliff |
| `memory_challenger` | the memory challenger, which signs its rounds and on the deployed path replaces `audit.UnsignedRounds`: `CoinTree` (its `k` indices from its own secret, committed on the device before the session and released after it; `CoinOpening`, `verify_coin`), `MemoryChallenger(coins, signer, log=, limit_ns=, block_bytes=, binding=)` with its `SessionCommitment` (`commitment`, released once the session is signed; `binding` the setup's 64-byte root of `C*`, required; `session` the auditor's id for the audit, which `log`, a `SessionLog`, records as used before the first challenge and refuses again), `challenge()`, `receive(block)` (one `Round` each: index, send and receipt times, `H(block)`; the first late round ends the session), `close()` (one signature, `SignedSession`) and `openings()`; `LmsSigner`, `register`; `verdict` (what the prover will prove of a signed session, checked in the clear; its public output is `Verdict.public`); `covering_certificate` |
| `streamed_wipe` | the streamed-label wipe (experimental): `wipe_key`, `label`, `labels`, `draw_indices`; `WipeParams(labels, k, batch, limit_ns, tiers)` (`covered`: batch 1, the only size `verdict` accepts), `WipeCommitment`; `Challenger(secret, server, session, params, signer, log=)` with `fill(chunk)`, `challenge()`, `receive(answers)`, `run`, `close(egress_bits)` (one signature, `SignedWipe`) and `coins()`; `Prover`, the honest server; `verdict` (its public output `Verdict.public`: the verdict and `egress_mib`), the reference of the Lean verdict that decides it (`MemoryChallenger.StreamedWipe`, `wipe_verdict.judge`), with `to_json` on the records it reads; the bound, `log2_error`, `retained_bits`, `challenges_for` |
| `run` | the composed check of a recorded run: `check(trusted, setup, digests, session, session_id)` over the verifier's list of memory-challenger keys (never the run's record), the setup's public outputs (`Setup`), the memory challenger's table of `C*`'s row digests, one `Session` (signed session, commitment, openings) and the id the auditor designated for the session, at P2's deployed point (`DEPLOYED`) under `memory_challenger.SCHEMES`, raising `Refused` with the first reason; `check_at` at another `Point`; `PousCheck.Run.check` (`pous-run`) in Lean |
| `adversary` | holdings (Lean `ColState`), `schedule`, `answer_prog`, strategies, `measure`; `col_round` is Lean's `colRound` |
| `protocol` | the generic protocol: `Scheme` (abstract), `Certificate`, `TimedAudit` (refuses what none of its certificates covers), `StorageProfile`, `PRIVATE`/`PUBLIC`, `verifier_key`, `commit`, `draw_salt`, `encode`, `decode`, `decode_tensor`, `challenge_key`, `responder`, `claim`, `LOCAL_CLAIMS` |
| `schemes` | `band.Band`, `dense.Dense`, `p3.P3`, `p2.P2` (experimental; one module each) as `protocol.Scheme`s with their certificates; `REGISTRY`, `DEFAULT` (`band-chain/d12/v1`), `get(name, segments, params)` |

**Large `W`.** Nothing above the codec holds `W` as one integer: encode each segment with `scheme.encode_segment` (the
segments are independent, so a process pool works), feed the results to `VerifierKey.from_segments`, or pass them to
`setup.verifier_key(..., encoded=...)` with `W` as its segments for the commitment check.

**Codecs.** Setup, the layout, `vk` and the audits take `codec=` (default: the parameters' own) and read only the
geometry: `blocks_per_segment`, `m`, `block_bytes`, `segment_bytes`, `segments`, `blocks`, `w_bytes`, `codec`,
`with_segments` and `manifest`. Both `dense.DenseParams` and `params.Params` provide it.

## Stable byte-format surface

These names and values are the wire format a second implementation (the GPU codec) must match. They change only with
a version bump in the strings they carry.

| Name | Value |
|---|---|
| `params.SALT_BITS`, `params.TAG_BITS` | 192, 256: `tag_s = salt ‖ u64(s)` |
| `primitives.CAPACITY_BITS` | 512: `Π₂` is `2m + 512` bits |
| `primitives.HEADER` | `b"pous-p3-v1-H"`, the first 12 bytes of `h_0` |
| `primitives.PI_DOMAIN`, `primitives.P_DOMAIN` | `0x48` (`Π₂`), `0x50` (`P_T`): the first byte of every round-function input |
| `primitives.EM_KEY_DOMAIN` | `0x4B`: the heuristic ARX `P_T`'s key |
| `verifier.leaf_hash`, `verifier.node_hash` | prefixes `0x00`, `0x01` |
| `setup.COMMIT_TAG`, `setup.SALT_DOMAIN` | `"verity-pous/w-commitment/v1"`, `"verity/pous/salt/v1"` |
| `audit.CHALLENGE_DOMAIN` | `"verity/pous/audit-challenge/v1"` |

## Backends

- **`reference` (the default).** `docs/p3-instantiation.md` §1, byte for byte.
  - `Π₂` is a 10-round Feistel network on `2m + 64` B with round functions `SHAKE256(0x48 || u8(i) || x)`.
  - `P_T` is a 10-round Feistel network on `m` with `SHAKE256(0x50 || u8(i) || T || x)`, where
    `T = tag_s || u8(layer) || u16(v)`.
  - `H` is the overwrite chain over `Π₂`. `h_0` is the 48-byte header (`"pous-p3-v1-H"`, `tag_s`, layer, `v`, the
    number of parents) zero-filled to `m + 64` B, with no call. Each parent, newest first, and then a pad10*1 block
    (`0x01 … 0x80`) goes through `(r, h) ← Π₂(B ‖ h)`: the first `m` B (`r`) is dropped, the last `m + 64` B (`h`) is
    chained. The key is the pad call's `r`. That is `npar + 1` calls of `Π₂` per key.
  - A chaining value finishes only its own row's key, because walking back needs the dropped `r`. Anything that
    reveals a parent and also finishes the row is a whole call state of `2m + 512` bits, more than the two labels it
    yields.
- **Broken modes, kept for attack replay only.** `XorSponge` is the previous default: rate `m`, capacity 512, a header
  block first. One stored state regenerates two labels, because `Π⁻¹` walks it back to the public first state
  (`lean/submissions/sponge-dense`, `dense_two_labels_from_one_state`). `MiyaguchiPreneel` stops the walk back, but on
  band rows the pre-image of one absorption is a checkpoint and a masked parent.
- **`heuristic_arx`.** Fast on GPUs, not analysed. It is the overwrite chain over an ARX `Π₂`, a wide butterfly with
  every word rotated in every stage. The butterfly runs twice the stages bit-level diffusion needs (`butterfly_stages`):
  40 at 256 words, where a flip test reaches the ideal from 16 stages. `P_T` is 5-round Even–Mansour over five
  butterflies, with key `SHAKE256(0x4B || T)`.
- **`ideal`.** The overwrite chain over a lazily sampled ideal `Π₂`, and an ideal tweakable `P` (the two-permutation
  model).
- **`RandomOracleKey`** is a monolithic `H` (Lean's `p3Model'`), used to check the code against Lean.

`vk` (`docs/p3-instantiation.md` §4) is a SHA-512 Merkle tree over the blocks, with the shape of RFC 6962. Each leaf
is `SHA512(0x00 || tag_s || u16(v) || R(c2_v))`, 99 bytes and one compression, where `R(c) = SHA512(P(8|c|) || c)` is
the block's `sha512/row/v2` digest (`rowleaf.sha512_row_v2_digest`), the digest `answer_circuit`'s `PousRowDigest`
checks; `tag_s` is 32 bytes and `v` 2, little-endian. Each node is `SHA512(0x01 || l || r)`. The memory challenger keeps
the root and every 64-byte leaf digest, or only the root and checks the server's audit path. This follows the
instantiation's byte format rather than `verity.ml.commitments`' frame v3.

## Rounds

A program's rounds are counted by `oracle.run`, never claimed. `audit.simulate` sets the round budget per answer at
`(Δ + rtt_allowance) / call_ns`. Here `call_ns` is the fastest whole oracle call, the ideal-permutation model's unit
and not an adversary's floor at the reference `Π₂` ("Main caveat"), and its 242 µs default is
`P_T`'s Feistel-SHAKE256 floor at 64 KB (a `Π₂` call is at least 480 µs), so the budget is 2 rounds at Δ = 0.5 ms.
`CERTIFIED_ROUNDS = 10` is what P3 is certified for at the operating point, counted in `P` and `Π₂` calls:
`Pous.SecurityProofs.P3ConcreteMeets64` holds at `D = 10` on `B(220, 12)`, and `P3ConcreteMeets64D12` at 12 on
`B(220, 16)`, both for Lean's `h_0`, not yet the code's ("P3 (research)"). The column-game certificate
`band_twoWay_220_12` covers 10 game levels, which the instantiation's rule of 3 calls per level (`Π₂` of the oldest
parent, `Π₂` of the pad, `P`) reads as 28 rounds; no `Meets` statement covers that reading.

`adversary.schedule` computes, for any holding, the fewest rounds after which each value is known. It covers keys
(each query waits for the parent it absorbs), forward labels, promotion from held or revealed columns, and the
key-gated reverse. `answer_prog` issues exactly those queries, and `measure` checks every answer bit-exactly and
asserts that the measured depth equals the scheduled one. The strategies of the POUS cryptanalysis are holdings:
column storage, whole lower labels as separators, meet in the middle through `P⁻¹`, and staggered runs.

## Tests

`tests/` covers:

- round trips on every backend and shape;
- known-answer vectors for `F`, `Π`, `P_T`, `H` and whole segments (`tests/fixtures/vectors.json`; the 64 KB segment with
  `VERITY_POUS_SLOW=1`);
- `H` recomputed from the spec's text;
- the RFC 6962 tree and the timed and simulated audits;
- per-tensor decode;
- replayed holdings;
- the narrow-state attack, which compresses `C` with SHAKE-256 as `H` and fails against the overwrite chain;
- the public-encoder setup (`test_pous_public_encoder.py`), on the band codec and on dense: the spec's seven tests
  (honest, a flipped byte, corrupt and continue, wrong sizes, server data unused, no encoder, the Lean vectors of
  `chainBand` and `chainDense`), encode-after-decode on arbitrary `C`, negative controls, and the flag;
- the two-labels-from-one-item attacks (`test_pous_two_for_one.py`), bit-exact on real labels: they succeed on the XOR
  sponge and on Miyaguchi–Preneel over band rows, and fail on the overwrite chain. The designer's state game
  (`pous_state_game.py`) agrees on dense and band rows;
- the continuous audit's memory challengers against the Lean model (`test_pous_continuous_difftest.py`).
  `pous_continuous_model.py` transcribes the memory challenger of
  `Security/Definitions/Pous/Model/Continuous.lean` definition by definition (a test finds each one word for word
  in the Lean file) and restates it on the memory challenger's clock. Fuzzed and scripted traces (answers at and one
  nanosecond past each deadline, wrong, pipelined, stale and dropped answers) must agree with it action for action, and
  Lean `ContMachine`s run against the memory challenger must get `contRecords`' verdicts and `contAccepts`;
- the Lean vectors (`tests/fixtures/lean_vectors.json`).

The Lean vectors come from `tests/lean/PousRefCheck.lean`, which evaluates Lean's `c1'`, `topLabel'`, `transpose`,
`decodeLower`, `tagOf`, the decoder, `bandGraph` and `colRound` on small cases. It proves by `rfl` that its generic
`decodeW'`, `Wlab'`, `enc` and `dec` are `P3Meets`' at `n = 220`. Lean evaluates labels lazily, so its encode cases
stop at `n = 3`.

The band vectors come from `tests/lean/PousBandCheck.lean`, which evaluates `parentsDesc (bandDAG B d)` for five
shapes and `labelG` with the decoder's formula on three `d = 1` cases, where the band already cuts row 2's oldest
parent, and encode after decode on three codes that are not encodings (`labelG_dec`). Its `rfl` theorems tie them to
`chainBand`'s `enc` and `dec`, and it imports `BandEncDec.lean` and `BandPubEncoder.lean` to pin `chainBand_encDec` and
`band_pub_meets_64_d12` at the secure-first point; a `d = 2` label case exceeds Lean's evaluation budget. `tests/test_pous_band.py` also recomputes a segment from the Lean text, runs setup, per-tensor decode and the
timed audit with the band as default, and replays the gap attack at the cliff.

The dense vectors come from `tests/lean/PousDenseCheck.lean`, which evaluates `chainDense`'s own `labelOv`, `ovKey`,
`dom`, `iv`, `padBlk` and the decoder's formula with a nonlinear toy `Π₂`. Its `rfl` theorems tie the encoder and
decoder to `chainDense`'s `enc` and `dec`. Compiled `ovStep` recomputes its call on every bit read, so the evaluated
cases are small:

- whole encodings of `B ≤ 2` blocks, with the real 512-bit `dom` and with a 4-bit analogue;
- the absorption order through `ovKey` over four blocks;
- the `dom`, `iv` and pad layout.

`tests/test_pous_dense.py` also recomputes a segment from the Lean definitions' text and runs setup, per-tensor decode,
the timed audit and the simulator with the dense codec as default.

## In Lean (`lean/`)

- **Trusted without proof:** `Security/Definitions/Pous/Game` (bits, uniform probability, parallel oracle programs `Prog`),
  `Security/Definitions/Pous/Model` (schemes, the audit game `AuditSecure`, `INC`/`TimedINC`, and the models below),
  `Security/Definitions/Pous/Accounting/{Params,LemmaA}.lean` (`ρ`, `δ`, `Meets`), `Security/Definitions/Pous/Assumptions.lean`,
  `Security/Definitions/Pous/Guarantees.lean` and, for the lemmas, `Security/Proofs/Pous/Statements.lean`.
- **`Meets sch reveal D Q k ε`:** `ε ≤ 2^-128`, `sch` is correct, `|C| ≥ |W|`, the space bound holds, and
  `AuditSecure` holds at a state of `⌊ρ|C|⌋` bits with bound `δ + ε`.
- **Proofs:** `Security/Proofs/Pous/`, checked by Verity's `tools/verity/lean/check.py` against `lean-audit.json`.
  `Security/Proofs/Pous.lean` proves `Pous.Guarantees.X` as `Pous.SecurityProofs.X`.
- **Untrusted submissions:** `grade.sh FILE Pous.Guarantees.X` grades one file against one guarantee (`lean/README.md`).

### The band

`Pous.BandChain.chainBand B d ℓ c dom` (`Security/Definitions/Pous/Model/BandChain.lean`) is the band over the overwrite chain on
one ideal permutation `Π₂`, and `Pous.BandMulti.chainBandSeg` lays out segments of it. The proofs are in this package
(`Security/Proofs/Pous/Chain/Flat.lean`, from the POUS store's `lean/submissions/band-chain/`, `band-multi/` and
`sponge-dense/`), with no hypothesis: `Pous.SecurityProofs.BandChainMeets64` gives `Meets` at 64 KiB, `B = 512`,
`D = 5`, `Q = 2^20`, `k = 111` and `ε = 2^-128` for every `d ≥ 5`; `BandMultiMeets14` the same for 448 segments
(14 GiB), with per-segment and with global salts, `BandMultiMeets14K106` at `k = 106` per segment, and
`BandMultiMeetsFamily` the `k = 111` row for every segment count `N` with `1 ≤ N ≤ 2^64`
(`Security/Proofs/Pous/Chain/Family.lean`). `BandMultiMeetsFamilyD2`, `BandMultiMeetsFamilyD1` and
`BandMultiMeetsFamilyD0` are that family at fewer rounds, `D = 2` with `k = 97` for every `d ≥ 2`, `D = 1` with
`k = 93` for every `d ≥ 1` and `D = 0` with `k = 89` for every `d`, and
`BandChainMeets64Rounds2` and `BandChainMeets64Rounds1` their one-segment rows
(`Security/Proofs/Pous/Chain/FamilyRounds.lean`). `BandPubMeets*` are the same rows with an untrusted server encoding and the
exact setup check (`Security/Definitions/Pous/Model/PubEncoder.lean`), under the four deployment conditions in their docstrings;
`BandPubMeetsFamily`, `BandPubMeetsFamilyD2` and `BandPubMeetsFamilyD1` are `BandMultiMeetsFamily`'s and its two rows at
fewer rounds (`Security/Proofs/Pous/Chain/PubFamily.lean`, `PubRounds.lean`). The
band needs `d ≥ D`.

### The graded audit

The graded audit (`Security/Definitions/Pous/Model/Graded.lean`, `GradedMeets` in `Security/Definitions/Pous/Accounting/GradedMeets.lean`) is
the sequential audit with a grace of `t` late answers. A round is late when its program takes more than `D` rounds or
`Q` queries on its own path, and every round, late or not, stays within a hard bound `(D_late, Q_late)`. The audit
accepts when every answer is correct and at most `t` are late; at `t = 0` it is the sequential audit
(`gradedMeets_zero_iff`). `Pous.SecurityProofs.BandGradedMeetsFamily` proves, with no hypothesis, the band at every
segment count from 1 to `2^64` and every `d ≥ 5`, at `D = 5`, `Q = 2^20`, `ε = 2^-128`, `δ = 1%`, for
`(t, k) = (0, 111), (1, 160), (2, 203), (3, 243)`, whenever `5 ≤ D_late ≤ d`, `D_late ≤ 12` and `Q_late ≥ 2^20`
(`Security/Proofs/Pous/Graded/`: `theorem1_ii_graded` with the binomial tail `pr_hits_ge_le_binomial`).

The game assumes three requirements on the protocol, and the certificate says nothing about a protocol that breaks one:

1. **No pipelining.** The memory challenger sends challenge `j + 1` only after it receives answer `j`, and each round's
   deadline runs from its own challenge's send time.
2. **A hard deadline.** An answer past `Δ_late` rejects, and a wrong or missing answer rejects. `D_late` is the
   sequential `Π₂` calls an adversary completes within `Δ_late + RTT`, as `D` is within `Δ + RTT`: `D_late ≤ 12` at
   241.5 µs per call is `Δ_late + RTT < 13 × 241.5 µs ≈ 3.14` ms, the band's cliff at `d = 12`, in the
   ideal-permutation model (not at the reference `Π₂`: "Main caveat").
3. **Every answer past `Δ` is counted late.**

**Help over the whole graded window.** The game's "no outside help" ("The problem") covers the whole graded audit, not
round by round: from the first challenge to the last answer the server has no outside help, and its state when each
challenge arrives is its storage at rest. In a deployment the help-accounted row counts, in `h`, the help received over
that same window. The worst row is `t = 3`, `k = 243` rounds. After a late answer the prover
times the next arrival, so the certificate needs the whole window to be too short to re-encode `C` from `W`.
Re-encoding a segment takes `R(d) = Σ_{v<512} (min(v, d) + 1) ≥ R(5) = 3,057` sequential `Π₂` calls (6,578 at
`d = 12`). The window is wall-clock time, and a rebuild runs through the rounds and the memory challenger's gaps between
them. A round answered at its deadline lasts just under `D + 1` call-times, so under the cap on `D_late` a row's rounds
last under `240·6 + 3·13 = 1,479` call-times, and no deadline bounds the gaps. **The graded audit's memory challenger
must therefore cap the audit's duration:** it aborts, without accepting, an audit that lasts more than `1,505 · call_ns`
(≈ 363.5 ms at 241.5 µs) on its own clock, from its first challenge's send to its last answer's receipt. That leaves 26
call-times (≈ 6.3 ms) for the worst row's 242 gaps if every round answers at its deadline.
`Pous.SecurityProofs.BackgroundRebuildExcluded` proves the counts, and that programs which know `W` but hold no state
about `Π₂`, each within any `T ≤ 1,505` sequential calls and `2^128` queries, output every label of a segment with
probability at most `2^-128` (`Security/Proofs/Pous/Rebuild/`). The bound rests on the cap, which is the proof's reach,
not on `R(d)`.

It excludes only re-encoding a *whole* segment from `W` with *no* state about `Π₂`, and is not a claim that no rebuild
is possible. Within the cap, `W` alone rebuilds a segment's first 253 labels (49%) at `d = 5` and its first 121 (24%) at
`d = 12`. `d` consecutive held labels restart the chain, so two held runs (10 labels, 2.0% of the segment) at `d = 5`,
or four (48 labels, 9.4%) at `d = 12`, let a prover re-encode a whole segment within the cap. That gap between the
storage held at rest and what is present when a challenge arrives is not bounded by this guarantee. The window argument,
and the cap's conversion into wall-clock time, count whole `Π₂` calls and so hold in the ideal-permutation model only:
at the reference Feistel-SHAKE256 `Π₂` a prover re-encodes nothing, answering each challenge from 1.53% of `C` in 481
Keccak-f, less than one call ("Main caveat").

No graded memory challenger exists yet: `audit.UnsignedRounds` and `protocol.TimedAudit` accept only when every answer
is correct and on time, the `t = 0` rule (`DISCREPANCIES.md` B5).

### The dense scheme

`Pous.Dense.denseScheme B ℓ` is one segment of `B` blocks of `ℓ` bits: `C_i = W_i ⊕ H(i, C_{<i})` over the complete DAG,
with no `pp` and no coins, so its rate is 1. Decoding recomputes each block with one query. `Pous.Dense.segScheme N B ℓ`
lays out `N` such segments on global block indices, so each block reads only its own segment's earlier blocks.
`Pous.Dense.segTagScheme N B ℓ salt` is the same store keyed as the implementation keys it: segment `s` is the dense
scheme on `B` blocks with oracle `H(tag_s, ·)` and local queries, where `tag_s = salt ‖ u64 s` is 256 bits (a 192-bit
salt, then the segment index).

**Proved, with H a random oracle.** Each is `Meets` with no named assumption: correct, rate 1, the space bound, and the
sequential audit at `ε = 2^-128`, so an adversary holding `⌊ρ|C|⌋` bits passes with probability at most `1% + 2^-128`.

| Theorem | Store | `D` rounds, `Q` queries | `k` |
|---|---|---|---|
| `Pous.SecurityProofs.DenseMeets64` (`Pous.Guarantees.DenseMeets64`) | one segment: 512 blocks of 64 KB (32 MiB) | 5, `2^20` | 106 |
| `Pous.SecurityProofs.SegMeets14` (`Pous.Guarantees.SegMeets14`) | 448 such segments (14 GB) | 5, `2^20` | 106 |
| `Pous.SecurityProofs.SegTagMeets14` (`Pous.Guarantees.SegTagMeets14`) | the same 448 segments, keyed as the implementation keys them, for every salt | 5, `2^20` | 106 |
| `Pous.SecurityProofs.DenseMeets` (`Pous.Guarantees.DenseMeets`) | one segment: `2^16` blocks of 1 KB | 64, `2^20` | 107 |

Each composes B5 (`dense_timedINC`, `seg_timedINC`, from the pebbling bounds `complete_hard` and `seg_hard`) with
Theorem 1(ii) sequential. At the 64 KB point the sampling term is `p₀^106 ≈ 0.98%`, while `p₀^105 > 1%`, so `k = 106` is
tight. The 14 GB store needs the same `k` as one segment. `segTag_meets_14` follows from `seg_meets_14` by
`segTag_auditSecure`, which reduces the tagged scheme's audit game to the global scheme's. An injective map of the
tagged queries into the global ones sends each segment's queries to their global form (`tagEmbed`, for `2 ≤ N ≤ 2^64`
and `ℓ ≥ 256`). Under it the two schemes write the same codeword, every tagged adversary is a global adversary with the
same rounds, work and acceptance probability, and the restricted oracle is again uniform. `domainSep_equiv` and
`tag_injective_14` record the index-level fact: the tagged keying is the global one relabeled injectively.

**Over the overwrite chain** (`Pous.Sponge.chainDense`, `H` the overwrite chain on an ideal permutation, proved with no
hypothesis in `Security/Proofs/Pous/Chain/Flat.lean`): `Pous.SecurityProofs.ChainDenseMeets64`, 512 blocks of 64 KiB at
`D = 5`, `Q = 2^20`, `k = 111`; and the deployment point, `Pous.SecurityProofs.ChainDenseMeets111`, `2^16` blocks of
1 KiB at `D = 64`, `Q = 2^20`, `k = 111`. The deployment was pinned at `k = 107`, which the ex post facto argument does
not reach; it is re-pinned at `k = 111`, four challenges more.

**Not proved here:**
- a concrete permutation for `Π₂` (the main caveat);
- the conversion of the deadline into `D = 5` rounds, which assumes no ASIC and at least 50 ns per Keccak-f, and the
  reading of help through the help-accounted row ("The problem"): both outside Lean;
- decode cost, which isn't modeled.

## Guarantees and lemmas

The frozen statements (claims about a model, an ideal primitive or the model's memory challenger, which no Lean function
computes) are in `Security/Definitions/Pous/Guarantees.lean` and its `Guarantees/` modules, each docstring ending with its
category. The lemmas (`KFloor`, Theorem 1, Track A, the pebbling and sampled-audit results, the tagged keying's
transfer, the column game's rows, `PerAnswer`, `ContinuousTimeAverage`, and the falsified
`InvExPostFactoTwoLayer` beside `N10Refuted`) are stated under the same `Pous.Guarantees` names in
`Security/Proofs/Pous/Statements.lean`, and `ContinuousComplete`, the memory challenger's completeness, with them.
The properties of the Lean verdicts `PousCheck.TimedAudit` and `MemoryChallenger.verdict`, and of the composed check
`PousCheck.Run.check`, are in `Security/Properties/Pous/`.

| Statements | What | Status |
|---|---|---|
| `KFloor` | `k = 86` is the least `k` with `ρ^k ≤ δ` | proved |
| `Theorem1ii`, `Theorem1iiTimed` | report Theorem 1(ii): `INC(β, π)` (timed: per block) gives `π + p₀^k` for product-form responders | proved |
| `Theorem1iTimed`, `Theorem1iiSeq`, `Theorem1iiiTimed` | Theorem 1(i) timed, 1(ii) sequential, 1(iii) timed | proved |
| `FibreCount`, `A1CompressionCard`, `A1CompressionPr`, `A6RederiveUntimed`, `A6RederiveTimed` | fibre counting, compression, cheap re-derivation | proved |
| `A3Prime`, `A3PrimeLam` | the sequential audit from Lemma A, taken as a hypothesis | proved |
| `A2LemmaAM1`, `A5RabinGcd`, `A7Params`, `M1Meets` | Lemma A in model M1, the Rabin gcd step, A7′'s parameter rows, `Meets` in M1 given B1′ | open |
| `B1IndepSet`, `B2Stacking`, `B3Chain` | independent sets, Fisch's stacking lemma (Claim 12), the tweaked hash chain | proved |
| `FischClaim13`, `FischClaim14` | Fisch ePrint 2018/702, Claims 13–14 | proved |
| `B5ExPostFactoBounded`, `B5ExPostFacto` | Pietrzak's ex post facto bound for one-way labels | proved |
| `TransposeMixing`, `ReverseNeedsAllChildren` | N10's two core lemmas for P3's invertible labels | proved |
| `N10Refuted` | `¬ InvExPostFactoTwoLayer`: N10 as first stated (PIEs Conjecture 1 at `L = 2`) is false | proved |
| `SampledAuditProd`, `SampledAuditSeq`, `SampledAuditSim` | "fewer than `k` answerable" gives a sampled-audit bound | proved |
| `StackedAuditSeq`, `StackedAuditOnePercent` | the stacked one-way scheme's audit clause; the second is conditional on a depth-robust graph | proved |
| `DenseMeets` | the dense scheme S1 meets `Meets` with no named assumption, in the ideal-hash model only (security, correctness and rate, not decode cost) | proved |
| `P3ConcreteMeets64`, `P3ConcreteMeets14GB`, `P3ConcreteMeets64D12` | P3 with the overwrite-chain `H` (`Security/Definitions/Pous/Model/P3Chain.lean`), 64 KB labels, `k = 106`: one segment and 971 segments (14.005 GB) on `B(220, 12)` at `D = 10`, and `B(220, 16)` at `D = 12`, in the ideal two-permutation model | proved |
| `BandChainMeets64`, `BandChainMeets64D5`, `BandChainMeets64D12`, `BandMultiMeets14`, `BandMultiMeets14K106`, `BandMultiMeetsFamily` | the band over the overwrite chain (`Security/Definitions/Pous/Model/BandChain.lean`): 64 KiB labels, `D = 5`, `k = 111`, every `d ≥ 5` and `d = 5`, `12`; 448 segments (14 GiB) at `k = 111`, and at `k = 106` per segment; every segment count from 1 to `2^64` at `k = 111` | proved |
| `BandMultiMeetsFamilyD2`, `BandMultiMeetsFamilyD1`, `BandMultiMeetsFamilyD0`, `BandChainMeets64Rounds2`, `BandChainMeets64Rounds1` | the band at fewer rounds: every segment count from 1 to `2^64` at `D = 2`, `k = 97`, every `d ≥ 2`, at `D = 1`, `k = 93`, every `d ≥ 1`, and at `D = 0`, `k = 89`, every `d`; one segment (64 KiB) at `D = 2` and `D = 1` | proved |
| `BandPubMeets64`, `BandPubMeets14`, `BandPubMeets14Global`, `BandPubMeetsFamily`, `BandPubMeetsFamilyD2`, `BandPubMeetsFamilyD1` | the same band rows with a public encoder and the exact setup check (`Security/Definitions/Pous/Model/PubEncoder.lean`, `Security/Definitions/Pous/Accounting/PubMeets.lean`) | proved |
| `BandGradedMeetsFamily` | the band with up to `t` late answers (`Security/Definitions/Pous/Model/Graded.lean`, `Security/Definitions/Pous/Accounting/GradedMeets.lean`): every segment count from 1 to `2^64`, `d ≥ 5`, `D = 5`, `(t, k) = (0, 111), (1, 160), (2, 203), (3, 243)`, each late round within `(D_late, Q_late)` with `5 ≤ D_late ≤ d`, `D_late ≤ 12` | proved |
| `BackgroundRebuildExcluded` | under the graded memory challenger's cap of `1,505 · call_ns` on an audit's duration, no audit re-encodes a whole band segment from `W` alone (`Security/Definitions/Pous/Accounting/Rebuild.lean`): every row's rounds last under 1,479 call-times at `D_late ≤ 12`, re-encoding costs `R(d) ≥ R(5) = 3,057`, and programs with no state about `Π₂` re-encode a whole segment within `1,505` calls with probability at most `2^-128`. A prefix, or a whole segment from a few held labels, can be rebuilt within the cap | proved |
| `ChainDenseMeets64`, `ChainDenseMeets111` | the dense scheme over the overwrite chain: 64 KiB at `D = 5`, and the deployment point (`2^16` blocks of 1 KiB, `D = 64`), both at `k = 111` | proved |
| `B5Stacked` | the one-way bound on Fisch's stacked graph | proved |
| `DenseMeets64`, `SegMeets14`, `SegTagMeets14` | the dense scheme at 64 KB, `D = 5`, `k = 106`, in the ideal-hash model: one segment, 448 segments (14 GB), and the 448 segments keyed as the implementation keys them, for every salt | proved |
| `SegTagAuditSecure`, `DomainSepEquiv`, `TagInjective14` | the tagged keying's transfer: the tagged scheme is as secure as the global one, and the tags relabel the global indices injectively (448 distinct tags) | proved |
| `BandTwoWay220_12`, `BandTwoWay220_12D11`, `BandTwoWay220_16` | the band certificate for the two-directional column game (`Pous.ColumnGame`): `B(220, 12)` at game depths 10 and 11, and `B(220, 16)` at depth 12, save at most `X = 3` labels | proved |
| `StreamedWipeMeetsReadsTight`, `StreamedWipeMeetsServer`, `StreamedWipeMeetsServer5GB`, `StreamedWipeMeetsServerH200`, `StreamedWipeMeetsServer8H200HBM`, `StreamedWipeMeetsServer8H200HBMRAM`, `StreamedWipeMeetsServer8H200` | the streamed-label wipe as `StreamedWipeMeetsReads` states it, with the count of the missing labels, `m·(⌊log₂⌊3N/m⌋⌋ + 1)` for `m = ⌈N/r⌉ − 1`, in place of `N`; per server against its assumed memory capacity `M_s = 512·N + R`, at most `2^-128 + Adv_PRF` for `q ≥ B(N, r, k, R) + E`, and at the 5 GB point `B ≤ 4·10^10` for `q ≥ 4·10^10 + E`; one H200 as measured at its deployed `k = 5500`, and an 8×H200 server with 3.9 GB outside the fill, its HBM at `k = 93,843`, with RAM at `k = 261,229` and every tier at `k = 1,268,143` | proved |
| `StreamedWipe`, `StreamedWipeMeets2600GB` | the streamed-label wipe ("The streamed-label wipe"): for every prover of the sequential wipe, `Pr[accepted ∧ the prior content survives] ≤ (g/N)^k + k·C(N, g+1)·2^(M+E)/2^(q+(g+1)·b) + Adv_PRF`, with no named assumption; at 2.6 TB and `k = 74,000`, at most `2^-40 + Adv_PRF` for `q ≥ E + 8.6·10⁹` bits | proved |
| `StreamedWipeMeetsFamily`, `StreamedWipeMeets96GiB`, `StreamedWipeMeets141GB` | the streamed-label wipe at `k = 2152` reads: for every fill of at least one 32 MiB segment, at most `2^-128 + Adv_PRF` for `q ≥ ⌈M/19⌉ + E`; at 96 GiB and 141 GB the explicit thresholds | proved |
| `StreamedWipeMeetsReads`, `StreamedWipeMeetsH200` | the streamed-label wipe at any `k` reads with `(1 − 1/r)^k ≤ 2^-129`, `verity/wipe/v0`'s figure at the session's `k`: for every fill and `R` bits outside it, at most `2^-128 + Adv_PRF` for `q ≥ 512·(⌈N/r⌉ − 1) + N + ⌊log₂ k⌋ + 130 + R + E`; an H200 (141 GiB, 151,397,597,184 bytes, filled whole) at `k = 2906`, `q ≥ 39,068,035,213 + E` bits, at most 5 GB | proved |

**Why N10 fails.** Top label `v` reads only column `v` of the lower layer, so an adversary can pay for a lower label a
column at a time, where whole-label pebbling charges it a whole label. N10′, the column-aware restatement over
`Pous.ColumnGame`, is being drafted.

### P2 end to end (`Security/Properties/Pous/EndToEnd.lean`)

Two kinds of statement at P2's deployed point, which nothing joins inside Lean:

- `Pous.EndToEnd.RunCheckAccepts` and `RunCertifies`, about the deployed code, deterministic and with no assumption.
  When the composed check (`run.check`, Lean's `PousCheck.Run.check`) accepts, against the verifier's own list of
  memory-challenger keys (`trusted`, which never comes from the run's record), the setup's public outputs, the memory
  challenger's table and its signed session under the id the auditor designated, the session audits that setup at the
  deployed point (`AuditsSetup`: every block, `k = 105`, within `Δ + RTT`, 2,056-byte answers, bound to the published
  root), runs under the designated id and is signed under a key in `trusted`, and for every `C*` of the point's shape
  whose leaves have the published root, every one of the device's 105 recorded answers was the challenged block of
  `C*` on time on the device's own clock; or the run holds a SHA-512 collision or an LMS forgery. `RunCheckAccepts` is
  proved (`Pous.EndToEnd.SecurityProofs.runCheckAccepts`). `RunCertifies` adds setup's
  acceptance (`Accepted`): `C*` decodes to the committed `W` and its root is published. It is proved
  (`Pous.EndToEnd.SecurityProofs.runCertifies`) from `RunCheckAccepts` and setup's `AcceptsDecodes` and `PublishesRoot`
  (`PousCheck.Properties.SecurityProofs.acceptsDecodes` and `publishesRoot`, guarantees), and `periodHeld` reads it; no
  guarantee states it alone (`unstated`).
- `Pous.EndToEnd.EndToEndRing`, **in the ring model, with ω uniform**: a server holding at most `⌊ρ·|C|⌋` bits
  between setup, the session and its rounds, each answer `(D, Q, L)`-bounded (`Server S D Q L`), shown each index as
  its round opens and nothing else of the memory challenger's coins (the coin tree's root and key leave the device only
  after the last answer), with the coins drawn online by the code's own sampler and committed in the code's coin tree,
  passes with probability at most `1% + 2^-128`. It is proved (`Pous.EndToEnd.SecurityProofs.endToEndRing`) from one
  named assumption, `P2RingAuditSecure D Q L` (open); it takes no hiding premise on the coin tree. The budget is
  `2^-129 + 2^-130 + ε_rs ≤ 2^-128`, with `ε_rs = 0`
  proved (`DrawExact`: `uniformOn` reads 11 bytes at `2^19` blocks and `2^19` divides `2^88`, so no word is rejected).
  `EndToEndRingDeployed` is it at one root's depth, `D = 16,446`, about half of `RingTight`'s falsification at
  32,894, with `Q = 2^20` and `L = 2^64` (the truncation attack passes `k = 105` only at about `2^168`). In the ring
  model a root is 16,447 multiplication rounds (16,446 over the exponent's bits and one to fix the sign,
  `rootProg_cost`), and recomputing a block takes two, `2 · 16,447 = 32,894`; 16,446 is the largest `D` at which that
  program cannot finish one root. The timing keeps its 2× margin there: at SeqRoot's floor, the timing red team's
  modelled 361.55 ns per single-core squaring cut by the measured 8-core speedup of 3.490× to about 103.6 ns per
  cooperating squaring, 16,447 rounds take 1,704 µs, 1.995× `Δ + RTT`, and 1.64× and 1.25× under the red team's sync
  model. `HonestPassesRing` and
  `HonestPassesAccepts` prove that the honest full-state server passes P2's ring audit, and the end-to-end event, with
  probability 1. Beside each ring-game and period guarantee sit its honest and nontrivial lemmas
  (`endToEndRing_honest` and `endToEndRing_nontrivial`, likewise for `endToEndRingAt`, `periodHeld`, `periodHeldAt`,
  `periodCapacity`, `obligationHeld` and `SpaceOfPeriodHeld`; `periodHeldHonest` and `periodHeldNontrivial` for
  `periodHeldDeployed`): the first shows the premises hold at an instance, the second that the bound fails for a
  verifier that accepts everything.

`P2RingAuditSecure` is also derived from the per-round conjecture `P2RingRoundSecure p ε₀ D Q L`:
at every block count from `2^19` to `2^25`, at a boundary where the server holds at most `⌊ρ·|C|⌋`
bits, a uniform challenge is answered with probability at most `1 − p` given the earlier answers, outside an event of
mass `k·ε₀` over `k` rounds (one `ε₀` per round, the same at every block count). `Pous.SecurityProofs.P2RingAuditSecureOfRoundSecure` gives `P2RingAuditSecure` from it when
`(1 − p)^105 + 105·ε₀ ≤ 1% + 2^-129 + 2^-130` (`p = 0.043` with `ε₀ = 2^-137` suffices; `p < 0.04291` doesn't), and
`Pous.SecurityProofs.P2RingGradedOfRoundSecure` is the graded bound: a server held at most `⌊ρ·|C|⌋` bits at `m` of
the `k` boundaries passes with probability at most `(1 − p)^m + k·ε₀`. Its falsification tests are
`Security/Proofs/Pous/P2/RoundTests.lean`'s, at every block count in the range: recomputation forces `p = 0` at
`D ≥ 32,894`, and truncation refutes `p = 0.043` from `L ≥ 2^167` at `2^19`.

`EndToEndRing` reaches the deployed system only through four claims, each cited on its own:

- `pous-ring-device` (the model): the deployed server is a `Server`, its state bounded between setup, the session and
  its rounds, with one session per audit (the memory challenger's `SessionLog`, of which its host never restores an
  earlier copy: `trusted-host`) and the coins drawn
  online;
- `pous-ring-speed` (the numbers): within `Δ + RTT` no allowed hardware finishes a 16,447th sequential multiplication
  round (`D = 16,446`), more than `2^20` key queries or `2^64` ring operations per answer. It rests on the floor
  behind `ROOT_CALL_NS` (SeqRoot, under P2 in "The generic protocol"), a model cut by a measurement: the timing red
  team's modelled single-core floor and the 8-core cooperation speedup measured on one Zen 4 chiplet; its margin is
  1.995×, and 1.64× and 1.25× under the red team's sync model;
- `pous-ring-instantiation`: P2 with the identity labeling and SHA-512-CTR keys is no easier than `p2ring` with
  uniform ω;
- `prf/sha-512`: the memory challenger's derived index streams are as good as uniform bytes.

Two gaps stood between these statements and the deployed verifier:

- G-1, one session per audit: the check's half is fixed in code; one session per id is the memory challenger's
  `SessionLog` under its host presumption, which `trusted-host` covers. Every session names its setup: a `SessionCommitment`'s
  `binding` has no default, the `MemoryChallenger` refuses one that is not a 64-byte root, and the harness's memory
  challenger (`benchmarks/pous/memory_challenger/live.py`) binds every session to the root the setup publishes for the
  `C*` its digests describe (`VerifierKey.from_row_digests`). Every session is also its audit's: its id (`session`, in
  the signed header) is the auditor's, designated for one audit, and the device runs at most one session under an id.
  Its `SessionLog` is a file of its own, never in the signer's key directory (`live.py --session-log`, `bench.py
  --mc-session-log`). It is append-only: the id goes in, fsynced, when the session commits, before its first challenge,
  so a session that never runs has used its id too. A used id is refused, a log that is missing or unreadable refuses
  every session, and nothing rewrites, truncates or remakes a log. A file cannot see its host restore an earlier copy of
  it, from a snapshot, a backup or otherwise, nor restart the memory challenger onto one. That the host never does is
  `trusted-host` (`verity.core.claims`): "the auditor's device runs its stated code, timing
  included, on a host the prover doesn't control, with its CPU isolated from other tenants and a monotonic clock; the
  host never restores its earlier state". This log is part of that state, as is the LMS leaf state.
  Under one secret a reused id would also replay the coins, which `CoinTree.draw` derives from the secret and the id.
  The composed check accepts a session only under the id the auditor designated (`run.check`'s `session_id`), so an
  audit has at most one session it can accept. The Lean states the check's side (`RunCheckAccepts`: an accepted session
  carries the designated id); the device's refusal is its code and log, which the Lean does not model.
- G-2, the composed setup-to-challenger check, fixed: `run.check` (`PousCheck.Run.check`, `pous-run TRUSTED.json
  SETUP.json DIGESTS.bin SESSION.json SESSION_ID`) refuses a run unless the setup published `2^19` blocks under a
  24-byte salt, the table's leaves have the published root, the session audits that setup at the deployed point, is
  bound to that root and carries the designated id, and the verdict accepts it against the table under a key of the
  verifier's own list (`TRUSTED.json`, never the run's record). `check/tests/test_run.py` holds the two
  to one another, and `Accepted` is now setup's acceptance and this check's.

## Named assumptions (`Security/Definitions/Pous/Assumptions.lean`)

Each is a `Prop` taken as a hypothesis, never an `axiom`.

- `LemmaA sch S D Q g Γ G`: on the good setups `G`, an `S`-bit state and a per-block `(D, Q)` responder are right on at
  least `g` blocks with probability at most `Γ`. For M1 (`G = True`) it is `A2LemmaAM1`.
- `B1Prime sch Mdom S D Q k` (Conjecture B1′): `sch`'s sequential audit obeys model M1's completion bound. Used by
  `M1Meets`.
- Retracted: `InvertibleLabels`, which was N10's statement. It is false (`N10Refuted`).

## Provisional

- `|C| ≥ |W|` in `Meets` (so `|pp| ≤ 0.05·|W|`): otherwise a scheme can keep `W` in the free `pp`.
- `ε_crypto ≤ 2^-128`: an unbounded `ε` makes the requirement vacuous.
- Which reveal a graded `Meets` may use, which waits on the per-answer versus shared deadline. `GradedMeets` is the
  sequential reveal with a deadline per answer.

## Choices that could be wrong

- **H is a monolithic random oracle:** the main caveat, at the top.
- **One segment.** Every model plays one segment against a fresh primitive. A deployment must put a salt and the
  segment index in every oracle query.

## Not here yet

Done, in this package: in the random-oracle model, the dense re-certificate at `POINT_64K` (`dense_meets_64`), the
multi-segment lemma (`seg_meets_14`, 14 GB) and its form for the code's per-segment `tag_s` domains
(`segTag_meets_14`), in `Security/Proofs/Pous/Dense/`. Over the overwrite chain, the ex post facto bounds
`ChainExPostFactoG` and `ChainExPostFacto` (`chainExPostFactoG_v6`, `chainExPostFacto_v6`), and from them the band,
the dense chain scheme and P3 with the concrete `H` ("Guarantees"), in
`Security/Proofs/Pous/Chain/Flat.lean`. Pending:

- P3's certificate: the code's `h_0` and Lean's `iv(domP3 tag L v)` differ ("P3 (research)").
- A GPU codec for the dense scheme (the GPU codec there is P3's).
- Native and GPU kernels for encode and decode, and a responder in native code on a high-priority stream.
- The bulk bandwidth audit that would pin storage to HBM rather than the whole server.
- In `lean/`: N10′ (`ColumnAwareDraft`, `ForwardCoreDraft`, `n10prime`; its core `n10prime_core` is open) and
  `p3-meets` (P3 with a random-oracle `H`).
