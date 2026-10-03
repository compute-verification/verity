# Proof of useful space (POUS)

A GPU keeps an encoding `C` of model weights `W` in its memory and decodes `W` from it on every use. A timed audit must
certify that the GPU still holds more than a `ρ` fraction of `C`: answering a random block within the deadline requires
the block itself, because rebuilding it from a smaller state takes more sequential oracle rounds than the deadline
allows. This directory holds the reference implementation, `verity_pous` ("Codecs" through "Tests" below), and the Lean
formalization, `lean/` (from "In Lean" on): the definitions, the guarantees and the proofs so far.

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
post facto bounds are proved in this package (`Pous/SecurityProofs/Chain/Flat.lean`): `ChainExPostFacto` (B5 for `Π₂`)
for dense, and its DAG-generic form `ChainExPostFactoG` for the band, with the pinned finals below (P3 with the concrete
`H`, the band, and the dense chain scheme). They hold in the ideal-permutation model only. `Π₂` itself (Feistel-SHAKE)
is a heuristic ideal permutation. Sources: the POUS store's `docs/p3-cryptanalysis.md`, "Narrow-state H", and
`docs/band-vs-dense.md`.

## The problem: timed audit, perfect isolation

- **Syntax.** A trusted encoder `Enc(W) → (pp, C, vk)` runs once and erases its secrets. `W` is arbitrary and fixed
  before any setup randomness (all zeros included). Decoding `Dec(pp, C) = W` is public and bit-exact. `C` is `N`
  blocks of `b` bits.
- **Parameters.** `λ = 128`; `|C| + |pp| ≤ 1.05·|W|`; retention `ρ = 18/19`; soundness `δ = 1/100` plus the
  construction's `ε_crypto`.
- **Time.** A responder is `(D, Q)`-bounded: `D` sequential steps and `Q` operations in total. In ideal-model proofs a
  step is one round of parallel queries to the primitive, and `Q` counts queries.
- **Game.** The adversary picks `W`; then `Enc(W)`. Preprocessing, with no time limit, leaves a state `σ` and nothing
  else. The verifier issues `k` uniform block indices, all at once or one at a time. Each answer is due within the
  deadline, so the responder is `(D, Q)`-bounded, with no outside help. Whenever a challenge arrives the adversary's
  state is at most `ρ|C|` bits; `W` and `pp` are free.
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
  `Pous.Protocol.Model.Invertible`.

The concrete primitives are those of POUS `docs/p3-instantiation.md`. Each set of parameters names its codec
(`params.codec`), and every function that takes a codec defaults to it. P2 (`schemes/p2.py`, experimental) is a
`protocol.Scheme` and not a codec: see "The generic protocol".

The secure-first scheme is the band graph with in-degree `d = 12` (Daniel, 28 Sep 2026, 01:10Z; POUS
`docs/band-vs-dense.md`), superseding the dense choice of `docs/scheme-choice-e2e.md`. Dense stays selectable
(`dense.POINT_64K`).

## The band scheme (secure-first)

`band.SECURE_FIRST` is `BandParams(512, 524288, d = 12)`: 512 blocks of 65,536 B per segment (32 MiB), audited with
`k = 111` challenges and a certificate for `D = 5` calls of `Π₂`, each at least 241.5 µs
(`Pous.SecurityProofs.BandMultiMeetsFamily`, for every `d ≥ 5` and every segment count, with no hypothesis). Segment `s`
is exactly `chainBand B d ℓ 512 (dom tag_s)`. It is the dense scheme
below with one change: block `v` absorbs only its `d` newest parents, `C_{v-1}, ..., C_{max(0, v-d)}` (Lean
`parentsDesc (bandDAG B d) v`), then the pad.

**The per-call floor** (`primitives.pi2_keccak_f`, `pi2_call_ns`). A 64 KiB `Π₂` call is ten Feistel rounds, and
each round function absorbs a 2-byte prefix and a 65,568-byte half: 65,570 bytes, 483 blocks at SHAKE256's rate of
136. Round `i + 1`'s absorb block `k` needs only round `i`'s squeeze block `k`, so the rounds stream, and in the
overwrite chain a call boundary is a round boundary. A call in a chain costs 10 × 483 = 4,830 sequential Keccak-f,
241.5 µs at 50 ns per Keccak-f; counting each round's 482 further squeeze blocks in sequence too would give 9,650
(482.5 µs). The band's and dense's `call_ns` is 241,500 ns, so `D = 5` covers `Δ + RTT < 6 × 241.5 µs = 1.449` ms.

- Block `v` costs `min(v, d) + 1` calls: 6,578 per segment at `d = 12`, 20 times fewer than dense's 131,328. Decoding
  is `d + 1 = 13` rounds deep, and a segment decodes in 24.6 s on one core in Python (dense: 493.9 s).
- **Deployment condition, the cliff.** The effective `D`, the sequential `Π₂` calls an adversary completes within
  `Δ + RTT`, must not exceed `d`. With `d ≥ D′` the band is exactly as hard as dense at `D′` (`band_chain_hard`). At
  `D′ = d + 1` the gap attack (`band_gap_attack`, `band_cliff_d12`) holds every block but each `(d + 1)`-th: every
  dropped row has all its parents held, and the audit accepts with probability 1. At 241.5 µs per call, `d = 12`
  holds while `Δ + RTT < 13 × 241.5 µs ≈ 3.14` ms. `audit.TimedVerifier` refuses a deadline past it
  (`band.check_deployment`), and `test_gap_attack_at_the_cliff` replays the attack on real labels.
- `band.codec_for(d)` gives the codec for any in-degree; with `d ≥ B − 1` the band is the complete DAG, dense.

## The dense scheme (selectable)

`dense.POINT_64K` has `B = 512` blocks of `ℓ = 65,536` B (524,288 bits) per segment, so a segment is 32 MiB: the
point `dense_meets_64` certifies. P3 keeps 65,560 B, the multiple of `n = 220` it needs. The audit runs with
`k = 111` challenges and a certificate for `D = 5` calls of `Π₂`, each at least 241.5 µs. The 111 is what
`Pous.SecurityProofs.ChainDenseMeets64` certifies for this overwrite-chain `H`, for one segment, with no hypothesis; a
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
`h_0` is the instantiation's 48-byte header ("Backends") in the low bytes, with the capacity part zero. The pins are
about Lean's `H`, so they do not certify the code's (`DISCREPANCIES.md` P3 item 16).

## Lifecycle

1. **Commit.** The server commits to `W` before any salt exists (`setup.commit`: tagged SHA-256 over the parameters and
   `W`).
2. **Salt.** The verifier draws the 24-byte salt after the commitment (`setup.draw_salt`, through `verity.randomness`
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
     chain), given `ChainExPostFacto`, which `chainExPostFacto_v6` proves (`lean/Pous/SecurityProofs/Chain/Flat.lean`;
     not pinned).

   An accept-all check provably fails (`acceptAll_fails_dense`). P3 goes through the same code, but its
   encode-after-decode lemma is not proved.

   The encoder is deterministic and public, so the server running it changes nothing in the timed game. A
   server-chosen or predictable salt would break the model's "`W` before coins" order.
4. **Serve.** `layout.decode_tensor` decodes one tensor from the segments it overlaps.
5. **Audit.** The verifier reveals `k` indices one at a time (`audit.TimedVerifier`, uniform with replacement, keyed
   by `audit.challenge_key` and bound to `vk`'s root). Each block must arrive within `Δ = 0.5 ms` plus
   `rtt_allowance_ns`, and must match `vk`. The audit accepts only if every answer is correct and on time. At 300 µs
   about 1.5% of honest 132-answer audits fail on the host clock (POUS `docs/p3-gpu-measurements.md`). The network
   allowance defaults to 0 until the verifier's RTT is measured.

## The generic protocol (`protocol`, `schemes`)

The lifecycle above is written once, in `protocol`, for any `protocol.Scheme`.

- **What a scheme supplies.** It implements `encode_segment(salt, s, w_seg)` and `decode_segment(salt, s, blocks)`, and
  overrides `respond` only if an honest answer is more than the stored block. It also sets `name` (its byte-format
  string), `backend` (its reference primitives) and `certificate`. As with `verity.commitments.scheme.CommitmentScheme`,
  a scheme is named and versioned, looked up by name, and has conformance vectors pinned under its name.
- **The registry.** `schemes.get(name, segments)` returns a registered scheme at its own parameters, or at
  `params=` of the same name. Each scheme is one module, `schemes/<name>.py`:
  - `band-chain/d12/v1`, the default;
  - `dense-chain/v1`;
  - `p3/v1`;
  - `p2-16448/v2`, **experimental**.
- **`protocol.Certificate`** is what an audit at the scheme's parameters establishes, Lean's
  `Meets S .sequential D Q k ε`. It holds:
  - `k`, `rounds` (D), `work` (Q), `call_ns`, `rho`, `delta`, `eps_log2`;
  - `cliff`: the most rounds the structure tolerates at any `k`;
  - `segments`: how many segments of `W` the theorem is about;
  - `theorem`, and its open `hypotheses`;
  - `claims`: ids in `verity.claims`' format.

  Each `theorem` is the pinned proof `Pous.SecurityProofs.X` it rests on, of the guarantee `Pous.Guarantees.X`. At
  the frozen point, the band's certificate is `Pous.SecurityProofs.BandMultiMeetsFamily`, with no hypothesis: k 111,
  D 5, Q 2^20, 241.5 µs, cliff at `d`, for every segment count N from 1 to `2^64` (all that `BandParams` allows), so no
  segment count is a gap. It is one `Meets` for the whole
  store: challenges are uniform over all its blocks, the state bound is the whole store's, and one permutation serves
  every segment. A 70 GiB store, 2,240 segments, is certified whole. At one segment it is also `BandChainMeets64`, and
  at 448 (14 GB) `BandMultiMeets14`.

  Dense is `Pous.SecurityProofs.ChainDenseMeets64`, with no hypothesis and no cliff, for one segment; other segment
  counts are gaps in the storage profile. P2 (`schemes/p2.py`) is **experimental**:
  - It is the squaring-based scheme: two big-integer squarings per 2,056-byte block to decode, and no hash call.
  - It follows the P2 spec (the POUS store's `docs/efficient-crypto/p2-spec-draft.md`, draft v0) except for its
    expander:
    - the payload is 2,054 bytes per block, which is the spec's profile `p2-2054B`, and the manifest names it;
    - every block, payload and key is a fixed-width, unsigned, big-endian integer;
    - the tweak is rejection-sampled below p, never reduced.
  - Its keys are fresh per block, expanded where they are used: ChaCha8 keyed by a 32-byte seed per segment
    (`SHAKE256("verity/pous/p2/seed/v1" ‖ tag_s)`), with nonce = the block index. Nothing key-sized is stored, which
    keeps the 1.05× space bound. **The proof's setup assumption, the spec's P2-EXP-IO, names SHAKE256 and not ChaCha8**
    for these keys. SHAKE256 would cost about 12% of device throughput (an estimate). The choice goes to Daniel with P2
    Decision 1.
  - Its certificate is `P2MeetsM1pB19Uncond`: k 88 at 2^19 blocks (8 segments of 2^16), for every payload width in
    range, `p2-2054B`'s among them, with no hypothesis. It is a **proposed** pin, graded PASS but not yet in the
    trusted set. It is an abstract-M1 result, and the SMS-to-M1 instantiation assumption is pending Daniel.
  - SeqRoot sets `call_ns`, half the fastest 16,448-bit root, and no root may fit within `Δ + RTT`.
  - `Scheme.experimental` carries that reason. The verifier audits an experimental scheme only with
    `experimental=True`, and its profile's statement begins with it. No experimental scheme is ever the default.

  P3 has no certificate: the concrete-`H` pins have its parameters, but their chain starts from a different `h_0`
  ("P3 (research)").
- **`protocol.Verifier`** is `audit.TimedVerifier` at the certificate's `k`. It refuses:
  - a `k` below the certificate's;
  - a deadline whose effective D (calls of `call_ns` within `Δ + RTT`) exceeds the certificate's `rounds`: for the
    band and dense, `Δ + RTT` must stay under 1.449 ms. `TimedVerifier` alone refuses only past the band's cliff, at
    about 3.14 ms (3,139,500 ns at `d = 12`);
  - a scheme with no certificate at its parameters.

  A scheme may have several certificates (`Scheme.certificates`), and the verifier audits under the first that
  covers its `k` and deadline. The band's are `BandMultiMeetsFamily` (`k = 111` at `D = 5`, the default), then
  `BandMultiMeetsFamilyD2` (`k = 97` at `D = 2`, `Δ + RTT` under 724.5 µs, `d ≥ 2`) and `BandMultiMeetsFamilyD1`
  (`k = 93` at `D = 1`, under 483 µs, `d ≥ 1`). An unset `k` stays at the first certificate's, so the default audit is
  unchanged; the two smaller `k`s are used only when asked for. A band with `d < 5` has only the last two.

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
  - every coin comes from `verity.randomness` through `setup` and `audit`, and a scheme draws none.
- **Segment tags must be injective across all segments.** This is a hard requirement of every implementation:
  segment `s` is keyed by `tag_s = salt ‖ u64 s` (`labels.tag`), or by one salt with global indices.
  - With a shared domain, identical segments get identical labels. An adversary storing one of them then answers every
    challenge, and the audit accepts with probability 1: Lean's `shared_domain_breaks` makes `Meets` false for every D,
    Q, k and ε.
  - `segDom_inj` is the certificates' hypothesis.
  - `tests/test_pous_protocol.py` checks it for every scheme, and replays the attack on real labels.
  - `benchmarks/pous` gates every implementation's tags against `salt ‖ u64 s` over 448 segments.
- **Claims.** `protocol.claim(id)` resolves an id from `verity.claims`, or from `protocol.LOCAL_CLAIMS` until core holds
  it: `ideal-permutation`, `perfect-isolation`, `sequential-step-floor`, `feistel-indifferentiability`, and P2's
  `m1-ideal-permutations`, `sms-m1-instantiation`, `seq-root`.
- **Tests.** `tests/test_pous_protocol.py` runs one conformance suite over every scheme, plus a toy that implements
  only the two methods. Each registered scheme's vectors are pinned in `tests/fixtures/schemes/<name>.json`, written
  by `tests/pous_scheme_vectors.py`: the certificate at the registered point, and a small instance's segment digests
  and `vk` root. `VERITY_POUS_SLOW=1` also encodes a whole band12 segment through the interface against
  `band/64kb/segment`.

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
| `verifier` | `VerifierKey` (`root` and leaf digests; `build`, `from_leaves`, `from_segments`), `leaf_hash` (whole or chunked blocks), `segment_leaves`, `MerkleTree` (audit paths), `verify_path` |
| `audit` | `challenge_key`, `HonestResponder`, `TimedVerifier` and `run_timed` (wall clock; `respond` may return `(block, report)`, kept in `Answer.report`), `simulate` (oracle rounds); `k` and the adversary's call time default to the parameters' own (band and dense: 111 and 241.5 µs); `TimedVerifier` refuses a band deadline past the cliff |
| `adversary` | holdings (Lean `ColState`), `schedule`, `answer_prog`, strategies, `measure`; `col_round` is Lean's `colRound` |
| `protocol` | the generic protocol: `Scheme` (abstract), `Certificate`, `Verifier` (refuses what none of its certificates covers), `StorageProfile`, `PRIVATE`/`PUBLIC`, `verifier_key`, `commit`, `draw_salt`, `encode`, `decode`, `decode_tensor`, `challenge_key`, `responder`, `claim`, `LOCAL_CLAIMS` |
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

`vk` (`docs/p3-instantiation.md` §4) is a SHA-256 Merkle tree over the blocks, with the shape of RFC 6962. Each leaf
is `SHA256(0x00 || tag_s || u16(v) || c2_v)` and each node is `SHA256(0x01 || l || r)`. The verifier keeps the root and
every leaf digest. A root-only verifier checks the server's audit path. This follows the instantiation's byte format
rather than `verity.commitments`' frame v3.

## Rounds

A program's rounds are counted by `oracle.run`, never claimed. `audit.simulate` sets the round budget per answer at
`(Δ + rtt_allowance) / call_ns`. Here `call_ns` is the adversary's fastest oracle call, and its 242 µs default is
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
- known-answer vectors for `F`, `Π`, `P_T`, `H` and whole segments (`fixtures/vectors.json`; the 64 KB segment with
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
  (`pous_state_game.py`, from `internal/p3-instantiation/overwrite-state-game.py`) agrees on dense and band rows;
- the Lean vectors (`fixtures/lean_vectors.json`).

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

- **Trusted, reviewed once:** `Pous/Protocol/Game` (bits, uniform probability, parallel oracle programs `Prog`),
  `Pous/Protocol/Model` (schemes, the audit game `AuditSecure`, `INC`/`TimedINC`, and the models below),
  `Pous/Protocol/Accounting/{Params,LemmaA}.lean` (`ρ`, `δ`, `Meets`), `Pous/Assumptions.lean` and
  `Pous/Guarantees.lean`.
- **`Meets sch reveal D Q k ε`:** `ε ≤ 2^-128`, `sch` is correct, `|C| ≥ |W|`, the space bound holds, and
  `AuditSecure` holds at a state of `⌊ρ|C|⌋` bits with bound `δ + ε`.
- **Proofs:** `Pous/SecurityProofs/`, checked by Verity's `tools/lean/audit.py` against `lean-audit.json`.
  `Pous/SecurityProofs.lean` proves `Pous.Guarantees.X` as `Pous.SecurityProofs.X`.
- **Untrusted submissions:** `grade.sh FILE Pous.Guarantees.X` grades one file against one guarantee (`lean/README.md`).

### The band

`Pous.BandChain.chainBand B d ℓ c dom` (`Pous/Protocol/Model/BandChain.lean`) is the band over the overwrite chain on
one ideal permutation `Π₂`, and `Pous.BandMulti.chainBandSeg` lays out segments of it. The proofs are in this package
(`Pous/SecurityProofs/Chain/Flat.lean`, from the POUS store's `lean/submissions/band-chain/`, `band-multi/` and
`sponge-dense/`), with no hypothesis: `Pous.SecurityProofs.BandChainMeets64` gives `Meets` at 64 KiB, `B = 512`,
`D = 5`, `Q = 2^20`, `k = 111` and `ε = 2^-128` for every `d ≥ 5`; `BandMultiMeets14` the same for 448 segments
(14 GiB), with per-segment and with global salts, `BandMultiMeets14K106` at `k = 106` per segment, and
`BandMultiMeetsFamily` the `k = 111` row for every segment count `N` with `1 ≤ N ≤ 2^64`
(`Pous/SecurityProofs/Chain/Family.lean`). `BandMultiMeetsFamilyD2` and `BandMultiMeetsFamilyD1` are that family at
fewer rounds, `D = 2` with `k = 97` for every `d ≥ 2` and `D = 1` with `k = 93` for every `d ≥ 1`, and
`BandChainMeets64Rounds2` and `BandChainMeets64Rounds1` their one-segment rows
(`Pous/SecurityProofs/Chain/FamilyRounds.lean`). `BandPubMeets*` are the same rows with an untrusted server encoding and the
exact setup check (`Pous/Protocol/Model/PubEncoder.lean`), under the four deployment conditions in their docstrings;
`BandPubMeetsFamily`, `BandPubMeetsFamilyD2` and `BandPubMeetsFamilyD1` are `BandMultiMeetsFamily`'s and its two rows at
fewer rounds (`Pous/SecurityProofs/Chain/PubFamily.lean`, `PubRounds.lean`). The
band needs `d ≥ D`.

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
hypothesis in `Pous/SecurityProofs/Chain/Flat.lean`): `Pous.SecurityProofs.ChainDenseMeets64`, 512 blocks of 64 KiB at
`D = 5`, `Q = 2^20`, `k = 111`; and the deployment point, `Pous.SecurityProofs.ChainDenseMeets111`, `2^16` blocks of
1 KiB at `D = 64`, `Q = 2^20`, `k = 111`. The deployment was pinned at `k = 107`, which the ex post facto argument does
not reach; it is re-pinned at `k = 111`, four challenges more.

**Not proved here:**
- a concrete permutation for `Π₂` (the main caveat);
- the conversion of the deadline into `D = 5` rounds, which assumes no ASIC and at least 50 ns per Keccak-f, and perfect
  isolation: both outside Lean;
- decode cost, which isn't modeled.

## Guarantees (`lean/Pous/Guarantees.lean`)

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
| `P3ConcreteMeets64`, `P3ConcreteMeets14GB`, `P3ConcreteMeets64D12` | P3 with the overwrite-chain `H` (`Pous/Protocol/Model/P3Chain.lean`), 64 KB labels, `k = 106`: one segment and 971 segments (14.005 GB) on `B(220, 12)` at `D = 10`, and `B(220, 16)` at `D = 12`, in the ideal two-permutation model | proved |
| `BandChainMeets64`, `BandChainMeets64D5`, `BandChainMeets64D12`, `BandMultiMeets14`, `BandMultiMeets14K106`, `BandMultiMeetsFamily` | the band over the overwrite chain (`Pous/Protocol/Model/BandChain.lean`): 64 KiB labels, `D = 5`, `k = 111`, every `d ≥ 5` and `d = 5`, `12`; 448 segments (14 GiB) at `k = 111`, and at `k = 106` per segment; every segment count from 1 to `2^64` at `k = 111` | proved |
| `BandMultiMeetsFamilyD2`, `BandMultiMeetsFamilyD1`, `BandChainMeets64Rounds2`, `BandChainMeets64Rounds1` | the band at fewer rounds: every segment count from 1 to `2^64` at `D = 2`, `k = 97`, every `d ≥ 2`, and at `D = 1`, `k = 93`, every `d ≥ 1`; one segment (64 KiB) at each | proved |
| `BandPubMeets64`, `BandPubMeets14`, `BandPubMeets14Global`, `BandPubMeetsFamily`, `BandPubMeetsFamilyD2`, `BandPubMeetsFamilyD1` | the same band rows with a public encoder and the exact setup check (`Pous/Protocol/Model/PubEncoder.lean`, `Pous/Protocol/Accounting/PubMeets.lean`) | proved |
| `ChainDenseMeets64`, `ChainDenseMeets111` | the dense scheme over the overwrite chain: 64 KiB at `D = 5`, and the deployment point (`2^16` blocks of 1 KiB, `D = 64`), both at `k = 111` | proved |
| `B5Stacked` | the one-way bound on Fisch's stacked graph | proved |
| `DenseMeets64`, `SegMeets14`, `SegTagMeets14` | the dense scheme at 64 KB, `D = 5`, `k = 106`, in the ideal-hash model: one segment, 448 segments (14 GB), and the 448 segments keyed as the implementation keys them, for every salt | proved |
| `SegTagAuditSecure`, `DomainSepEquiv`, `TagInjective14` | the tagged keying's transfer: the tagged scheme is as secure as the global one, and the tags relabel the global indices injectively (448 distinct tags) | proved |
| `BandTwoWay220_12`, `BandTwoWay220_12D11`, `BandTwoWay220_16` | the band certificate for the two-directional column game (`Pous.ColumnGame`): `B(220, 12)` at game depths 10 and 11, and `B(220, 16)` at depth 12, save at most `X = 3` labels | proved |

**Why N10 fails.** Top label `v` reads only column `v` of the lower layer, so an adversary can pay for a lower label a
column at a time, where whole-label pebbling charges it a whole label. N10′, the column-aware restatement over
`Pous.ColumnGame`, is being drafted.

## Named assumptions (`lean/Pous/Assumptions.lean`)

Each is a `Prop` taken as a hypothesis, never an `axiom`.

- `LemmaA sch S D Q g Γ G`: on the good setups `G`, an `S`-bit state and a per-block `(D, Q)` responder are right on at
  least `g` blocks with probability at most `Γ`. For M1 (`G = True`) it is `A2LemmaAM1`.
- `B1Prime sch Mdom S D Q k` (Conjecture B1′): `sch`'s sequential audit obeys model M1's completion bound. Used by
  `M1Meets`.
- Retracted: `InvertibleLabels`, which was N10's statement. It is false (`N10Refuted`).

## Provisional, waiting on Daniel

- `|C| ≥ |W|` in `Meets` (so `|pp| ≤ 0.05·|W|`): otherwise a scheme can keep `W` in the free `pp`.
- `ε_crypto ≤ 2^-128`: an unbounded `ε` makes the requirement vacuous.
- Which reveal a graded `Meets` may use, which waits on the per-answer versus shared deadline.

## Choices that could be wrong

- **H is a monolithic random oracle:** the main caveat, at the top.
- **One segment.** Every model plays one segment against a fresh primitive. A deployment must put a salt and the
  segment index in every oracle query.

## Not here yet

Done, in this package: in the random-oracle model, the dense re-certificate at `POINT_64K` (`dense_meets_64`), the
multi-segment lemma (`seg_meets_14`, 14 GB) and its form for the code's per-segment `tag_s` domains
(`segTag_meets_14`), in `lean/Pous/SecurityProofs/Dense/`. Over the overwrite chain, the ex post facto bounds
`ChainExPostFactoG` and `ChainExPostFacto` (`chainExPostFactoG_v6`, `chainExPostFacto_v6`), and from them the band,
the dense chain scheme and P3 with the concrete `H` ("Guarantees"), in
`lean/Pous/SecurityProofs/Chain/Flat.lean`. Pending:

- P3's certificate: the code's `h_0` and Lean's `iv(domP3 tag L v)` differ ("P3 (research)").
- A GPU codec for the dense scheme (#172's is P3's).
- Native and GPU kernels for encode and decode, and a responder in native code on a high-priority stream.
- The bulk bandwidth audit that would pin storage to HBM rather than the whole server.
- In `lean/`: N10′ (`ColumnAwareDraft`, `ForwardCoreDraft`, `n10prime`; its core `n10prime_core` is open) and
  `p3-meets` (P3 with a random-oracle `H`) come in later PRs.
