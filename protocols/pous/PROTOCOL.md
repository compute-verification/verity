# POUS

A server proves it stores an encoding `C` of the weights `W`. Answering a random block within the deadline requires the
block itself: rebuilding it from a smaller state takes more sequential oracle rounds than the deadline allows.

Three codecs share everything above the scheme (setup, layout, `vk`, audits):

- **The band scheme (`band.BAND12`), the secure-first default** (POUS `docs/band-vs-dense.md`). One layer on the band
  DAG with in-degree 12, Lean's `chainBand` (`lean/submissions/band-chain/BandChainModel.lean`).
- **The dense scheme (`dense.DENSE`), selectable.** One layer on the complete DAG, Lean's `chainDense`
  (`lean/submissions/sponge-dense/SpongeModel.lean` §9). It is the same construction with every earlier block a parent.
- **P3 (`scheme.P3`), the fast research codec.** Two layers with a transpose, Lean's `p3Scheme`
  (`lean/submissions/p3-meets/P3Meets.lean`) over `p3Model'` (`pebbling/ColumnAwareDraft.lean`) and
  `Pous.Model.Invertible`.

The concrete primitives are those of POUS `docs/p3-instantiation.md`. Each set of parameters names its codec
(`params.codec`), and every function that takes a codec defaults to it.

## The band scheme (secure-first)

`band.SECURE_FIRST` is `BandParams(512, 524288, d = 12)`: 512 blocks of 65,536 B per segment (32 MiB), audited with
`k = 111` challenges and a certificate for `D = 5` calls of `Π₂`, each at least 483 µs (`band_meets_64`, for every
`d ≥ 5`, given `ChainExPostFactoG`). Segment `s` is exactly `chainBand B d ℓ 512 (dom tag_s)`. It is the dense scheme
below with one change: block `v` absorbs only its `d` newest parents, `C_{v-1}, ..., C_{max(0, v-d)}` (Lean
`parentsDesc (bandDAG B d) v`), then the pad.

- Block `v` costs `min(v, d) + 1` calls: 6,578 per segment at `d = 12`, 20 times fewer than dense's 131,328. Decoding
  is `d + 1 = 13` rounds deep, and a segment decodes in 24.6 s on one core in Python (dense: 493.9 s).
- **Deployment condition, the cliff.** The effective `D`, the sequential `Π₂` calls an adversary completes within
  `Δ + RTT`, must not exceed `d`. With `d ≥ D′` the band is exactly as hard as dense at `D′` (`band_chain_hard`). At
  `D′ = d + 1` the gap attack (`band_gap_attack`, `band_cliff_d12`) holds every block but each `(d + 1)`-th: every
  dropped row has all its parents held, and the audit accepts with probability 1. At 483 µs per call, `d = 12` holds
  while `Δ + RTT < 6.279` ms. `audit.TimedVerifier` refuses a deadline past it (`band.check_deployment`), and
  `test_gap_attack_at_the_cliff` replays the attack on real labels.
- `band.codec_for(d)` gives the codec for any in-degree; with `d ≥ B − 1` the band is the complete DAG, dense.

## The dense scheme (selectable)

`dense.POINT_64K` has `B = 512` blocks of `ℓ = 65,536` B (524,288 bits) per segment, so a segment is 32 MiB: the
point `dense_meets_64` certifies. P3 keeps 65,560 B, the multiple of `n = 220` it needs. The audit runs with
`k = 111` challenges and a certificate for `D = 5` calls of `Π₂`, each at least 483 µs. The 111 is what
`chain_meets_64` (POUS `lean/submissions/sponge-dense/ChainEPF.lean`) certifies for this overwrite-chain `H`, given
`ChainExPostFacto`; a random-oracle `H` needs only 106 (`dense_meets_64`). Segment `s` is exactly
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
   - the band: `band_pub_meets_64` (`band-chain/BandPubEncoder.lean`), for every `d ≥ 5`, given `ChainExPostFactoG`;
   - dense: `dense_pub_meets_64`, `seg_pub_meets_14`, and `chain_pub_meets_64`, the last given `ChainExPostFacto`.

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
  - `p3/v1`.
- **`protocol.Certificate`** is what an audit at the scheme's parameters establishes, Lean's
  `Meets S .sequential D Q k ε`. It holds:
  - `k`, `rounds` (D), `work` (Q), `call_ns`, `rho`, `delta`, `eps_log2`;
  - `cliff`: the most rounds the structure tolerates at any `k`;
  - `segments`: how many segments of `W` the theorem is about;
  - `theorem`, and its open `hypotheses`;
  - `claims`: ids in `verity.claims`' format.

  At the frozen point, the band is `band_meets_64` given `ChainExPostFactoG`: k 111, D 5, Q 2^20, 483 µs, cliff at
  `d`, one segment. Dense is `chain_meets_64` given `ChainExPostFacto`, with no cliff. P3 has no certificate for the
  implemented `H`. Lean's `p3_meets_64_D10_final` and `p3_meets_multi_64_D10_final` (k 117, D 10, 64 KB and 971
  segments) are about a random-oracle `H` and an ideal tweakable `P`, while the code's `H` is the overwrite chain.
- **`protocol.Verifier`** is `audit.TimedVerifier` at the certificate's `k`. It refuses:
  - a `k` below the certificate's;
  - a deadline whose effective D (calls of `call_ns` within `Δ + RTT`) exceeds the certificate's `rounds`: for the
    band and dense, `Δ + RTT` must stay under 2.898 ms. `TimedVerifier` alone refuses only past the band's cliff, at
    6.279 ms;
  - a scheme with no certificate at its parameters.

  `research=True` runs such an audit anyway, except past the cliff.
- **`protocol.StorageProfile`** is what an accepted audit establishes: a server whose state held fewer than
  `⌊rho · |C|⌋` bits passes with probability at most `delta + 2^eps_log2`, by the certificate. It also lists the
  `gaps`, where the certificate does not cover the audit: a refusal run with `research=True`, or more segments of `W`
  than the theorem is about. The headline `band_meets_64` is about one segment; the 14 GB multi-segment form is the
  next item. `to_json` writes `verity-pous/storage-profile/v1`.
- **The rest:**
  - `protocol.verifier_key(scheme, commitment, salt, mode=PRIVATE or PUBLIC, w=..., encoded=...)` builds `vk`
    either way;
  - `commit`, `draw_salt`, `encode`, `decode`, `decode_tensor`, `challenge_key` and `responder` wrap the modules
    below;
  - every coin comes from `verity.randomness` through `setup` and `audit`, and a scheme draws none.
- **Claims.** `protocol.claim(id)` resolves an id from `verity.claims`, or from `protocol.LOCAL_CLAIMS` until core holds
  it: `ideal-permutation`, `perfect-isolation`, `sequential-step-floor`, `feistel-indifferentiability`.
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
| `audit` | `challenge_key`, `HonestResponder`, `TimedVerifier` and `run_timed` (wall clock; `respond` may return `(block, report)`, kept in `Answer.report`), `simulate` (oracle rounds); `k` and the adversary's call time default to the parameters' own (band and dense: 111 and 483 µs); `TimedVerifier` refuses a band deadline past the cliff |
| `adversary` | holdings (Lean `ColState`), `schedule`, `answer_prog`, strategies, `measure`; `col_round` is Lean's `colRound` |
| `protocol` | the generic protocol: `Scheme` (abstract), `Certificate`, `Verifier` (refuses what its certificate doesn't cover), `StorageProfile`, `PRIVATE`/`PUBLIC`, `verifier_key`, `commit`, `draw_salt`, `encode`, `decode`, `decode_tensor`, `challenge_key`, `responder`, `claim`, `LOCAL_CLAIMS` |
| `schemes` | `band.Band`, `dense.Dense`, `p3.P3` (one module each) as `protocol.Scheme`s with their certificates; `REGISTRY`, `DEFAULT` (`band-chain/d12/v1`), `get(name, segments, params)` |

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
`CERTIFIED_ROUNDS = 28` is the most the band certificate `band_twoWay_220_12` covers when a level costs 3 calls
(`Π₂` of the oldest parent, `Π₂` of the pad, `P`), `⌊(D − 1)/3⌋ + 1 ≤ 10`.

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

## Not here yet

Done in the random-oracle model (POUS `lean/submissions/dense/DenseReCert.lean`): the dense re-certificate at
`POINT_64K` (`dense_meets_64`) and the multi-segment lemma (`seg_meets_14`, 14 GB), which covers the code's
per-segment `tag_s` domains (`domainSep_equiv`). The band's statements are proved (`lean/submissions/band-chain/`,
red-team §39), and the red team is verifying the proofs (§40). Pending:

- `ChainExPostFactoG`, the hypothesis of `band_meets_64` (the band at `k = 111`), and its complete-DAG form
  `ChainExPostFacto`, the hypothesis of `chain_meets_64` (dense). Both carry the certificate from the random oracle to
  the overwrite chain (`DISCREPANCIES.md` D5, B1).
- A GPU codec for the dense scheme (#172's is P3's).
- Native and GPU kernels for encode and decode, and a responder in native code on a high-priority stream.
- A restated Lean model: two ideal permutations (`Π₂`, a tweakable `P`), `H` as the overwrite chain, levels charged at
  3 calls, and `n10prime` over all segments (`DISCREPANCIES.md`).
- The bulk bandwidth audit that would pin storage to HBM rather than the whole server.
