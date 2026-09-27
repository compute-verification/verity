# POUS P3

A server proves it stores an encoding `C` of the weights `W`. Answering a random block within the deadline requires the
block itself: rebuilding it from a smaller state takes more sequential oracle rounds than the deadline allows. The Lean
model is `p3Scheme` (`lean/submissions/p3-meets/P3Meets.lean` of the POUS project) over `p3Model'`
(`pebbling/ColumnAwareDraft.lean`) and `Pous.Model.Invertible`. The concrete primitives are those of the POUS project's
`docs/p3-instantiation.md`.

## The scheme

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
3. **Encode.** The server computes `C = Enc(W, salt)` (`scheme.encode`). The verifier computes `vk` from its *own*
   encoding (`setup.verifier_key`) and never accepts it from the server. Comparing roots (`setup.check_server_root`)
   catches a wrong encoding at setup. The encoder is deterministic and public, so the server running it changes
   nothing in the timed game. A server-chosen or predictable salt would break the model's "`W` before coins" order.
4. **Serve.** `layout.decode_tensor` decodes one tensor from the segments it overlaps.
5. **Audit.** The verifier reveals `k` indices one at a time (`audit.TimedVerifier`, uniform with replacement, keyed
   by `audit.challenge_key` and bound to `vk`'s root). Each block must arrive within `Δ = 0.5 ms` plus
   `rtt_allowance_ns`, and must match `vk`. The audit accepts only if every answer is correct and on time. At 300 µs
   about 1.5% of honest 132-answer audits fail on the host clock (POUS `docs/p3-gpu-measurements.md`). The network
   allowance defaults to 0 until the verifier's RTT is measured.

## API

| Module | What it holds |
|---|---|
| `params` | `Params(n, wc, k, segments)`, `OPERATING_POINT`, `LEAN_P3_MEETS` |
| `labels` | the bit layout: Lean's `codeEquiv` (bit `j` of block `i` at `i · width + j`), little-endian bytes; `transpose`; `tag` |
| `oracle` | queries (`HQuery`, `PQuery`/`PinvQuery` with a `Tweak`, `PiQuery`/`PiInvQuery`), programs (`Prog`), `run` with Lean's depth/work/width, `parallel` |
| `primitives` | `KeyDerivation` and `TweakablePermutation` interfaces; `OverwriteChain` (the default `H`), `RandomOracleKey`, and the broken `XorSponge` and `MiyaguchiPreneel`; the backends `reference`, `heuristic_arx`, `ideal` |
| `scheme` | `encode(params, prims, W, salt) -> Encoding` (`C`), `decode`, `decode_segment`, and their programs |
| `layout` | `Layout.of(params, [(name, size)], align)`, `pack`, `decode_tensor` |
| `setup` | `commit`, `draw_salt`, `verifier_key`, `server_root`, `check_server_root` |
| `verifier` | `VerifierKey` (`root` and leaf digests), `MerkleTree` (audit paths), `verify_path` |
| `audit` | `challenge_key`, `HonestResponder`, `TimedVerifier` and `run_timed` (wall clock), `simulate` (oracle rounds) |
| `adversary` | holdings (Lean `ColState`), `schedule`, `answer_prog`, strategies, `measure`; `col_round` is Lean's `colRound` |

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
- the two-labels-from-one-item attacks (`test_pous_two_for_one.py`), bit-exact on real labels: they succeed on the XOR
  sponge and on Miyaguchi–Preneel over band rows, and fail on the overwrite chain. The designer's state game
  (`pous_state_game.py`, from `internal/p3-instantiation/overwrite-state-game.py`) agrees on dense and band rows;
- the Lean vectors (`fixtures/lean_vectors.json`).

The Lean vectors come from `tests/lean/PousRefCheck.lean`, which evaluates Lean's `c1'`, `topLabel'`, `transpose`,
`decodeLower`, `tagOf`, the decoder, `bandGraph` and `colRound` on small cases. It proves by `rfl` that its generic
`decodeW'`, `Wlab'`, `enc` and `dec` are `P3Meets`' at `n = 220`. Lean evaluates labels lazily, so its encode cases
stop at `n = 3`.

## Not here yet

- Native and GPU kernels for encode and decode, and a responder in native code on a high-priority stream.
- A restated Lean model: two ideal permutations (`Π₂`, a tweakable `P`), `H` as the overwrite chain, levels charged at
  3 calls, and `n10prime` over all segments (`DISCREPANCIES.md`).
- The bulk bandwidth audit that would pin storage to HBM rather than the whole server.
