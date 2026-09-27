# Where the Lean models, the prose and this code differ

## The dense scheme (`dense.py`, Lean `chainDense`)

The Lean model is `chainDense` (POUS `lean/submissions/sponge-dense/SpongeModel.lean` §4 and §9). The prose is POUS
`docs/scheme-choice-e2e.md` and the `H` of `docs/p3-instantiation.md` §1. The code follows Lean.

| # | Topic | Lean model | Prose | This code |
|---|---|---|---|---|
| D1 | `H`'s initial state | `h_0 = iv(dom i) = 0^ℓ ‖ salt ‖ i`: the domain fills the 512-bit capacity part, no header, no call | the memo takes `H` from the instantiation §1, whose `h_0` is the 48-byte header `"pous-p3-v1-H" ‖ tag_s ‖ u8(layer) ‖ u16(v) ‖ u8(npar)` in the low bytes, zero-filled | Lean's `iv(dom)`, kept. The instantiation's `u8(npar)` also cannot count the dense scheme's up to 511 parents. **Follow-up:** the header in `docs/p3-instantiation.md` §1 needs a dense-scheme variant, or a note that the dense scheme uses `iv(dom)` |
| D2 | Salt and segments | `chainDense`: one scheme of `B` blocks, `dom salt i` with a 256-bit salt; `ChainDenseMeets` quantifies over every salt. The multi-segment lemma (`lean/submissions/dense/DenseReCert.lean`: `seg_hard`, `segScheme`, `seg_meets_14`) is in the random-oracle model and keys **one** oracle by the **global** block index: block `i` of segment `s` is node `s·B + i`, with parents the earlier blocks of its own segment | 24-byte salt drawn by the verifier, many segments | segment `s` is `chainDense B ℓ 512 (dom tag_s)` with `tag_s = salt ‖ u64(s)` as Lean's salt and the **local** index `i`, so each segment is an instance of the single-segment target. This per-segment tag does not yet match the lemma's global indexing; the pebbling prover is aligning the two. The code is unchanged |
| D3 | `pp` and coins | `t = 0`, `Coins = Unit`: the salt is a parameter, not coins | "no `pp`" | the salt is published beside `C` (24 B) and drawn after the commitment to `W`, as in `setup`; in the model that is a fixed parameter chosen after `W`, which the `∀ salt` target covers |
| D4 | Pinned numbers | `ChainDenseMeetsPinned`: `B = 2^16`, `ℓ = 8192` bits, `D = 64`, `Q = 2^20`, `k = 107`. **Re-certified** at the memo's point in `lean/submissions/dense/DenseReCert.lean`: `dense_meets_64` (`B = 512`, `ℓ = 524288` bits, `D = 5`, `Q = 2^20`, `k = 106`) and `seg_meets_14` (14 GB of such segments), confirmed by the red team (§30). Both are for the random-oracle `denseScheme`, not `chainDense` | `B = 512`, 64 KB labels (`ℓ = 65,560` B, P3's width), `D = 5`, `k = 106` | the certificate's point: `SECURE_FIRST` has `ℓ = 524288` bits (65,536 B), not the memo's 65,560 B, which only P3 needs. One gap remains: the certificate covers the random-oracle `H`, so carrying it to the overwrite chain needs D5's `ChainExPostFacto` |
| D5 | The chain's security | `ChainExPostFacto` is a target (not proved); `chainHard_of_pebblingHard` is proved | "in progress" | nothing here depends on it; the two-for-one tests exercise the same chaining mode on P3's rows, not on the dense rows |

## P3 (`scheme.py`, Lean `p3Scheme`)

The Lean model is `p3Scheme` (POUS `lean/submissions/p3-meets/P3Meets.lean`) over `p3Model'`
(`pebbling/ColumnAwareDraft.lean`) and `Pous.Model.Invertible`. The prose is POUS `docs/p3-scheme.md` (§1, §2c) and
`docs/p3-instantiation.md`. Where they disagree, the code follows Lean, except where the instantiation deliberately
changes the scheme (items 3, 5 and 6), which the Lean model has yet to absorb.

| # | Topic | Lean model | Prose | This code |
|---|---|---|---|---|
| 1 | Label width | `Lab n wc`, so `n` divides `m`; `p3Scheme` has `m = 220 · 37 = 8140` bits | `m = 8192` with `w_c = m/n ≈ 37` (§1); "64 KB", `m/n ≈ 2380` (§2c); `m = 65,560` B, `w_c = 2384` bits (instantiation) | `Params(n, wc, k)` with `m = n · wc`; default `wc = 2384` (65,560 B); `LEAN_P3_MEETS` has `wc = 37` |
| 2 | Layer indices | `Fin 2`: 0 lower, 1 top | 1 and 2 | 0 and 1 |
| 3 | Key tag and salt | `KQ'` carries `Bits 256`; `tagOf` puts a 128-bit salt in bits 0..127, zeros above (segment 0 only) | `(salt, s)` as fields (§1); `tag_s = salt (24 B) ‖ u64(s)` (instantiation) | the instantiation's; for segment 0 with a salt under 2^128 it equals `tagOf` |
| 4 | Segments | `p3Scheme` is one segment of 220 labels; `n10prime` covers one segment | many segments per `W` | `N` segments, segment-major in `W` and `C`; `N = 1` is Lean's scheme |
| 5 | `H` | a monolithic random oracle `KQ' → Lab`; the pilot `p3SpongeModel` (`sponge-dense`) has the XOR sponge over `Π` | a sponge over `Π` of `m + 512` bits (§2c), then an overwrite chain over `Π₂` of `2m + 512` bits (instantiation, 16:45Z) | pluggable; the default is the overwrite chain, and the broken XOR sponge and Miyaguchi–Preneel are kept for attack replay. The header encodes only `npar`, so `H` is injective on the scheme's queries (`parentsOf`), not on every partial assignment in `KQ'` |
| 6 | `P` | one untweaked ideal permutation `ω.2` | tweakable `P_T`, `T = (tag_s, layer, v)` (instantiation) | `P_T`; the Lean cross-check uses a toy that ignores `T` |
| 7 | Decode rounds | `decodeW`: key and `P⁻¹` independent, so one round each pass with a monolithic `H` | "key, then `P⁻¹`" (§1) | key and `P⁻¹` side by side; an overwrite-chain key takes `npar + 1` rounds |
| 8 | Rounds per level | `n10prime` assumes `ColumnHard G ⌊(D + 1)/2⌋` (2 rounds per level) | `⌊(D − 1)/13⌋ + 1` (§2c); 3 calls per level at worst, `⌊(D − 1)/3⌋ + 1` (instantiation §1) | rounds measured exactly per derivation; `CERTIFIED_ROUNDS = 28` uses the instantiation's rule |
| 9 | The column game's units | `colRound` charges one game round each for a reverse and for a promotion | — | an oracle adversary pays 1 round for a reverse (with a monolithic `H`) and 0 for a promotion. See below |
| 10 | Band graph at `k = 0` | `bandGraph n 0` keeps the spine parent `v − 1` | `par(v) = {v − k, …, v − 1}`, empty at `k = 0` | Lean's; the same for `k ≥ 1` |
| 11 | Verifier | the audit compares answers with the true blocks ("idealizes the per-block tags") | per-block tags; a SHA-256 Merkle root (instantiation §4) | the instantiation's tree (RFC 6962 shape) with leaf digests |
| 12 | Audit parameters | `p3_meets`: `D = 20`, `Q = 2^20`, `k = 119` (relies on the `pos = 0` collapse) or `185` (no-mask charge) | `k = 423`/`229` (§2a); `k = 132`, `Δ = 300` µs (§2c); `Δ = 0.5` ms plus RTT (`p3-gpu-measurements.md`); `k = 117` at `D = 10` (`scheme-choice-e2e.md`) | `k = 132`, `Δ = 0.5` ms, `rtt_allowance_ns = 0`, `Q = 2^14`, `CERTIFIED_ROUNDS = 28` |
| 13 | Key inputs | full parent labels (P3.0) | P3.1 digest keys, later dropped | P3.0 |
| 14 | Encoder and coins | `enc W ω salt`, with the salt uniform in `Coins` after `W` | the server encodes after committing to `W`; the verifier draws the salt and computes `vk` itself (§2c) | the prose's protocol (`setup`), which matches Lean's order when the commitment precedes the salt |
| 15 | Bytes | `Bits` via `codeEquiv` | little-endian integers (instantiation) | consistent: bit `i` of a string is bit `i mod 8` of byte `⌊i/8⌋` |

**Item 9, the column game's units.** `n10prime` grants `⌊(D + 1)/2⌋` game rounds for `D` oracle rounds: two rounds per
level, and one for a top whose key is ready. A meet-in-the-middle holding gets more than that from the oracle model.
It reverses held tops, promotes lower labels from the revealed columns, and rebuilds forward from them. With a
monolithic `H`, the reverse costs 1 oracle round and the promotion none, while `colRound` charges a round for each.

- `tests/test_pous_adversary.py` pins a witness at `n = 12`, `k = 2`: every top is recovered bit-exactly within
  `D = 20` oracle rounds, but the game needs 12 rounds, and N10′ grants 10.
- At `B(220, 12)` with a monolithic `H` there is a holding that saves 2.29 labels and answers all 220 challenges within
  20 oracle rounds, using 211 `P⁻¹` queries. The game needs 12 rounds for it. That is under the certified `X = 3`, but
  it shows the certificate is needed at 12 game rounds, while `band_twoWay_220_12` is proved at 10 and 11.
- With the instantiation's overwrite chain, a reverse costs at least `npar + 1` rounds, since it needs the top's key.
  Whether a gap remains under `⌊(D − 1)/3⌋ + 1` has not been checked.

**Item 5, the storage breaks of the earlier `H`.** The pilot proved that the XOR sponge's pre-squeeze states are not
full-price pebbles once `Π⁻¹` is queryable. With a public first state, one stored state regenerates two labels
(`dense_two_labels_from_one_state`). The model's B5 freshness step therefore fails for it. `tests/test_pous_two_for_one.py`
replays that attack, and the Miyaguchi–Preneel pre-image attack on band rows, on real labels, and shows both fail on the
overwrite chain. A Lean statement for the overwrite chain (the two-permutation model with `Π₂`) is not written yet.

**What the Lean cross-check covers.** Lean evaluated `c1'`, `topLabel'`, `transpose`, `decodeLower`, `tagOf`, the
decoder, `bandGraph` and `colRound` for `n ≤ 3` (encode) and `n ≤ 6` (game), and the code matches on every value. The
`rfl` theorems in `tests/lean/PousRefCheck.lean` tie the generic definitions it evaluates to `P3Meets`' own at
`n = 220`. Items 3, 5 and 6 are outside the model, so the check plugs in a monolithic toy `H` and a tweak-ignoring
toy `P`.
