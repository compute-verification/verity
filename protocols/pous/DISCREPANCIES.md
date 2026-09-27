# Where the Lean model, the prose spec and this code differ

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
| 5 | `H` | a monolithic random oracle `KQ' → Lab` | a sponge over an ideal permutation `Π` of `m + 512` bits (§2c, instantiation) | pluggable; the default is the sponge. Its header encodes only `npar`, so it is injective on the scheme's queries (`parentsOf`), not on every partial assignment in `KQ'` |
| 6 | `P` | one untweaked ideal permutation `ω.2` | tweakable `P_T`, `T = (tag_s, layer, v)` (instantiation) | `P_T`; the Lean cross-check uses a toy that ignores `T` |
| 7 | Decode rounds | `decodeW`: key and `P⁻¹` independent, so one round each pass with a monolithic `H` | "key, then `P⁻¹`" (§1) | key and `P⁻¹` side by side; a sponge key takes `npar + 2` rounds |
| 8 | Rounds per level | `n10prime` assumes `ColumnHard G ⌊(D + 1)/2⌋` (2 rounds per level) | `⌊(D − 1)/13⌋ + 1` (§2c); 3 calls per level at worst, `⌊(D − 1)/3⌋ + 1` (instantiation §1) | rounds measured exactly per derivation; `CERTIFIED_ROUNDS = 28` uses the instantiation's rule |
| 9 | The column game's units | `colRound` charges one game round each for a reverse and for a promotion | — | an oracle adversary pays 1 round for a reverse (with a monolithic `H`) and 0 for a promotion. See below |
| 10 | Band graph at `k = 0` | `bandGraph n 0` keeps the spine parent `v − 1` | `par(v) = {v − k, …, v − 1}`, empty at `k = 0` | Lean's; the same for `k ≥ 1` |
| 11 | Verifier | the audit compares answers with the true blocks ("idealizes the per-block tags") | per-block tags; a SHA-256 Merkle root (instantiation §4) | the instantiation's tree (RFC 6962 shape) with leaf digests |
| 12 | Audit parameters | `p3_meets`: `D = 20`, `Q = 2^20`, `k = 119` (relies on the `pos = 0` collapse) or `185` (no-mask charge) | `k = 423`/`229` (§2a); `k = 132`, `Δ = 300` µs (§2c); `Δ = 0.5` ms plus RTT (`p3-gpu-measurements.md`) | `k = 132`, `Δ = 0.5` ms, `rtt_allowance_ns = 0`, `Q = 2^14` |
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
- With the instantiation's sponge, a reverse costs at least `npar + 2` rounds. Whether a gap remains under
  `⌊(D − 1)/3⌋ + 1` has not been checked.

**What the Lean cross-check covers.** Lean evaluated `c1'`, `topLabel'`, `transpose`, `decodeLower`, `tagOf`, the
decoder, `bandGraph` and `colRound` for `n ≤ 3` (encode) and `n ≤ 6` (game), and the code matches on every value. The
`rfl` theorems in `tests/lean/PousRefCheck.lean` tie the generic definitions it evaluates to `P3Meets`' own at
`n = 220`. Items 3, 5 and 6 are outside the model, so the check plugs in a monolithic toy `H` and a tweak-ignoring
toy `P`.
