# Where the Lean models, the prose and this code differ

## The band scheme (`band.py`, Lean `chainBand`)

The Lean model is `chainBand` (POUS `lean/submissions/band-chain/BandChainModel.lean`, sha256 `ba654b0f…`), the prose
POUS `docs/band-vs-dense.md`. The band is `chainScheme (bandDAG B d)` with the dense chain's `H`, `dom` and `iv`, so D1
(the instantiation header), D2 (the segment tags) and D3 (the salt) apply unchanged.

| # | Topic | Lean model | Prose | This code |
|---|---|---|---|---|
| B1 | Security | `Pous.SecurityProofs.BandChainMeets64` (proved in `Security/Proofs/Pous/Chain/Flat.lean`, no hypothesis): `Meets` at `B = 512`, `ℓ = 524288` bits, `D = 5`, `Q = 2^20`, `k = 111` for every `d ≥ 5`; `ChainExPostFactoG` is proved (`chainExPostFactoG_v6`, red team §52) | the same, "recommend `d = 12`" | `band.SECURE_FIRST` is that point at `d = 12` (`BandChainMeets64D12`). The code's certificate is `Pous.SecurityProofs.BandMultiMeetsFamily`, with no hypothesis, which covers every segment count from 1 to `2^64` with the code's per-segment tags (`segDom`) |
| B2 | The cliff | `band_chain_hard` needs `D ≤ d`; `band_gap_attack` and `band_cliff_d12` show the bound fails at `D = d + 1` | "the effective `D` must not exceed `d`" | enforced where it can be: `UnsignedRounds` refuses a deadline whose effective `D` (calls of at least 241.5 µs within `Δ + RTT`) exceeds `d`, and `protocol.TimedAudit` also refuses one past the certificate's `D = 5` (`Δ + RTT ≥ 1.449` ms) and a `k` below 111. The 241.5 µs floor itself is a timing assumption the code cannot check. `simulate` does not refuse, so the attack can be replayed |
| B3 | Lean-evaluated vectors | — | — | Lean evaluated the band's parents for five shapes, but labels only for `d = 1` (a `d = 2` case exceeds the evaluation budget). The larger band structure is checked against the Lean text in Python |
| B4 | Public encoder | **Proved and pinned** (`Pous.SecurityProofs.BandPubMeets64`, `BandPubMeets14`, `BandPubMeets14Global`, no hypothesis; red team §§53, 55; and `BandPubMeetsFamily`, every segment count from 1 to `2^64`, from `BandMultiMeetsFamily`, with `BandPubMeetsFamilyD2` and `BandPubMeetsFamilyD1` at `D = 2` and `D = 1` from `BandMultiMeetsFamilyD2` and `BandMultiMeetsFamilyD1`). `chainBand_encDec` (`band-chain/BandEncDec.lean`) through the DAG-generic `chainScheme_encDec` (`labelG_dec`, the port of `labelOv_dec` to `labelG`); `band_pub_meets_64` (`BandPubEncoder.lean`) gives `PubMeets` with the exact check at `k = 111` for every `d ≥ 5`, from `band_meets_64` | the band is the codec milestone 1 runs on | matches: the decode check is `exactCheck`'s. `PousBandCheck.lean` imports both files and pins `chainBand_encDec` and `band_pub_meets_64_d12` at the secure-first point; `test_enc_dec_matches_lean` matches Lean's encode after decode on codes that are not encodings, and `test_enc_dec` checks it at larger sizes |
| B5 | Late answers (grace `t`) | **Proved and pinned** (`Pous.SecurityProofs.BandGradedMeetsFamily`, no hypothesis): the graded audit (`Security/Definitions/Pous/Model/Graded.lean`) accepts up to `t` late answers, each within a hard `(D_late, Q_late)` with `5 ≤ D_late ≤ d` and `D_late ≤ 12`, at `(t, k) = (0, 111), (1, 160), (2, 203), (3, 243)` | `PROTOCOL.md`, "The graded audit": no pipelining, a hard deadline `Δ_late`, every answer past `Δ` counted late, isolation over the whole window | **not implemented**: `AuditResult.accepted` requires every answer correct and on time, so `UnsignedRounds` and `protocol.TimedAudit` are the `t = 0` rule. A graded memory challenger must count `elapsed > Δ + RTT` as late, reject past `Δ_late + RTT`, refuse a `Δ_late` past `D_late ≤ min(d, 12)` and any `(k, t)` outside the certificate |
| B6 | The concrete `Π₂` (band, dense and P3's key alike) | `Π₂` is an ideal permutation, and a step is one round of whole calls; `Meets`, `GradedMeets`, `ContinuousAudit`, `BandPub*` and `P3Concrete*` are sound in that model | `feistel-indifferentiability`: Feistel10 over SHAKE256 in place of the ideal permutation keeps the storage bound; a deadline maps to whole calls of 241.5 µs | **broken** (red team, 4 Oct 2026, art:7617c8017e80): every chain key is the first `m` bits of `pad ⊕ F_1 ⊕ F_3 ⊕ F_5 ⊕ F_7 ⊕ F_9`, each `F_i` a SHAKE256 squeeze from a 200-byte post-absorb state, so a prover that keeps the five states per 64 KiB block (1000 B, 1.53% of `C`) rebuilds any label in 481 sequential Keccak-f (24 µs at 50 ns), with no call and no chain walk, bit-exact against `band-chain/d12/v1` and `dense-chain/v1`; P3's key floor breaks identically. `feistel-indifferentiability` is refuted, and `call_ns`, the effective `D`, the cliff's 3.14 ms, `LATE_ROUNDS` and `AUDIT_CAP` bound an adversary only in the model. Closing it takes a `Π₂` whose output does not split into resumable squeeze streams, or a model that charges partial evaluation |

## The dense scheme (`dense.py`, Lean `chainDense`)

The Lean model is `chainDense` (POUS `lean/submissions/sponge-dense/SpongeModel.lean` §4 and §9). The prose is POUS
`docs/scheme-choice-e2e.md` and the `H` of `docs/p3-instantiation.md` §1. The code follows Lean.

| # | Topic | Lean model | Prose | This code |
|---|---|---|---|---|
| D1 | `H`'s initial state | `h_0 = iv(dom i) = 0^ℓ ‖ salt ‖ i`: the domain fills the 512-bit capacity part, no header, no call | the memo takes `H` from the instantiation §1, whose `h_0` is the 48-byte header `"pous-p3-v1-H" ‖ tag_s ‖ u8(layer) ‖ u16(v) ‖ u8(npar)` in the low bytes, zero-filled | Lean's `iv(dom)`, kept. The instantiation's `u8(npar)` also cannot count the dense scheme's up to 511 parents. **Follow-up:** the header in `docs/p3-instantiation.md` §1 needs a dense-scheme variant, or a note that the dense scheme uses `iv(dom)` |
| D2 | Salt and segments | `chainDense`: one scheme of `B` blocks, `dom salt i` with a 256-bit salt; `ChainDenseMeets` quantifies over every salt. The multi-segment lemma (`lean/submissions/dense/DenseReCert.lean`: `seg_hard`, `segScheme`, `seg_meets_14`) is in the random-oracle model and keys **one** oracle by the **global** block index: block `i` of segment `s` is node `s·B + i`, with parents the earlier blocks of its own segment | 24-byte salt sampled by the verifier, many segments | segment `s` is `chainDense B ℓ 512 (dom tag_s)` with `tag_s = salt ‖ u64(s)` as Lean's salt and the **local** index `i`. **Closed:** `domainSep_equiv` (same file, sha256 `7d3db033…`) proves that keying by `tag_s` with a local index separates blocks exactly as the global index does, for `N ≤ 2^64`, and `tag_injective_14` witnesses it at 14 GB, so `seg_meets_14` covers the code's per-segment domains. Lean's `tag` is a 256-bit salt ‖ `u64 s` (320 bits); the code's is a 192-bit salt ‖ `u64(s)` (256 bits, `dom`'s salt slot). Both rest on the same injectivity in `s` |
| D3 | `pp` and coins | `t = 0`, `Coins = Unit`: the salt is a parameter, not coins | "no `pp`" | the salt is published beside `C` (24 B) and sampled after the commitment to `W`, as in `setup`; in the model that is a fixed parameter chosen after `W`, which the `∀ salt` target covers |
| D4 | Pinned numbers | **Re-pinned at `k = 111`** (29 Sep 2026): `Pous.SecurityProofs.ChainDenseMeets111`, proved, is `chainDense` at `B = 2^16`, `ℓ = 8192` bits, `D = 64`, `Q = 2^20`, `k = 111`. The lane's `ChainDenseMeetsPinned` at `k = 107` stays an open target, which the ex post facto argument does not reach. The random-oracle `denseScheme` is **re-certified** at the memo's point in `lean/submissions/dense/DenseReCert.lean`: `dense_meets_64` (`B = 512`, `ℓ = 524288` bits, `D = 5`, `Q = 2^20`, `k = 106`) and `seg_meets_14` (14 GB of such segments), confirmed by the red team (§30). Both are for the random-oracle `denseScheme`, not `chainDense` | `B = 512`, 64 KB labels (`ℓ = 65,560` B, P3's width), `D = 5`, `k = 106` | the certificate's point: `SECURE_FIRST` has `ℓ = 524288` bits (65,536 B), not the memo's 65,560 B, which only P3 needs, and **`k = 111`**, not the memo's 106. The 106 holds only for a random-oracle `H`; over the overwrite chain the code uses, `chain_meets_64` (`lean/submissions/sponge-dense/ChainEPF.lean`) certifies `k = 111` at the same `B`, `ℓ`, `D = 5`, `Q = 2^20`, now with no hypothesis (`Pous.SecurityProofs.ChainDenseMeets64`) |
| D5 | The chain's security | `ChainExPostFacto` is proved (`chainExPostFacto_v6`, `Security/Proofs/Pous/Chain/Flat.lean`; red team §52); `chainHard_of_pebblingHard` is proved; `chain_meets_64` derives `Meets` at `k = 111` from it (`Pous.SecurityProofs.ChainDenseMeets64`) | "in progress" | `POINT_64K`'s `k = 111` is `ChainDenseMeets64`'s, the certificate's theorem, with no hypothesis; the two-for-one tests exercise the same chaining mode on P3's rows, not on the dense rows |

## P3 (`scheme.py`, Lean `p3Scheme`)

The Lean model is `p3Scheme` (POUS `lean/submissions/p3-meets/P3Meets.lean`) over `p3Model'`
(`pebbling/ColumnAwareDraft.lean`) and `Definitions.Pous.Model.Invertible`. The prose is POUS `docs/p3-scheme.md` (§1,
§2c) and `docs/p3-instantiation.md`. Where they disagree, the code follows Lean, except where the instantiation
deliberately changes the scheme (items 3, 5 and 6). `p3Scheme` has not absorbed those changes, but the concrete-`H`
model has: `Pous.P3Concrete.p3SchemeC` and `p3SchemeNC` over `Pous.Sponge.p3ChainModel`
(`Security/Definitions/Pous/Model/P3Chain.lean`). The pins `P3ConcreteMeets64`, `P3ConcreteMeets14GB` and
`P3ConcreteMeets64D12` are proved over it with no hypothesis (`Security/Proofs/Pous/Chain/Flat.lean`), at the
code's parameters, but their chain starts from another `h_0` (item 16), so P3 has no certificate.

| # | Topic | Lean model | Prose | This code |
|---|---|---|---|---|
| 1 | Label width | `Lab n wc`, so `n` divides `m`; `p3Scheme` has `m = 220 · 37 = 8140` bits | `m = 8192` with `w_c = m/n ≈ 37` (§1); "64 KB", `m/n ≈ 2380` (§2c); `m = 65,560` B, `w_c = 2384` bits (instantiation) | `Params(n, wc, k)` with `m = n · wc`; default `wc = 2384` (65,560 B); `LEAN_P3_MEETS` has `wc = 37` |
| 2 | Layer indices | `Fin 2`: 0 lower, 1 top | 1 and 2 | 0 and 1 |
| 3 | Key tag and salt | `KQ'` carries `Bits 256`; `tagOf` puts a 128-bit salt in bits 0..127, zeros above (segment 0 only). `p3ChainModel` has the instantiation's `tagR salt s = salt (24 B) ‖ u64(s)`, in the key domain `domP3` and in `P`'s tweak | `(salt, s)` as fields (§1); `tag_s = salt (24 B) ‖ u64(s)` (instantiation) | the instantiation's; for segment 0 with a salt under 2^128 it equals `tagOf` |
| 4 | Segments | `p3Scheme` is one segment of 220 labels; `n10prime` covers one segment | many segments per `W` | `N` segments, segment-major in `W` and `C`; `N = 1` is Lean's scheme |
| 5 | `H` | a monolithic random oracle `KQ' → Lab`; the pilot `p3SpongeModel` (`sponge-dense`) has the XOR sponge over `Π`; `p3ChainModel` has the overwrite chain over `Π₂` (`ovKey`), the code's default | a sponge over `Π` of `m + 512` bits (§2c), then an overwrite chain over `Π₂` of `2m + 512` bits (instantiation, 16:45Z) | pluggable; the default is the overwrite chain, and the broken XOR sponge and Miyaguchi–Preneel are kept for attack replay. The header encodes only `npar`, so `H` is injective on the scheme's queries (`parentsOf`), not on every partial assignment in `KQ'` |
| 6 | `P` | one untweaked ideal permutation `ω.2`; `p3ChainModel` has `P_T`, an independent uniform permutation for each `T = (tag_s, layer, v)` | tweakable `P_T`, `T = (tag_s, layer, v)` (instantiation) | `P_T`; the Lean cross-check uses a toy that ignores `T` |
| 7 | Decode rounds | `decodeW`: key and `P⁻¹` independent, so one round each pass with a monolithic `H` | "key, then `P⁻¹`" (§1) | key and `P⁻¹` side by side; an overwrite-chain key takes `npar + 1` rounds |
| 8 | Rounds per level | `n10prime` assumes `ColumnHard G ⌊(D + 1)/2⌋` (2 rounds per level); the concrete-`H` pins count `P`/`Π₂` rounds directly, `D = 10` on `B(220, 12)` and 12 on `B(220, 16)` | `⌊(D − 1)/13⌋ + 1` (§2c); 3 calls per level at worst, `⌊(D − 1)/3⌋ + 1` (instantiation §1) | rounds measured exactly per derivation; `CERTIFIED_ROUNDS = 10`, the pins' `D` (the instantiation's rule read `band_twoWay_220_12`'s 10 game levels as 28 rounds, which no `Meets` statement covers) |
| 9 | The column game's units | `colRound` charges one game round each for a reverse and for a promotion | — | an oracle adversary pays 1 round for a reverse (with a monolithic `H`) and 0 for a promotion. See below |
| 10 | Band graph at `k = 0` | `bandGraph n 0` keeps the spine parent `v − 1` | `par(v) = {v − k, …, v − 1}`, empty at `k = 0` | Lean's; the same for `k ≥ 1` |
| 11 | Verifier | the audit compares answers with the true blocks ("idealizes the per-block tags") | per-block tags; a SHA-256 Merkle root (instantiation §4) | the instantiation's tree (RFC 6962 shape) over SHA-512, with leaf digests |
| 12 | Audit parameters | `P3ConcreteMeets64` (concrete `H`): `D = 10`, `Q = 2^20`, `k = 106`, one segment on `B(220, 12)`; `P3ConcreteMeets14GB`, the same at 971 segments; `P3ConcreteMeets64D12`, `D = 12` on `B(220, 16)`. `p3_meets` (monolithic `H`): `D = 20`, `Q = 2^20`, `k = 119` (relies on the `pos = 0` collapse) or `185` (no-mask charge) | `k = 423`/`229` (§2a); `k = 132`, `Δ = 300` µs (§2c); `Δ = 0.5` ms plus RTT (`p3-gpu-measurements.md`); `k = 117` at `D = 10` (`scheme-choice-e2e.md`) | `k = 132`, `Δ = 0.5` ms, `rtt_allowance_ns = 0`, `Q = 2^14`, `CERTIFIED_ROUNDS = 10` |
| 13 | Key inputs | full parent labels (P3.0) | P3.1 digest keys, later dropped | P3.0 |
| 14 | Encoder and coins | `enc W ω salt`, with the salt uniform in `Coins` after `W` | the server encodes after committing to `W`; the verifier samples the salt and computes `vk` itself (§2c) | the prose's protocol (`setup`), which matches Lean's order when the commitment precedes the salt |
| 15 | Bytes | `Bits` via `codeEquiv` | little-endian integers (instantiation) | consistent: bit `i` of a string is bit `i mod 8` of byte `⌊i/8⌋` |
| 16 | `H`'s initial state | `p3ChainModel`'s `keyP3c` starts from `iv(domP3 tag L v) = 0^m ‖ tag_s ‖ u256(2v + L)`: the domain fills the 512-bit capacity part, no header, no call (the dense chain's `iv(dom)`, D1) | the 48-byte header `"pous-p3-v1-H" ‖ tag_s ‖ u8(layer) ‖ u16(v) ‖ u8(npar)` in the low bytes, zero-filled (instantiation §1) | the instantiation's (`primitives.OverwriteChain.initial`), so the concrete-`H` pins are not about the code's `H`, and `schemes.P3` has no certificate (`test_p3_has_no_certificate_because_its_chain_does_not_start_from_leans`). Closing it takes Lean's `h_0` in the code (a new byte format) or a Lean transfer to the header |

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
  Whether a gap remains under `⌊(D − 1)/3⌋ + 1` has not been checked. The concrete-`H` pins do not use that rule: they
  count oracle rounds of `p3ChainModel` directly.

**Item 5, the storage breaks of the earlier `H`.** The pilot proved that the XOR sponge's pre-squeeze states are not
full-price pebbles once `Π⁻¹` is queryable. With a public first state, one stored state regenerates two labels
(`dense_two_labels_from_one_state`). The model's B5 freshness step therefore fails for it. `tests/test_pous_two_for_one.py`
replays that attack, and the Miyaguchi–Preneel pre-image attack on band rows, on real labels, and shows both fail on the
overwrite chain. The overwrite chain's Lean statement is the two-permutation model of `Security/Definitions/Pous/Model/P3Chain.lean`,
with the pins `P3ConcreteMeets64`, `P3ConcreteMeets14GB` and `P3ConcreteMeets64D12`, proved with no hypothesis.

**What the Lean cross-check covers.** Lean evaluated `c1'`, `topLabel'`, `transpose`, `decodeLower`, `tagOf`, the
decoder, `bandGraph` and `colRound` for `n ≤ 3` (encode) and `n ≤ 6` (game), and the code matches on every value. The
`rfl` theorems in `tests/lean/PousRefCheck.lean` tie the generic definitions it evaluates to `P3Meets`' own at
`n = 220`. Items 3, 5 and 6 are outside `p3Scheme`, so the check plugs in a monolithic toy `H` and a tweak-ignoring
toy `P`. It does not evaluate `p3ChainModel` yet.
