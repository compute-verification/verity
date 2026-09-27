# verity/flock-circuit: the masked protocol (M1)

What `flock-circuit --zk` sends, where every mask comes from, and what the verifier checks. This is the layout the Lean
reference prover must reproduce byte for byte (soundness DESIGN.md §6: a deterministic function of statement, witness,
prover seed and verifier coins), and the statement its zero-knowledge theorem is about. Code: `src/zk_veil.rs` (masking,
replay, inner proof, simulator pieces) and `src/bin/flock-circuit.rs` (`prove_cpu_zk`, `CircuitVerifier::verify_zk`,
`zkstat`, `zkaudit`).

**Status.** CPU prototype, one template (RoPE d64), lane flock-zk. The algebraic phase (zerocheck and lincheck) and the
ring switch are masked. Ligerito is not (§8). So the proof class stays `NON_ZK_PROOF`, and the identity's
`hooks.not_yet_masked` lists what is missing. The target notion per repetition is VEIL's: special honest-verifier zero
knowledge, meaning one simulator for every fixed coin sequence (DESIGN.md §11). Malicious-verifier zero knowledge comes
from M2's coin commitment and single rewind (§9).

Conventions. The field is GF(2^128) (`F128`, characteristic 2, so `+` is `−`). An F128's bytes are `lo` then `hi`,
each little-endian. `v'` is a sent (masked) value, `v` the value upstream's prover computed, `h` a pad, and `v' = v + h`.
`m` is the statement's dense variable count, `k_log` its block size (RoPE: `m = 25..27`, `k_log = 22`).

## 1. Order of one repetition

Everything below is on the rep's live stream (`circuit/rep{r}`), in this order. Items marked M1 are new; the rest is
M0 and upstream Flock.

~~~text
bind_statement      label flock-r1cs-v0 · bytes statement digest · bytes witness cap
zerocheck           label flock-zerocheck-v0
  M1                label verity-flock-zk-pads-v0 · bytes pads root (64)
                    squeeze r_skip (6) · squeeze r_outer (m-13)
                    slice round1_ab' (64) · slice round1_c' (64) · squeeze z
                    per j < m-6: scalar G_j(1)' · scalar G_j(∞)' · squeeze ρ_j
                    scalar final_a' · scalar final_b'
lincheck            label flock-lincheck-v0 · squeeze α · squeeze β (const pin)
                    per j < k_log-6: scalar q_j(1)' · scalar q_j(∞)' · squeeze r_j
                    slice z_partial' (64) · squeeze z_skip'
opening             label flock-pcs-open-batch-v0            (masking ends here)
                    per claim: label flock-ring-switch-v0 · slice s_hat_v (128) · squeeze r'' (7)
                    ... Ligerito, unchanged (§8) ...
M1 inner proof      label verity-flock-zk-inner-v0 · squeeze ε · slice [ρ, σ] · squeeze γ (11)
                    scalar τ · squeeze β · slice Y (s+192) · squeeze queries (168)
proof bytes         bincode (Commitment, R1csProofLigerito with the masked zerocheck and lincheck, InnerProof)
~~~

`final_c'` is sent only in the proof: upstream never absorbs it. The claim order in the opening is ab, then c, then the
region claims (region-major, then the two link points).

## 2. The pads

Per rep, `s = 2·64 + 2(m−6) + 3 + 2(k_log−6) + 64 + 4` field elements (RoPE at `m = 25`: 269). `Layout` in
`zk_veil.rs` is normative.

| indices | pad of | count (RoPE m=25) |
|---|---|---|
| `[0, 64)` | `round1_ab` | 64 |
| `[64, 128)` | `round1_c` | 64 |
| `128 + 2j + t` | multilinear round `j`, `t = 0: G(1)`, `t = 1: G(∞)` | 38 |
| `fa, fb, fc = 128 + 2(m−6) + {0,1,2}` | `final_a`, `final_b`, `final_c` | 3 |
| `fc + 1 + 2j + t` | lincheck round `j`, `t = 0: q(1)`, `t = 1: q(∞)` | 32 |
| `zp + i`, `zp = fc + 1 + 2(k_log−6)` | `z_partial[i]` | 64 |
| `hab = zp + 64` | `h_ab = h_fa · h_fb` | 1 |
| `tx, ty, tw = hab + {1,2,3}` | the sacrificed triple `x, y, w = x · y` | 3 |

The first `hab` pads mask sent values; the last four are committed auxiliaries. The absorption order is
`Layout::absorbed()`: the table's rows in order, minus `fc`.

**Randomness.** Everything comes from `ProverRng`, which is ChaCha20 keyed by the session's `ProverSeed`: a fresh
`getrandom` draw, and the same seed that fills the mask slot. `f128s(stream, index, n)` reads `16n` bytes from byte
offset `32·index` of the stream. With `base = rep << 32`:

~~~text
h      = f128s(4, base, s)            then h[hab] := h[fa]·h[fb],  h[tw] := h[tx]·h[ty]
mu     = f128s(5, base, s)            the dot-product mask
pad_h  = f128s(4, base + 2^24, 192)   the pads codeword's padding
pad_mu = f128s(5, base + 2^24, 192)
mask slot, block b: f128s(1, b · 2^(mask_log−7), 2^(mask_log−7))      (M0's hook, uniform in --zk)
~~~

Streams 1–3 belong to M0: mask, padding, and the injected-seed salt tape. The pads are never derived from the transcript.

## 3. The pads commitment

The code is Reed–Solomon over GF(2^128). The message is `(h ‖ pad_h)` of length `L = s + 192`, read as coefficients
of `Σ m_i X^i`. The code is evaluated at the `N = 8 · next_pow2(L)` points `x_j = F128(lo = j + 1, hi = 0)`, which
gives RoPE `N = 4096` at rate `L/N ≤ 1/8`. There are two rows, `C(h ‖ pad_h)` and `C(mu ‖ pad_mu)`. Leaf `j` is the
`hm96-sha512/v1` leaf of `x = SHA-512(bytes(C_h[j]) ‖ bytes(C_mu[j]))`. Its salt is leaf `j` of a fresh tree id from
the proof's `LeafSalts`, that is, ChaCha20 blocks `3j..3j+2` under the proof's salt key with nonce `id`. The id is drawn
with `next_id()` before the rep's first message, so it is never 0, never a Ligerito tree's, and never reused. Nodes are
`SHA-512(left ‖ right)`. The 64-byte root is the item after `verity-flock-zk-pads-v0`.

## 4. The constraints

The verifier re-runs upstream's zerocheck and lincheck verifiers on the masked proof (`zk_veil::replay`). It draws the
same coins and absorbs the same items, and every message-dependent quantity is an affine form `c + Σ a_i h_i` (`Lin`).
Each check `lhs = rhs` becomes the constraint `lhs + rhs = 0`. `Lagrange` weights come from upstream's own
`interpolate_at_z_on_lambda` and `interpolate_at_z_combined`, evaluated on unit vectors.

| name | constraint (all `= 0`) |
|---|---|
| zerocheck c_eval | `Σ_i W_c(z)_i (round1_c'_i + h_{r1c(i)}) + final_c' + h_fc` |
| zerocheck final a·b | `G_final + final_a'·final_b' + final_a'·h_fb + final_b'·h_fa + h_ab` (`G_final` the running claim, an affine form) |
| lincheck final | `running + Σ_i comb_i (z_partial'_i + h_{zp(i)})` (`comb` = `fold_alpha_batched` folded at the rounds' coins, public) |
| ring switch ab | `claim_check(w_ab, s_hat_v_ab) + Σ_i λ_i (z_partial'_i + h_{zp(i)})` (`λ` = φ8 weights at `z_skip'`) |
| ring switch c | `claim_check(w_c, s_hat_v_c) + final_c' + h_fc` |
| triple ρ | `ε h_fa + h_tx + ρ` |
| triple σ | `h_fb + h_ty + σ` |
| triple sacrifice | `ε h_ab + h_tw + σ h_tx + ρ h_ty + ρσ` |

The two ring-switch rows replace upstream's claim check on the ab and c claims. The verifier calls upstream's opening
with `claims[0..2] = claim_check(...)`, so upstream's own check passes, and every other part of the opening runs
unchanged. The triple rows make `h_ab = h_fa h_fb` binding: if `h_ab = h_fa h_fb + δ1` and `h_tw = h_tx h_ty + δ2`,
the sacrifice row evaluates to `ε δ1 + δ2`, so it holds for at most one `ε`.

## 5. The inner proof

It proves `⟨c, h⟩ = t` for the batched constraint, where `(c, t)` comes from `batch(cons, γ)` over the 11 rows:

1. `ε` ← V. P sends `ρ = ε h_fa + h_tx` and `σ = h_fb + h_ty`.
2. `γ` (11) ← V. Both sides form `(c, t)`. P sends `τ = ⟨c, mu⟩`.
3. `β` ← V. P refuses `β = 0`. P sends `Y = (h + β mu ‖ pad_h + β pad_mu)`.
4. 168 query words ← V, reduced to `j = lo mod N`. P opens `(C_h[j], C_mu[j])` with salts and paths.
5. V checks `⟨c, Y[..s]⟩ = t + β τ`, `C(Y)_j = C_h[j] + β C_mu[j]` at every query, and each path to the pads root.

**Soundness terms, for review.** Batching costs `1/|F|`, the sacrifice `1/|F|`, and `β` `1/|F|`. RS proximity is at
rate `L/N ≈ 0.11` with 168 queries: `(1 − δ)^168` for the unique-decoding `δ ≈ 0.44` is about `2^-140`, plus BCIKS20's
`N/|F|` for the two-row combination. The pads bind before the first exposed value, because the root is absorbed before
`r_skip`. The base protocol's soundness then carries over as in VEIL Lemma 4.3: `v' − h` is a prover strategy for the
unmasked protocol.

## 6. The mask slot hides the masked claims' `s_hat_v`

`s_hat_v^p[v] = Σ_x eq(p_w, x) z[128x + v]` over word indices `x`, where `p_w = x_full[1..]` of the claim point, taken
LSB first. The audit checks this indexing by recomputing every region claim's `s_hat_v` from public data. The mask
slot's words (`2^(mask_log−7)` per block, all blocks, uniform bits `u`) add `Σ_{x ∈ mask} eq(p_w, x) u_{v,x}`. For the
claim points `p_1..p_k` seen so far (ab and c of each rep, up to 4), that sum is uniform on `F^k`, independently for
every `v`, exactly when the vectors `(eq(p_1, x), .., eq(p_k, x))` over the mask words span `GF(2)^{128k}`. The prover
checks this rank at the opening's label, before any `s_hat_v` leaves (`mask_rank_ok`, 2–4 ms), and stops otherwise.
RoPE has 1024 mask words, so the rank needed is 512 of 1024.

Region claims have Boolean coordinates on their fixed bits, so their `s_hat_v` is a function of public region data
(`b ‖ c` and outputs). `Stmt::region_s_hat_v` recomputes it, and the audit checks equality. The unmasked claim values
`claim_check(s_hat_v)` of ab and c are functions of the uniform `s_hat_v`.

## 7. The simulator (per rep, fixed coins)

The input is the public statement and the verifier's coins: the algebraic phase's (the `Tape` of §1's squeezes up to the
opening), and the inner proof's. The simulator (`simulate_rep`, `zk_veil::simulate_inner`):

1. Draw the exposed values uniformly (`hab` elements), and draw `s_hat_v_ab` and `s_hat_v_c` uniformly (128 each).
2. Compute the region `s_hat_v` from public data.
3. Build the constraints by `replay` on the tape, plus the two ring-switch rows.
4. Draw `ρ, σ, Y` uniformly and solve `τ = (⟨c, Y⟩ + t) / β`.
5. Draw each opened `C_h[j]` uniformly and set `C_mu[j] = (C(Y)_j + C_h[j]) / β`. Any `Q ≤ 192` columns of one padded
   row are uniform.
6. Commit a tree holding those columns and zero columns elsewhere; hm96 hides every leaf.

The simulator refuses the coins the prover refuses. Evidence at N = 64 sessions (128 transcripts each):
`tests/zk_stats.py` finds real and simulated indistinguishable on every class, with all p above 0.02 on byte histograms,
bits and cross-run XOR. The unmasked control (zero pads, zero mask slot) is distinguished with p ≈ 0 and every position
constant. See the lane's evidence `lanes/flock-zk/evidence/20260927T0830Z-*`.

## 8. Not yet masked: Ligerito (what M1 still needs)

The audit (`flock-circuit zkaudit --zk`) attributes every byte the verifier keeps. For RoPE at `m = 25`, per session,
three Ligerito classes are in the clear:

- the level-0 opened rows, 446 KB (218 queries of 64 lanes);
- the recursive levels' rows, 118 KB (106, 71 and 53 queries);
- the sumcheck messages, the final vector and the out-of-domain values, 4.3 KB. These include one level-0 OOD value
  `P̂(z)`, which is a bare functional of the witness.

The Ligerito parameters are `initial_k = 6`, `2^12` rows, rates `1/2, 1/4, 1/8, 1/16`, and OOD samples `1, 2, 2, 2`.

The fix is VEIL Figs. 8–9 applied to level 0 of the F256 split ladder (`pcs/ligerito/extension.rs`,
`recursive_prover_with_basis_impl`, and `recursive_verifier_with_basis_succinct`):

1. **Per-lane padding.** Each lane's message becomes `(lane ‖ mu_lane)`, with `t_pad ≥ 2 · 218` uniform
   coefficients (both reps open level 0), in the same `2^13`-symbol codeword. Relative distance drops to
   `1/2 − t_pad/2^13 ≈ 0.447` at `m = 25`, and the level-0 query count must be recomputed; at production sizes
   (`2^18`-row lanes) the drop is below `2^-9`. This happens in `pcs/commit.rs::commit_into`, which must stop
   replicating `z` into the upper half and write the padding there before the NTT.
2. **One extra uniform lane `R`.** The leaf gets 65 entries (+1/64 encoding and hashing). After the six lane rounds
   (challenges `r̄`), V draws `ρ_R`, and the fold becomes `y_1' = y_1 + ρ_R R`. P sends `e = ⟨b(r̄, ·), R⟩` and
   `ybar = Σ eq(r̄, ℓ) mu_ℓ + ρ_R mu_R`. Level 1 commits the uniform `y_1'`. Every later message (levels 1–3, the final
   vector, the OOD values at levels 1–3) is then a function of a uniform vector and simulates by running the honest
   prover on a uniform `y_1'`.
3. **Exposed.** Pad the level-0 OOD value, the six lane-round messages (F256: two pads per coordinate pair) and `e`.
   They join `Layout` and the masking phase, which then ends at the level-1 commitment instead of the opening's label.
   Their checks become rows: the lane sumcheck chain (affine over F128 coordinates), the OOD glue, and the level-1 start
   `T + ρ_R e = T'`, with `T'` sent in the clear because it is uniform.
4. **Level-0 row check.** `⟨row_q[..64], eq(r̄)⟩ + ρ_R row_q[64] + Enc(0 ‖ ybar)(q) = Enc(y_1')(q)`, folded into
   the induced basis's enforced sum as upstream already does.

The expected cost is about +2% on Ligerito's level 0 plus the extra queries at small `m`. At the C1 check, the padding
and `R` are drawn once per session, so both reps still bind one level-0 root.

## 9. What M2 still needs

- **The coin commitment** (DESIGN.md §11.1). One up-front commitment per session: a SHA-512 Merkle root over
  Halevi–Micali commitments to every coin, one per (table, repetition, round), with each stream a subtree. Every prover
  machine checks each coin's opening before using it. This replaces M0's `coin_seed` step 0 (a seed commitment with
  `derive()` streams, which needs a PRF assumption), and the prover's refusals (`β = 0`, mask rank, and §8's `ρ_R = 0`)
  stay functions of committed coins.
- **The single-rewind simulator** (Goldreich–Kahan). The simulator runs the honest prover on a dummy witness until the
  final openings reveal the committed coins. It then rewinds once to just after the coin commitment and runs §7's
  per-table simulator on those coins, for every table, with independent masks. This needs §8 first: until then,
  Ligerito's messages leave before the last coin and depend on the witness.
- **C1's level-0 root check**, kept under M1. M0's commit hook compares rep 1's level-0 root with rep 0's and stops
  before rep 1 opens anything, and the device refuses salt stream 0 for any tree whose leaf count is not level 0's
  (commit `e2190ca3`). M1 keeps it: both reps prove one witness, uniform mask slot included. The pads trees take
  `next_id() ≥ 1`. §8's padding and `R` must be drawn once per session and belong to level 0, so the check covers them.
- **GPU.** `--zk` is CPU-only. The masking challenger and the inner proof are host-side and small. The mask slot fill and
  §8's padding and `R` belong on the device, expanded from ChaCha20 like the hm96 salts.
- **Cost.** The prover re-runs the lincheck's matrix fold (0.13 s per rep at `k_log = 22`, constant in `m`) to rebuild
  its constraints. Upstream's lincheck prover already holds that fold, and returning it removes the cost. The inner
  proof's 168 paths (130 KB per rep) could share a cap.
- **Review.** A red team on this document and on §8's implementation, then the Lean reference prover and its
  differential test, before any `COMPLETE_*` class.
