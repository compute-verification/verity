# verity/flock-circuit: the masked protocol (M1)

What `flock-circuit --zk` sends, where every mask comes from, and what the verifier checks. This is the layout the Lean
reference prover must reproduce byte for byte (soundness DESIGN.md §6: a deterministic function of statement, witness,
prover seed and verifier coins), and the statement its zero-knowledge theorem is about.

Code:
- `src/zk_veil.rs`: masking, replay, constraints, the inner proof, the simulator's pieces.
- `src/bin/flock-circuit.rs`: `prove_cpu_zk`, `zk_constraints`, `verify_zk_with`, `simulate_full`, `zkstat`, `zkaudit`.
- `../flock-zk-b684b12.patch`: Ligerito level 0's hiding in upstream Flock (`pcs::ligerito::zk`, `commit::commit_zk`, the F256
  split prover and verifier).

**Status.** CPU prototype, one template (RoPE d64), lane flock-zk. Every message a repetition sends is masked, committed
or public, and the audit (`zkaudit`) attributes every byte. The target notion per repetition is VEIL's: special
honest-verifier zero knowledge, meaning one simulator for every fixed coin sequence (DESIGN.md §11). Malicious-verifier
zero knowledge comes from M2's coin commitment and single rewind (§9). The proof class stays `NON_ZK_PROOF` until three
things are done: the red team reviews this document, the Lean proof covers it, and level 0's query count is recomputed
for its padded code (§8).

Conventions. The field is GF(2^128) (`F128`, characteristic 2, so `+` is `−`). F256 is `F128 ⊕ F128·U`, written by
coordinates `(c0, c1)`. An F128's bytes are `lo` then `hi`, each little-endian. `v'` is a sent (masked) value, `v` the
value upstream's prover computed, `h` a pad, and `v' = v + h`. `m` is the dense variable count and `k_log` the block size
(RoPE: `m = 25..27`, `k_log = 22`). Ligerito at `m = 25`: `initial_k = 6` (64 lanes), `L = 2^12` rows per lane, rates
`1/2, 1/4, 1/8, 1/16`, queries `218, 106, 71, 53`, OOD samples `1, 2, 2, 2`.

## 1. Order of one repetition

Everything below is on the rep's live stream (`circuit/rep{r}`), in this order. Items marked M1 are new; the rest is
M0 and upstream Flock.

~~~text
bind_statement      label flock-r1cs-v0 · bytes statement digest · bytes witness cap (66-lane level-0 tree)
zerocheck           label flock-zerocheck-v0
  M1                label verity-flock-zk-pads-v0 · bytes pads root (64)
                    squeeze r_skip (6) · squeeze r_outer (m-13)
                    slice round1_ab' (64) · slice round1_c' (64) · squeeze z
                    per j < m-6: scalar G_j(1)' · scalar G_j(∞)' · squeeze ρ_j
                    scalar final_a' · scalar final_b'
lincheck            label flock-lincheck-v0 · squeeze α · squeeze β (const pin)
                    per j < k_log-6: scalar q_j(1)' · scalar q_j(∞)' · squeeze r_j
                    slice z_partial' (64) · squeeze z_skip'
opening             label flock-pcs-open-batch-v0                         (the algebraic phase's masking ends)
                    per claim: label flock-ring-switch-v0 · slice s_hat_v (128) · squeeze r'' (7)
                    squeeze γ (claims)
Ligerito level 0    label flock-ligerito-basis-f256-split-v0 · scalar target · bytes cap
  M1                label verity-flock-zk-l0-open-v0                      (level 0's masking begins)
                    per OOD sample: squeeze z (log_n) · scalar y' · bytes nonce · squeeze β
                    msg_0' (u_0 slice 2, u_2 slice 2)
                    per j < initial_k: squeeze r_j (F256) · msg_{j+1}' for j+1 < initial_k
  M1                slice e_0' (2) · slice e_1' (2) · squeeze ρ_0, ρ_1 (F256)
  M1                label verity-flock-zk-l0-close-v0                     (level 0's masking ends)
  M1                slice T' (2) · slice ybar (2 t_pad)
                    code-switch message · cap_1 · ... levels 1+, unchanged ...
M1 inner proof      label verity-flock-zk-inner-v0 · squeeze ε · slice [ρ, σ] · squeeze γ (10)
                    scalar τ · squeeze β · slice Y (s+192) · squeeze queries (168)
proof bytes         bincode (Commitment, R1csProofLigerito with the masked fields, InnerProof)
~~~

`final_c'` is sent only in the proof: upstream never absorbs it. The claim order in the opening is ab, then c, then the
region claims (region-major, then the two link points).

## 2. The pads

Per rep, the pad count is

~~~text
s = 2·64 + 2(m−6) + 3 + 2(k_log−6) + 64 + (n_ood0 + 4·initial_k + 4) + 4
~~~

RoPE at `m = 25`: 298. `Layout` in `zk_veil.rs` is normative.

| indices | pad of | count (RoPE m=25) |
|---|---|---|
| `[0, 64)` | `round1_ab` | 64 |
| `[64, 128)` | `round1_c` | 64 |
| `128 + 2j + t` | multilinear round `j`, `t = 0: G(1)`, `t = 1: G(∞)` | 38 |
| `fa, fb, fc = 128 + 2(m−6) + {0,1,2}` | `final_a`, `final_b`, `final_c` | 3 |
| `fc + 1 + 2j + t` | lincheck round `j`, `t = 0: q(1)`, `t = 1: q(∞)` | 32 |
| `zp + i`, `zp = fc + 1 + 2(k_log−6)` | `z_partial[i]` | 64 |
| `lg_ood(i) = zp + 64 + i` | Ligerito level 0's OOD value `i` | 1 |
| `lg_msg(j, t) = zp + 64 + n_ood0 + 4j + t` | lane round `j`'s message, `t`: `u_0.c0, u_0.c1, u_2.c0, u_2.c1` | 24 |
| `lg_e(i, c)` (next four) | extra lane `i`'s sum, coordinate `c` | 4 |
| `hab` (next) | `h_ab = h_fa · h_fb` | 1 |
| `tx, ty, tw = hab + {1,2,3}` | the sacrificed triple `x, y, w = x · y` | 3 |

The first `hab` pads mask sent values, and the last four are committed auxiliaries. The absorption order is
`Layout::absorbed()`: the rows above in order, minus `fc`. The algebraic phase is masked between the zerocheck label and
the opening's label. Level 0 is masked between `verity-flock-zk-l0-open-v0` and `-close-v0`.

**Randomness.** Everything comes from `ProverRng`, which is ChaCha20 keyed by the session's `ProverSeed` (a fresh
`getrandom` draw). `f128s(stream, index, n)` reads `16n` bytes from byte offset `32·index` of the stream. With
`base = rep << 32`:

~~~text
h      = f128s(4, base, s)            then h[hab] := h[fa]·h[fb],  h[tw] := h[tx]·h[ty]
mu     = f128s(5, base, s)            the dot-product mask
pad_h  = f128s(4, base + 2^24, 192)   the pads codeword's padding
pad_mu = f128s(5, base + 2^24, 192)
mask slot, block b: f128s(1, b · 2^(mask_log−7), 2^(mask_log−7))      (M0's hook, uniform in --zk)
Ligerito level 0 (once per session, both reps):
  extra lane i < 2:       f128s(6, i << 32, L)
  lane l's padding:       f128s(7, l << 32, t_pad)       l < 64 + 2,  t_pad = 2·queries_0 + 64 (500 at m = 25)
~~~

Streams 1–3 belong to M0: mask, padding, and the injected-seed salt tape. None of this is derived from the transcript.

## 3. The pads commitment

The code is Reed–Solomon over GF(2^128). The message is `(h ‖ pad_h)` of length `L_h = s + 192`, read as coefficients
of `Σ m_i X^i`. The code is evaluated at the `N = 8 · next_pow2(L_h)` points `x_j = F128(lo = j + 1, hi = 0)`, which
gives RoPE `N = 4096`. There are two rows, `C(h ‖ pad_h)` and `C(mu ‖ pad_mu)`. Leaf `j` is the `hm96-sha512/v1` leaf of
`x = SHA-512(bytes(C_h[j]) ‖ bytes(C_mu[j]))`. Its salt is leaf `j` of a fresh tree id from the proof's `LeafSalts`
(ChaCha20 blocks `3j..3j+2` under the salt key, nonce `id`). The id comes from `next_id()`, so it is never 0 and never
reused. Nodes are `SHA-512(left ‖ right)`. The 64-byte root is the item after `verity-flock-zk-pads-v0`.

## 4. The constraints

The verifier re-runs upstream's zerocheck and lincheck verifiers on the masked proof (`zk_veil::replay`), drawing the
same coins and absorbing the same items. Every message-dependent quantity is an affine form `c + Σ a_i h_i` (`Lin`; over
F256, two of them, `Lin2`). Each check `lhs = rhs` becomes `lhs + rhs = 0`. The Lagrange weights come from upstream's
`interpolate_at_z_on_lambda` and `interpolate_at_z_combined`, evaluated on unit vectors.

| name | constraint (all `= 0`) |
|---|---|
| zerocheck c_eval | `Σ_i W_c(z)_i (round1_c'_i + h_{r1c(i)}) + final_c' + h_fc` |
| zerocheck final a·b | `G_final + final_a'·final_b' + final_a'·h_fb + final_b'·h_fa + h_ab` (`G_final` the running claim) |
| lincheck final | `running + Σ_i comb_i (z_partial'_i + h_{zp(i)})` (`comb` = `fold_alpha_batched` folded at the rounds' coins) |
| ring switch ab | `claim_check(w_ab, s_hat_v_ab) + Σ_i λ_i (z_partial'_i + h_{zp(i)})` (`λ` = φ8 weights at `z_skip'`) |
| ring switch c | `claim_check(w_c, s_hat_v_c) + final_c' + h_fc` |
| ligerito lane phase (c0, c1) | `claim_k + ρ_0 e_0 + ρ_1 e_1 + T'`, where `claim_0 = target + Σ_i β_i (y'_i + h)` and `claim_{j+1} = u_0 + r_j (claim_j + u_2) + r_j² u_2` over the masked lane messages (upstream's `RoundQuad256`) |
| triple ρ | `ε h_fa + h_tx + ρ` |
| triple σ | `h_fb + h_ty + σ` |
| triple sacrifice | `ε h_ab + h_tw + σ h_tx + ρ h_ty + ρσ` |

The ring-switch rows replace upstream's claim check on the ab and c claims: the verifier calls upstream's opening with
`claims[0..2] = claim_check(...)`, and every other part of the opening runs unchanged. The lane-phase rows replace the
Ligerito verifier's lane-round arithmetic, which starts again from `T'`. The triple rows bind `h_ab = h_fa h_fb`: with
`h_ab = h_fa h_fb + δ1` and `h_tw = h_tx h_ty + δ2`, the sacrifice row evaluates to `ε δ1 + δ2`.

## 5. The inner proof

It proves `⟨c, h⟩ = t` for the batched constraint, where `(c, t)` comes from `batch(cons, γ)` over the 10 rows:

1. `ε` ← V. P sends `ρ = ε h_fa + h_tx` and `σ = h_fb + h_ty`.
2. `γ` (10) ← V. Both sides form `(c, t)`. P sends `τ = ⟨c, mu⟩`.
3. `β` ← V. P refuses `β = 0`. P sends `Y = (h + β mu ‖ pad_h + β pad_mu)`.
4. 168 query words ← V, reduced to `j = lo mod N`. P opens `(C_h[j], C_mu[j])` with salts and paths.
5. V checks `⟨c, Y[..s]⟩ = t + β τ`, `C(Y)_j = C_h[j] + β C_mu[j]` at every query, and each path to the pads root.

**Soundness terms, for review.** Batching costs `1/|F|`, the sacrifice `1/|F|`, and `β` `1/|F|`. RS proximity is at
rate `L_h/N ≈ 0.12` with 168 queries: `(1 − δ)^168` at the unique-decoding `δ ≈ 0.44` is about `2^-140`, plus BCIKS20's
`N/|F|`. The pads bind before the first exposed value, because the root is absorbed before `r_skip`. The base protocol's
soundness then carries over as in VEIL Lemma 4.3: `v' − h` is a prover strategy for the unmasked protocol.

## 6. The mask slot hides the masked claims' `s_hat_v`

`s_hat_v^p[v] = Σ_x eq(p_w, x) z[128x + v]` over word indices `x`, where `p_w = x_full[1..]` of the claim point, taken
LSB first. The audit checks this indexing by recomputing every region claim's `s_hat_v` from public data. The mask slot's
words (`2^(mask_log−7)` per block, all blocks, uniform bits `u`) add `Σ_{x ∈ mask} eq(p_w, x) u_{v,x}`. For the claim
points `p_1..p_k` seen so far (ab and c of each rep, up to 4), that sum is uniform on `F^k`, independently for every `v`,
exactly when `(eq(p_1, x), .., eq(p_k, x))` over the mask words span `GF(2)^{128k}`. The prover checks this rank at the
opening's label, before any `s_hat_v` leaves (`mask_rank_ok`, 2–4 ms), and stops otherwise. RoPE: rank 512 needed, from
1024 words. Region claims' `s_hat_v` are functions of public region data (`Stmt::region_s_hat_v`).

## 7. Ligerito level 0 (VEIL Figs. 8–9)

Without this, Ligerito sends witness functionals in the clear. At `m = 25` the audit finds per session:

- the level-0 rows, 446 KB;
- the recursive rows, 118 KB;
- the sumcheck messages, the OOD values and the final vector, 4.3 KB. One level-0 OOD value is `P̂(z)`.

What is changed:

1. **Per-lane padding** (`zk::lane_codeword`). Lane `l`'s codeword encodes `P_l + (c·pad_l ‖ 0) ‖ pad_l ‖ 0`: `t_pad`
   uniform coefficients at `[L, L + t_pad)` and `c` times them at `[0, t_pad)`, with `c = x` (`PAD_C`). That is, the
   padding polynomial is `(X_L + c)·μ` with `μ = Σ pad_i X_i` of degree `< t_pad`.
   - *The binary-field point:* in the additive NTT's novel basis `X_L = Ŵ_n`, which is 0 on the first half of the
     codeword's positions and 1 on the other half, which I checked on upstream's NTT. So VEIL's high-coefficient
     padding alone would leave half the opened positions unmasked.
   - `X_L + c` vanishes nowhere, so any `t_pad` distinct positions of a lane are uniform.
   - The committed word still determines `P_l` (its first `L` coefficients minus `c` times the next `t_pad`).
2. **Two uniform lanes** (`zk::EXTRA_LANES`), padded the same way. The leaf holds 66 F128 values (`commit::commit_zk`:
   one tree over the same positions, so the same hm96 level-0 salts and C1's shared root).
3. **After the lane rounds** (challenges `r̄`), in this order:
   - P sends `e_i = ⟨b(r̄, ·), R_i⟩` (exposed, padded). They precede the coins that combine them, as in VEIL step 1;
     otherwise a prover could choose `e` to fit any claim.
   - V draws `ρ_0, ρ_1 ∈ F256`. P refuses them unless they are independent over F128.
   - The fold becomes `y_1' = y_1 + ρ_0 R_0 + ρ_1 R_1`, which is uniform on F256^L.
   - P sends `T' = ⟨b(r̄, ·), y_1'⟩` and `ybar = Σ_l eq(r̄, l) pad_l + ρ_0 pad_{R0} + ρ_1 pad_{R1}` (`t_pad` F256 values).
   - The code switch and level 1 commit `y_1'`. The verifier's running claim restarts at `T'`.
4. **The level-0 row check**:

   ~~~text
   ⟨row_q[..64], eq(r̄)⟩ + ρ_0 row_q[64] + ρ_1 row_q[65] + Enc(c·ybar ‖ ybar)(q) = Enc(y_1')(q)
   ~~~

   It is folded into the induced basis's enforced sum. The induced basis itself is unchanged.
5. **Exposed** (padded, §2): the level-0 OOD value, the six lane-round messages and `e`. The prover's lane chain becomes
   the two lane-phase rows (§4).

Why this is zero knowledge, as in VEIL Lemma 4.7:
- `y_1'` and `ybar` are uniform, through `R`, `pad_R` and the F128-independence of `ρ`.
- The P-lane entries at the opened positions are uniform, through per-lane padding with `t_pad ≥` all level-0 positions
  of both reps. The R-lane entries are then determined by the row check.
- Every later message (the code switch, levels 1–3, their OOD values, the final vector) is a function of `y_1'` and public
  data.

Negatives, each rejected on both reps (RoPE CPU selftest):
- another extra lane folded than the committed one;
- `T'` forged;
- `ybar` forged;
- `e` forged with an honest `T'`. Ligerito accepts this one, and only the inner proof's lane-phase rows reject it.

## 8. The simulator, evidence, and what M1 still needs

**The simulator** (`simulate_full`, per rep, fixed coins). Its input is the public statement, the verifier's coins for the
rep's stream (every squeeze, in order) and the two link points. It builds:

1. uniform exposed values and uniform masked-claim `s_hat_v`;
2. the region claims' `s_hat_v` from public data;
3. the opening as the honest prover makes it on a dummy (all-zero) witness with fresh masks, against the same coins, with
   level 0's exposed values drawn uniformly. This is Goldreich–Kahan's dummy-witness run, restricted to the part whose
   distribution the masks make witness-independent;
4. the inner proof simulated for the constraints the verifier derives from all of the above (`zk_veil::simulate_inner`):
   `ρ, σ, Y` uniform, `τ = (⟨c, Y⟩ + t)/β`, and the opened columns uniform subject to `C(Y)_j = C_h[j] + β C_mu[j]`.

The verifier (`verify_zk_with` on a coin tape) accepts every simulated proof. The simulator refuses the coins the prover
refuses.

**Statistics** (`zkstat --zk --n 64`, then `tests/zk_stats.py`). Real and simulated transcripts are compared per class:
exposed values, masked `s_hat_v`, the inner proof, level 0's exposed values, `T'` and `ybar`, level-0 rows, and levels 1+.
Each class gets byte-histogram, bit-balance and cross-run-XOR tests. Positions constant in every real and simulated
transcript are public structure: F256 messages after the code switch are base-valued (5 such positions in the
levels-1+ sample). The control (all prover randomness zero) must be distinguished.

Results for RoPE at `m = 25`, 128 transcripts per kind:
- real and simulated agree on all 11 classes; the smallest of the 55 p-values is 0.013;
- the control is distinguished on all 7 masked classes (p ≈ 0, every position constant);
- all 128 simulated proofs are accepted.

Evidence: `lanes/flock-zk/evidence/`.

**The audit** (`zkaudit --zk`). Every byte the verifier keeps, and the proofs' opened data, is attributed:
- masked values differ from the control's;
- region `s_hat_v` equal their public recomputation;
- every Ligerito value differs from the control's;
- nothing is unattributed or in the clear (`complete: true`).

**What M1 still needs:**
- **Level 0's query count.** Level 0's code now has message length `L + t_pad` in `2L` positions, so its relative
  distance drops from `1/2` to `1/2 − t_pad/2L` (0.439 at `m = 25`). Its 218 queries are Fast100's for rate 1/2 and must
  be recomputed (the soundness lane). At production sizes (`2^18`-row lanes) the drop is below `2^-9`.
- **A red-team review** of §4–§7.
- **Cost.** Measured on a 4-core VM, per rep, from loopback sessions (median of three timed runs):

  | | M0 prove | M1 prove | upstream under M1 | proof per rep |
  |---|---|---|---|---|
  | RoPE `m = 25` | 0.25–0.29 s | 0.31–0.32 s (+12–24%) | 0.25–0.27 s | 560 → 759 KB |
  | RoPE `m = 27` | 0.38 s | 0.46 s (+20%) | 0.39–0.40 s | 618 → 817 KB |

  - The prover's own replay takes its lincheck's folded `comb` (`LincheckProof::comb_partial`, never serialized). The
    remaining fixed part is 0.026 s of replay and 0.02 s for the pads commitment.
  - Upstream's share (+7% at `m = 27`) is mostly level 0's padding codewords, one full NTT per lane (66 × 2L). One
    interleaved NTT of the padded input from layer 0 would cost about `66/64` of upstream's encoding.
  - The proof grows by the inner proof, mostly its 168 paths (130 KB per rep), which could share a cap, plus `ybar`
    (16 KB) and the two extra lanes' entries.

## 9. What M2 still needs

- **The coin commitment** (DESIGN.md §11.1). One up-front commitment per session: a SHA-512 Merkle root over
  Halevi–Micali commitments to every coin, one per (table, repetition, round), each stream a subtree. Every prover
  machine checks each coin's opening before using it. This replaces M0's `coin_seed` step 0 (a seed commitment with
  `derive()` streams, which needs a PRF assumption). The prover's refusals (`β = 0`, mask rank, `ρ_0, ρ_1` dependent)
  stay functions of committed coins.
- **The single-rewind simulator** (Goldreich–Kahan). The simulator runs the honest prover on a dummy witness until the
  final openings reveal the committed coins. It then rewinds once to just after the coin commitment and runs §8's
  per-table simulator on those coins, for every table, with independent masks. Every message before the last coin is
  masked, as §7 now makes Ligerito's too. So the dummy run is indistinguishable from a real one until the openings, as
  §11.1 requires.
- **C1's level-0 root check**, kept under M1. M0's commit hook compares rep 1's level-0 root with rep 0's and stops
  before rep 1 opens anything, and the device refuses salt stream 0 for any tree whose leaf count is not level 0's
  (commit `e2190ca3`). M1 keeps it:
  - both reps prove one witness, uniform mask slot included;
  - §7's padding and extra lanes are drawn once per session and belong to level 0's one tree;
  - the pads trees take `next_id() ≥ 1`.
- **GPU.** `--zk` is CPU-only. The masking challenger and the inner proof are host-side and small. The mask slot fill,
  §7's padding and extra lanes, and `commit_zk` belong on the device, expanded from ChaCha20 like the hm96 salts. The
  device prover also needs §7's fold and messages.
- **Review.** A red team on this document, then the Lean reference prover and its differential test, before any
  `COMPLETE_*` class.
