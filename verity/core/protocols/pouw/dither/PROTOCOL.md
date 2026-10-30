# Dither v3.2

Dither credits the int8 matmuls of a served model. Each call is computed on dithered and noised operands, its running
sums are committed while it runs, and live coins pick which of them the developer proves. An accepted monitoring period
is credited at the public best price of its work, and the work the developer could have done instead is bounded by
ε_skip·W_reg. This is v3.2: v3.1 (CMP-108's D1, [#4746](https://github.com/compute-verification/verity/pull/4746)) with
the fixes RT-POOL routed to it (note:compute-research/20261026T1752Z-draft-accel-rt-pool, on CMP-151, CMP-152 and
CMP-154), written in the shared notation (CMP-158).

Its credit at pub(μ°) against W_reg is superseded (Daniel, 26 Oct, 6:40 PM PDT, tentative): credit is a sampling block's
size in AND gates, a statement about values, and 𝓜's resources appear only inside the exclusion argument
(`GLOSSARY.md`, "Size, credit and exclusion"). CMP-158 restates this document and D6 in it.

The protocol is experimental. Its guarantee is stated in Lean on open PRs: `Pouw.Guarantees.Dither.WorkSound` and its
card instances `Pouw.Guarantees.DitherInstance.WorkSoundAt` on
[#4665](https://github.com/compute-verification/verity/pull/4665), over the ACCEL model `Accel` on
[#4662](https://github.com/compute-verification/verity/pull/4662), with the clocks of `Timed` on
[#4787](https://github.com/compute-verification/verity/pull/4787). The PR whose Lean covers this document deletes it.
The numbers are D6 v1.1 (art:6926be1c98b7855aca699278551040538e83843d28b72fdf85044cfd291bf3ef, CMP-118), and the segment
and noise periods NOFOLD-OPT's (see "The segment and the noise period").

**Names** (Daniel, 26 Oct, 7:51 PM PDT): Dither is the protocol, PoUW's no-fold design, and the ACCEL model is the
physical model 𝓜 it is proved over (the Lean's `Accel`).

**The receipt is no-fold** (CMP-172, 27 Oct, 03:05Z). Each segment's 16-bit noised prefix sums are committed as SHA-512
hiding leaves before the check coin, and the sampled cells are opened after it; no word is folded into a keyed receipt.
The served PoUW moves to this receipt, and `SharedNoiseLucks`, the premise its folds' detection rests on, has no place in
Dither: detection rests on the value root's binding before the coin, SHA-512's extraction and the proved sampling rate.

Under "One proof service" (`AGENTS.md`), Dither owns what is committed, which proof blocks need proofs, the public Program
each must pass, and what it does with the outcome. Coins, selection, timing, proving, verifying and the record are the
service's. The second half of this document states what Dither needs from the service, each need as a requirement on the
challenger, the certifier or the firewall, with the premise it makes true.

## Terms

The Glossary's terms (`ARCHITECTURE.md`), the shared notation (CMP-158), and these:

- **Runs.** Dither is part of PoUW, one of the Glossary's subprotocols, and each of its runs goes through the Glossary's
  three phases: commit, challenge (the challenger draws live coins) and response. Every commitment is bound before any
  challenge (Daniel, 26 Oct, 11:09 AM PDT). Each of Dither's sampling blocks takes two runs: a dither run, whose response
  is the block's value roots, and a check run, whose commitment they are.
- **The ACCEL model 𝓜**: a period's devices as the Lean's `Accel` reads them: their resources (tensor cores, issue
  slots, shared-memory and register ports, device memory, links), each at its data-sheet rate, and each run's use of
  each. On the H200 and the RTX PRO 6000 it is read at 𝓜's floors (`Card.floors`, #4662). A price μ ≥ 0 is per unit of
  each resource; a run's cost at μ is Σ_r μ_r u_r.
- **pub_μ(b)**: sampling block b's least cost at μ over D5 v2.1's menu of public methods, which holds the named matmul
  programs (with route (iii) on the RTX PRO 6000) and the pinned SHA-512 schedules (the Lean's `PubBest`).
- **μ°**: the honest served period's dual prices, which follow from the period's public shapes.
- **Sampling block**: the proof blocks of a set of independent calls (none reads another's output), which go through
  their runs together. Unlike the Glossary's default, its draw is not stratified by subcircuit (see "The sampling
  design").
- **Proof block**: what one check proves with C-Flock `--zk`: one segment of one output cell of a call, the Glossary's
  proof block in Dither's partition.
- **W_b and W_p**: sampling block b's and proof block p's public cost, the honest cost at μ° from the public shape. W_b
  is the sum of its proof blocks' W_p. W is work at the period's prices, as in the Glossary's W_cert.
- **W_reg**: a replica's budget for a monitoring period, registered before its first coin.
- **W_F, W_cap and W_S**: a fleet's registered total, or its registered cap, and the total of its accepted replicas'
  W_reg (see "The fleet").
- **k**: the number of a replica's sampling blocks in a period, about 1e8 to 1e9 a week.
- **q**: the expected number of a replica's sampling blocks selected in a period. The service's draw sets it.
- **Replica**: the devices serving one copy of the model, behind one network certifier boundary, whose runs the challenger
  puts in one sequence. **G** is the number of replicas a claim registers.
- **W_ans**: the honest cost at μ° of the period's answers, the selected proof blocks' proofs.
- **ε_skip**: Dither's ε, the share of W_reg a period may skip: the Notation's ε_W, PoUW's part of the fleet's ε_skip
  (`GLOSSARY.md`, "Narrower senses"). **δ_coin** is the sampling failure probability and
  **δ_crypto** the cryptographic one. **k_infil** and **k_exfil** are the bits the certifier counts into and out of the
  replica.
- **F**: the adversary resource model's total over a period: census `world-inference-2027`, 9.36e30 FP8 FLOP a year
  (#4731).

The Lean predates these names. Its round is a sampling block, and its word is a proof block's committed value. Its `k`
and `nsb` are k, and `k_draw` is q. Its `C` is W_reg, `H` is W_b, `H_o` is W_ans and `δ` is δ_coin. Its fleet's unit is
a replica, and θ is 𝓜. A rename follows #4665 and #4662's landing. Two premises have new names (Daniel, 26 Oct, 6:13 PM
PDT), which the Lean takes in DITHER-E2E's PR: SHA-512's premise is SAK for SHA, `SAKSha` (the Lean's `ShaSpeed`), and
the proof service's "the proof system is sound" is `ProofSystemSound` (the Lean's `ServiceSound`).

## What is committed

**A call** computes O = (A + d_A)(B + d_B)ᵀ over the integers, with int8 operands: A is m×K activations and B is N×K
weights (for PV, A is P and B is the value cache).

**Noise.** The live coins give each call its dithers d_A ∈ {0,1}^{m×K} and d_B ∈ {0,1}^{N×K}, and HYB noise:
- U is N×L′ and V is m×L′, both uniform on ±80, periodic along K with period L′.
- U's columns are permuted afresh (π_τ) in every noise tile τ.
- A segment is t noise tiles, so L = t·L′. L, the word interval (one 16-bit word a cell every L inner indices), and L′,
  the noise period, are separate parameters (see "The segment and the noise period").

**The tensor operands** are X = wrap8(A + d_A + V) and W′ = wrap8(B + d_B + U π_τ), tile by tile, with wrap indicators
w_X and w_W.

**Per output cell (i, j)**, the committed values are:
- the prefix sums T_s = Σ over the first s·L inner indices of X W′ᵀ, mod 2^16, for s = 1..S − 1 with S = ⌈K/L⌉;
- O at 32 bits, bound to them by O = T_S + 256(w_X W′ᵀ + X w_Wᵀ + 256 w_X w_Wᵀ) − Z_X Uᵀ − V Y_Wᵀ. Here
  Z_X = Σ_τ (A + d_A + V)_τ π_τᵀ, Y_W = Σ_τ (B + d_B)_τ, and T_S is taken at full width.

T_S is implied by O, not committed: the s = S proof block, which opens the whole rows, derives it from O by that
identity (NOFOLD-OPT; it takes no premise). At L ≥ K a cell commits O alone.

**Leaves and roots.** Every root is a binary frame-v3-sha512 tree of hm96-sha512/row/v2 rows of 16 KiB, hiding leaves
under the prover's salts (`verity.ml.commitments`: `rowleaf`, `hm96`, `merkle`). A 16 KiB row costs 8.6 SHA-512
compressions a KiB, against 8.9 for an 8-chunk leaf in a 16-ary tree, so Dither has no leaf format of its own
(NOFOLD-OPT, CMP-172).
- Each call has an operand root, bound in its sampling block's dither run's commit phase, and a value root holding its
  T_s and O, bound in that run's response phase (see "A sampling block's runs"), both before the block's check coin.
  The sampled cells are opened after it.
- The weights are in the period's weight root, bound once a period, before any of the period's coins.

**The value root's layout**, pinned for kernels (KCOMMIT-FB, HASH-27). It is V31-GPU's `v31.py` layout at 16 KiB rows,
under Dither's tag and without segment S's rows:
- A call's m × N output is cut into 128 × 128 tiles, tile t = t_m·(N/128) + t_n, each tile's cells row-major
  (c = 128 r + col). m and N are multiples of 128, and L′ a multiple of 32 dividing K.
- A tile's words of segment s, for s = 1..S − 1, are its 16,384 T_s as little-endian u16 (32 KiB, two rows). Its O
  values are 16,384 little-endian u32 (64 KiB, four rows). So a tile takes P = 2(S − 1) + 4 rows, and no row is padded:
  - row t·P + 2(s − 1) + r, for r = 0, 1, is bytes [16,384 r, 16,384 (r + 1)) of tile t's segment-s words;
  - row t·P + 2(S − 1) + h, for h = 0..3, is the O of tile t's cells 4,096 h to 4,096 h + 4,095.
- A row's leaf is hm96's leaf over the row's `sha512/row/v2` digest (131,072 bits), salted with `prf_salt(Ks, off + pos)`:
  192 bytes of commit_gpu's PRF (`benchmarks/pouw/commit_gpu/reference.py`), keyed by the prover's private Ks, at the
  call's salt offset off plus the row's position pos. No coin feeds a salt. frame-v3-sha512's leaf prefix (the domain id
  and the position) is `row_leaf`'s, around the hm96 digest.
- The tree is binary, one a call (`merkle.tree_levels`). Its domain is `CommitmentDomain(identity_digest_sha512(TAG,
  label), -1, RangeIndexedDomain(0, leaves), hash="sha512")`, with TAG = `verity/pouw/dither/call/v1` and label
  {m, n, k, lp, t, row, leaves}.
- A proof block (i, j, s) opens the rows holding T_s (for s < S) and T_{s−1} (for s > 1), and for s = S the row holding
  O. The filler's leaves are this layout without O's rows (DITHER-FILL, `verity/pouw/dither/fill/v1`).

**The prover hashes the words on-chip**, in the GEMM's epilogue, and doesn't write them to DRAM: they are 2/L bytes a MAC,
0.49 tensor-times of the RTX PRO 6000's GDDR7 at L = 1,024, and hashing them from DRAM doubles that. An opened row is
recomputed on demand rather than stored.

**Each call class's operands.** Every operand a credited call reads is bound before its sampling block's seeds: in the
weight root, in an earlier block's value root, or in the call's own operand root.

| call class | A | where A is bound |
|---|---|---|
| QKV (V4's q_a and kv_a), gate-up, the experts' gate-up, the LM head | RMSNorm of the residual stream; for the experts, the rows the router gathers | A itself, in the call's operand root |
| PV | P, from QKᵀ, which is served plain | P itself, in the call's operand root |
| O-projection | PV's O under the model's public elementwise map | PV's value root, with the map's per-row scalars in the call's operand root |
| down-projection, the experts' down-projection | SiLU·mul of the gate-up's O | the gate-up's value root, with the per-row scalars in the call's operand root |
| V4's q_b and kv_b | RMSNorm of q_a's and kv_a's O | their value roots, with the per-row scalars in the call's operand root |

B is the weights, in the weight root, except for PV. PV's B is the value cache: earlier calls' O under the public
elementwise map, in those calls' value roots, with the map's scalars in the PV call's operand root. A call whose operands
aren't bound this way is served plain.

An operand bound only with its call's value roots, after the seeds, could be chosen as A = −V − d_A. Then X = 0, every
check passes, and O = −V·Y_Wᵀ costs L′/K + 1/m of the call's MACs. That frees 24–63% of a forward pass's credited MACs
(RT-POOL's run M).

**What is credited.** Static matmuls and attention's PV, with O bound and a fresh dither bit on P. QKᵀ and V4's
indexer are served plain and uncredited.

**The segment and the noise period** per class of sampling block follow the 07:04Z rule. The pair (L, L′) maximizes
credited value per honest second, pub_μ°/T_honest, subject to γ ≤ 10 for that class. A class with no pair at γ ≤ 10 is
credited at its least-γ pair only where that adds credited value per honest second; otherwise it is served plain.
NOFOLD-OPT's picks (CMP-172, runs r20261027-031621-5469 and r20261027-032610-3a37), all with T_S implied by O and
16 KiB rows, γ being the call's time over the plain s8 GEMM's at η 0.85 with the hashing overlapped:
- static, RTX PRO 6000: L = 1,024 with L′ = 128 (t = 8), at γ 1.10 / 1.05 / 1.15 for 8B / 70B / V4, when registration
  pins B (the weight root is a published checkpoint's, or the weights expand from a public seed). The credit rests on
  that pin.
  - **Decided** (compute-research, under Daniel's delegation, 26 Oct, 8:46 PM PDT): the demo registers a published
    checkpoint's weights pinned by digest, such as Llama-3.1-8B and 70B quantized to s8, which meets the condition, so
    the RTX PRO 6000 runs L = 1,024 with L′ = 128.
  - For weights the prover chooses, the general case, the setting that leaks nothing is L = 768 with L′ = 256 (t = 3),
    at γ 1.38 / 1.22 / 1.53.
- The decided L = 4,096 to 8,192 at L′ = 512 is peel-bound on the RTX PRO 6000, at γ 1.20 to 1.41: the peel costs 4L′/K
  of extra tensor work, so what lowers γ is a small L′ at t ≤ 8, not a large L. t stays at most 8 because a proof
  block's cost at A = 0 falls as c_t/t, and at t = 8 it is 1.15 times the public best.
- static, H200: hash-bound, at γ 1.40 to 2.15 (L ≈ K, t ≤ 8). Its levers are the SHA-512 kernel and the filler, not L.
- PV: an L′ for each attended length (D6 v1.1).

ε_skip is the same at every L.

## A sampling block's runs

Before any of a period's coins, the replica registers (see "Registration") and binds the weight root. The certifier
signs its registration receipt of both, and the challenger draws the period's weight-side coins from it. Then the
replica's sampling blocks follow in one sequence, each through two runs.

**Block b's dither run.**
- **Commit phase.** It starts at block b − 1's value-root receipt (at the registration receipt for the first block). The
  replica sends b's public shape (its calls, their classes and shapes, and W_b) and each call's operand root. The
  certifier signs the operand-root receipt.
- **Challenge phase.** The challenger draws b's seeds from the operand-root receipt (see "Coins"), and they enter the
  boundary at t_b. No seed of b exists before that receipt, and none goes out before block b − 1's value-root receipt.
- **Response phase.** The replica runs b's calls on the noised operands and sends their value roots, due within Δ:
  arrive_b ≤ t_b + Δ. A late root fails the period. The certifier signs the value-root receipt.

**Block b's check run.**
- **Commit phase**: the value roots, bound by the value-root receipt.
- **Challenge phase.** The challenger draws b's check coin from the value-root receipt, so tc_b ≥ arrive_b. The coin
  carries either "not selected", with probability 1 − π_b, or the one proof block of b to prove (see "The sampling
  design"). One coin carries both, so the answer is timed from the coin `Timed` reads as tc.
- **Response phase**, if b is selected: the proof of its proof block, which reaches the certifier within τ_ans of the
  check coin: ansAt_b ≤ tc_b + τ_ans. It runs beside later blocks' runs (see "Timing"). An unselected block's check run
  ends at its check coin.

Block b + 1's dither run starts at b's value-root receipt, beside b's check run.

#4787's `Timed` reads this order as `arrive ob b ≤ t ob b + Δ`, `arrive ob b ≤ t ob b'` for b′ = b + 1,
`arrive ob b ≤ tc ob b` and `ansAt ob b ≤ tc ob b + τans`. It doesn't yet read the dither run's commit phase (see
"Open").

## The proof block and its Program

**A proof block** is one cell (i, j) and one segment s of a call. Its Program opens these from their hiding leaves,
inside the proof:
- A's and B's row segments s (for s = S, the whole rows A[i,·] and B[j,·]), each from where its class binds it;
- T_{s−1} (with T_0 = 0), and T_s for s < S or O for s = S, from the call's value root.

The call's coins are public inputs. **Its only output is its pass bit.** A proof block's outputs are at most 16 bits, and
T_s and O are exact functions of private operands (O of the LM head is a user's logit). Published, they would be
16 + 32/S bits a check: 24.2 bits at H200 V4, 41.7 kB a replica-week.

**The Program** is one Definition for each (K, L′, t, s = S or not, and how the class binds A), with the call's coins as
public inputs. It rebuilds the segment's X and W′ from the opened operands and the coins, and checks
T_s − T_{s−1} ≡ Σ X W′ᵀ (mod 2^16) over the segment. For s = S it takes T_S from O by the binding identity above,
mod 2^16, from the whole rows, their wraps and the coins, and also checks O ≡ Σ_ℓ (A + d_A)_{iℓ}(B + d_B)_{jℓ}
(mod 2^32).

C-Flock `--zk` proves it on the committed values. The Definition and its circuit-check sidecar are not written yet.

O is checked only in its cell's last segment, never in a proof block of its own. At A = B = 0 its 32 bits alone are
guessed with probability 0.1145 at K = 64, and its high word with probability 1.

## The outcome

**Acceptance.** The period is accepted only if all of these hold:
- every selected proof block's proof verifies;
- the certifier's signed times follow each run's order: every operand-root receipt before its block's seeds, every value
  root within Δ of its seeds, every check coin after its value roots, and every answer within τ_ans of its check coin;
- its signed counts k_infil and k_exfil are within E_in and E_out, caps set before the period.

Otherwise the period is refused. A failed proof is a visible failure and is never retried.

**Credit.** An accepted period credits Σ_b pub_μ°(b) over its sampling blocks. It publishes W_reg, q and ε_skip at δ_coin,
which is what its draw established, and a fleet's W_F or W_cap beside W_S.

**The bound.** Except with probability δ_coin + δ_crypto + k(κ_dmm + κ_dsha), the credited work plus any other work the
replica's devices did in the period, valued at μ°, is at most the honest optimum plus ε_skip·W_reg (`WorkSound`). Here:
- δ_coin is the draw's failure probability;
- δ_crypto = κ_S + κ_x(A) + κ_sha(A): the proof service's soundness error, and SHA-512's as a random oracle in the
  extraction and in `SAKSha`;
- κ_dmm and κ_dsha are `SAKMatmul`'s and `SAKSha`'s degenerate-dither terms at the prover's advice B_adv(A), paid once
  per sampling block.

**Its budget.** For a fleet of G replicas to keep these terms within δ_coin in total, each replica's are at most
δ_coin/G, and each block's κ_dmm + κ_dsha at most δ_coin/(G·k).
- At H200 V4 (δ_coin = 1e-6, G = 1.15e7, k = 1e9 a week), that is 8.67e-14, about 2^-43, a replica and 2^-73.3 a
  block. Across D6's rows a block's budget is 2^-70 to 2^-75 (RT-POOL's run N1).
- κ_x(A) holds per prover. One value for every prover would be 2^-310 (2^-308 in the worst row) once the answers' oracle
  calls are logged after the last root (RT-POOL's run L1b). That awaits the Lean: RP20, and the walk's oracle reach,
  which leaves the one-budget `WorkSoundNoFold` vacuous.
- κ_dmm and κ_dsha have no value yet. B_adv can be as large as a replica's memory, 1.35e13 bits against a block's 5.63e7
  seed bits, so they can't come from counting dithers against advice: they are claims about the product's structure.
- Until those values exist, a claim states ε_skip at δ_coin with these terms per prover, for any prover whose oracle calls
  fit the period's devices, and gives their form.

ε_skip = (ε_S + ε_net)(1 + W_ans/W_reg), with:
- ε_S = ln(G/δ_coin)/(ηq), each replica at δ_coin/G, with G the claim's registered replicas. Here η = 1 − p, where p is
  the most predictable proof block's measured free-guess rate: p ≤ 2.79e-5 on the H200 (T_1 at L = 64) and p ≤ 2.02e-5
  on the RTX PRO 6000 (T_1 at L = 384). D6 takes η = 1 − 2.8e-5 on both.
- ε_net = β(k_infil + k_exfil)/W_reg ≤ β·E/W_reg (P18), the channel term priced at β a bit while credit is work at μ°.
  β = n_max·W_mac, where W_mac is the honest cost of one MAC of the sampling block (hashing included), and n_max = N, 128,256 for
  the Llamas and 129,280 for V4. E = E_in + E_out:
  - tokens crossing the replica's boundary at 4 bytes a token (4,500-token requests; about 6.5 times that if HTTP/SSE
    crosses it);
  - in E_out, the answers' proofs, about 0.865 MB each, so about 12 GB a replica-week at q = 13,816. That is 0.75% (H200
    8B) to 10.1% (RTX PRO 6000 70B) of E, and it raises ε_net by at most 0.0006%. The size is C-Flock `--zk`'s on the
    RoPE rows, since the proof block's Definition isn't written.
- Over P one-week periods, P26 adds ln(P)/(ηq): +0.011% a month and +0.029% a year.

## What Dither needs from the service

### The sampling design

Each sampling block b is selected independently with probability π_b = q·W_b/W_reg, drawn live after b's value roots.
Inside a selected block, its proof block p is drawn with probability W_p/W_b.
- Each proof block is then checked with probability q·W_p/W_reg. That is `SAKMatmul`'s rate condition
  η·pub ≤ π·s·(1 − p) at s = 1, given W at μ° (pub ≤ W for each proof block).
- Uniform weighting inside the block fails the condition: a prover would skip the proof blocks whose share of cost
  exceeds their share of checks. So this weighting is required. The Lean's `NoFold.cells` takes any rule, and
  `SAKMatmul` holds only at this one.
- π_b ≤ 1 needs W_b ≤ W_reg/q (see "Registration").
- q is the service's. D6 takes q = 13,816 a replica-week on average. The count is random, about √q = 118 either side,
  and 14,378 at the 1 − 1e-6 quantile on the H200. Dither surfaces ε_skip at δ_coin for whatever q the draw sets.

**This is a sampling design main doesn't have**, proposed name `poisson:Q`:
- `bernoulli:NUM/DEN` and `rate:NUM/DEN[:T]` give every sampling block one flat rate (`rate` as a subset of size ⌈qN⌉), not a
  rate proportional to its cost.
- `work:K` and `stratified:K` draw a fixed-size subset per template stratum once, over the whole registration. Dither
  has no strata, a random count, and a draw interleaved with serving.

It needs a law object beside `draw.parse_law`'s and the Lean verifier's `Flock.Draw`. Its acceptance probability bound
is `WorkSound`'s sampling term, a moment bound at rate ηq/W_reg a unit of cost, rather than `work_escape_le`. The
integrity profile has no field for a cost-weighted moment bound; that goes to @proofs, whose sampled proofs produce the
profile.

### Registration

- **W_reg is a budget**, registered before the period's first coin. Each block's W_b is fixed in its dither run's
  commit phase, before its seeds, with W_b ≤ W_reg/q. The challenger draws no seeds for a sampling block that would take Σ_b W_b past W_reg.
  Sampling blocks follow live traffic, so their shapes needn't be known when the period starts.
- An unspent budget only raises ε_skip against the credit: the bound is ε_skip·W_reg whatever share of W_reg is served.

**The fleet.** A claim over several replicas registers its fleet before any member's first coin, in one of two forms:
- its members and every member's W_reg,u, with total W_F; a replica that starts later joins the next fleet period;
- or a fleet cap W_cap, with members joining while Σ_u W_reg,u ≤ W_cap.

The claim publishes W_F or W_cap beside W_S. A fleet named after the period's outcomes defeats a pooled bound: at
δ_coin = 1e-6 the claim is then false with probability 0.99999 (RT-POOL's run O1). Grown one replica at a time with no
cap, it multiplies δ_coin by up to the harmonic number H_G, ×16.8 at H200 V4 (O2).

### Coins

The challenger issues every live coin, in each replica's one sequence of runs; under the MVP it runs on the replica's
certifier device (see "The boundary"). Before the monitoring period it draws a random coin c for each draw and commits
to them in one root (see "The coin commitment"). Once the receipt a draw follows is in, it opens that draw's c against
the root and derives the draw's coins from c and the receipt, inside the challenger,
as `derive(c, "verity/pouw/dither/<kind>/v1", ctx)` (`verity.ml.randomness`: SHA-512 over the domain tag, the coin and
the receipt, as v3.2's RP25 and RP26 derive them, under tags distinct from PoUW's noise-seed tags), where:
- c's opening is checked by the certifiers against the root they hold privately. The prover gets no Merkle path and no
  root before its last root, and receives the paths and the root at close (@proofs, 27 Oct, 04:16Z; ONE-4's withheld
  mode, #4888): opening draw j to it early would hand it the root before later blocks' roots;
- kind is `weights` (the weight-side coins), `seeds` (a block's dither and noise seeds) or `check` (its check coin);
- ctx = {period, replica, sampling block index, receipt}, where receipt is the digest of the certifier's signed receipt
  of what the coin follows: the registration for `weights`, the block's operand roots for `seeds`, and its value roots
  for `check`.

So a derived coin never has to open against the root; its c does. No c serves two draws, no context repeats, and no
coin can be computed before the receipt it reads. Derived from roots alone, a
repeated junk root would repeat its "not selected", and repeated operands would repeat their dither: either raises
ε_skip to the credited share. This is the premise POOL-THM's `IndependentCoins` rests on, with SHA-512 as the
derivation's random oracle.

Only expanded coins enter the boundary, never a c, which is stricter than ONE-4's withheld mode, where the prover gets the
raw coin. A c sent in for local expansion would make a whole block's outsourcing cost one coin's bits, and P18's β would
fail.

**The coin commitment** (soundness's ruling, Daniel, 26 Oct, 6:15 PM PDT; settled 27 Oct, 03:45Z). The challenger
commits to its coins c before the monitoring period, and the commitment reaches the prover only after the prover has
committed, so no soundness statement takes a hiding premise on it.
- The public configuration names each replica's drawing challenger, its identity and LMS key, and never its coin root.
- Before the monitoring period, the challenger sends its committed root to the network certifiers over their private
  links (`WitnessPathPrivate`), never to the prover first.
- A draw's c opens against the root, to the certifiers, only after the receipt the draw follows, and the draw's coins
  are derived from c and that receipt (see above). So the challenger can't choose a block's seeds after seeing its roots, and the prover
  can't know them before it sends the roots.
- A multi-server replica has one draw, from the challenger on its lowest-numbered server. Each server's certifier exempts
  from k_infil only the coin frames derived from a c that opened against that root, with the receipt they name, and
  counts everything else as ingress (RP5). Each
  certifier re-clocks the forwarded coins on its own ingress grid (RP6), with the certifiers' clocks synced under the
  one-clock rule (see "The boundary").

**Seeds.** Every credited coin comes from streamed 512-bit seeds, each covering at most 512·N_max MACs, that is, at
least 1/N_max unpredictable bits a credited MAC. That is 0.97 to 2.38 GB/s a GPU, and 7.7 to 19.0 GB/s an 8-GPU node.
- Each replica's seeds are its own: no stream is broadcast across replicas.
- Dither takes no Fiat–Shamir exception: its seeds are live coins, not main's per-matmul noise seeds.

**Weight-side coins** (d_B, U and the permutations) are drawn per replica from the registration receipt, after the
weight root is bound, and never shared between replicas. They are reused across the period's calls as part of the registered instance (P14, P17).

### The boundary

The network certifier watches the replica's boundary, signs its receipts and arrival times, and counts k_infil and
k_exfil. Dither needs these of the challenger and the certifier:

- **Coins stay confidential until they cross the boundary** (`CoinsConfidential`, under `trusted-host`). No machine
  outside the replica's boundary may hold a coin before the coin enters the boundary, and afterwards only through bits
  the certifier counts. Both it and its placement premise are prose only (no `.lean` file defines either): the premise
  `WitnessPathPrivate` discharges it: private messages (the devices' signed records and salts, the prover's witness, a
  server's coins) travel only on private links, and the certifier sits on the server's own uplink (Daniel, 26 Oct,
  4:41 PM PDT).
  - **The MVP: the certifier's device is the challenger** (Daniel, 26 Oct, 1:30 PM PDT). The device that runs the
    replica's network certifier produces its live coins, so each coin is made at the boundary it enters: no path carries
    one in from outside and no key is needed for it. A multi-node replica's challenger is the certifier device of its
    lowest-numbered server, and its coins reach the other servers' certifiers only over the auditor's private links.
  - After the MVP, a challenger on a device of its own may reach the certifier only over a private link. A keyed path
    across hosts the prover controls is the secure-transport workstream's. A SHA-512 keystream with HMAC-SHA-512, at
    the measured 0.86 GB/s a core, would take about 27 certifier cores for an H200 V4 replica's 11.6 GB/s and 56 for an
    RTX PRO 6000 V4 replica's 24.0 GB/s.
  - The challenger's side is two premises, named as in `verity/core/protocols/sampled_proofs/README.md`:
    - `CoinsConfidential`: the challenger's coin stream reaches the network certifier only over a path on auditor
      hosts, or the challenger and the certifier are one device at the boundary. In the Lean it is the named Prop
      `Pouw.Assumptions.CoinsConfidential`, listed in the lock, which `FleetSound` takes: a replica's strategy reads
      none of its own coins.
    - `OneClock`: the challenger's release times and the certifier's ingress and receipt times are read on one clock,
      and a challenger on a device of its own keeps its clock within 1.03 s of the certifier's (the one clock below).
      It is a spec premise: the Lean reads every time on one axis and names no Prop for it.
  - Without it an outside machine reads the dither on the path and serves the sampling block, and ε_skip rises to the credited
    share. The replica imports only its roots: at H200 V4, E leaves 27.1 MB a check against 0.46 MB for a V4 decode
    call's activations.
- **The exclusion is exact** (what makes `CertifierSeesImports` true). Under the MVP the drawing server's coins are made
  on its certifier's device and are never imports there. Every other server's certifier, and any certifier whose
  challenger has a device of its own, excludes them only so (RP5):
  - k_infil leaves out only a coin frame derived from a c that opened against the drawing challenger's committed root,
    with the receipt it names, and carrying its expected (period, replica, sampling block, kind, sequence number) once,
    at its expected length.
  - Every other byte counts: malformed, unverifiable, duplicate, replayed or late messages into k_infil, and the
    stream's reverse traffic (acknowledgements, flow control) into k_exfil.
  - The certifier holds the committed root, the challenger's LMS public key from the configuration, and the schedule
    of expected frames and lengths. It needs each opened c only to check the frames derived from it.
- **Each certifier re-clocks the stream** (RP6). It terminates the forwarded stream and sends it into its boundary on
  its own ingress grid: fixed sizes, fixed order, no flow control from inside, and a drop is a visible failure.
  Otherwise the packets' delays and order carry uncounted bits, and V4's ε_net grows ×1.04 to ×7.3.
- **One clock.** The certifier stamps the coins' entry into the boundary (t_b, tc_b) and every receipt and arrival
  (the operand roots, arrive_b, ansAt_b) on its own clock, which under the MVP is the challenger's too. A challenger of
  its own would have to keep its clock within 1.03 s of the certifier's, and so does every other certifier of a
  multi-server replica against the drawing server's.
- **Receipts are authenticated.** The challenger learns that roots are in only from the certifier's signed receipts,
  whose digests are in the coins' contexts and messages. A forged receipt would draw a check coin before the roots.
- **A multi-node replica has one boundary**, carried by certifier-to-certifier traffic on auditor-private links under
  `WitnessPathPrivate`, never through the prover's network, with the replica's certifiers keeping one count, each
  stamping its own server's ingress on its synced clock. Everything else crossing between its nodes counts.
  V4's replica is 12 H200s (1.5 nodes) or 17 RTX PRO 6000s (2.1 nodes). Counting its inter-node expert and KV traffic
  would fail honest replicas, and excluding that traffic by address would let an outside machine join as a node.

**What each device signs.**
- The challenger, per coin message: the period, replica, sampling block, kind, sequence number, issue time and the
  payload's digest. A check coin's message also carries the value-root receipt's digest. It signs them with LMS in Merkle
  batches, one signature per batch root. A challenger of its own is checked live by the certifier with HMAC-SHA-512.
  It signs its committed coin root once, for the certifiers (see "Coins").
- The certifier: its receipts (operand roots, value roots, answers), the coins' entry times, k_infil and k_exfil, and its
  counts of rejected messages.
- All of it is witness and never reaches the auditor in the clear.

### Timing

- T = 1 week and Δ = 30 s.
- **Answers.** Each selected sampling block's answer is one outer proof, which the firewall releases on its own. Its
  statement, the proof block's Program on the call's committed values, is fixed by the value-root receipt and the check
  coin before its run's challenge phase. A replica's answers are a stream of one-statement batches, run concurrently
  beside its later sampling blocks and never serialized. τ_ans includes the wait for the firewall's release slot.
- **No q(Δ + τ_ans) ≤ T.** #4787's `Timed` covers any number of sampling blocks, with answers beside later ones: their
  sampling periods telescope to T. Selections arrive about every 43.8 s a replica (T/q).
- **τ_ans isn't set.** v3.1's 10 s on an "8 s latency floor" rested on run r20261024-092255-2edd, which failed (rc 2) and
  measured no Dither proof block. τ_ans is to be set from a passing `--zk` run of the proof block with live coins, the
  release slot included, with a wide margin.
  - One late answer refuses the week. An honest week fails 76.3% of the time if 1e-4 of answers are late and 1.43% at
    1e-6, and failing at most 0.1% of honest weeks needs a lateness rate ≤ 7.0e-8 (RT-POOL's run J).
  - The margin costs only the end guard: no coins go out in a period's last Δ + τ_ans, which leaves (Δ + τ_ans)/T of
    the period uncredited, 0.015% at τ_ans = 60 s.
- An answer's cost is in ε_skip through W_ans/W_reg, at full weight. If answers stalled the replica instead of running
  beside it, they would cost 22.8% of its time.

## D6 v1.1's headline

The H200 at measured hashing (4.79 G/s), the RTX PRO 6000 at 7.13 G/s, credit at μ°, ε_skip at δ_coin = 1e-6 for one
period.

| card, model | GPUs a replica | ε_skip | γ | coverage | perplexity | credited share at μ° |
|---|---|---|---|---|---|---|
| H200, 8B | 8 | 0.275% | 5.11 | 96.4% | +1.38% at 2K, +3.39% at 8K | 59.9% |
| H200, 70B | 8 | 0.228% | 4.40 | 98.0% | +3.01% at 2K (8K not run) | 59.0% |
| H200, V4 | 12 | 0.225% | 6.06 | 96.6% | not run | 60.7% |
| RTX PRO 6000, 8B | 8 | 0.282% | 1.58 | 96.3% | +1.38% at 2K, +3.39% at 8K | 84.0% |
| RTX PRO 6000, 70B | 8 | 0.233% | 1.35 | 98.0% | +3.01% at 2K (8K not run) | 85.5% |
| RTX PRO 6000, V4 | 17 | 0.228% | 1.61 | 95.9% | not run | 84.3% |

| measure | target | worst | ratio to target |
|---|---|---|---|
| ε_skip at δ_coin = 1e-6, one period | ≤ 1% | 0.282% | 0.28 |
| γ, overlapped, with the answers | ≤ 10 | 6.06 | 0.61 |
| coverage of matmul MACs | ≥ 95% | 95.9% | 1.009 |
| perplexity | < 5% | +3.39% | 0.68 |

Notes on the headline:
- ε_skip's terms, across the six rows:
  - ε_S is 0.218–0.226%, at G of 1.15e7–3.68e7 replicas;
  - ε_net is 0.0058–0.0546%;
  - W_ans/W_reg is 0.16–0.48%.
- Without the registration (P21's variant, the draw normalised by wall-clock time), ε_skip would be 0.270–0.424%.
- Coverage counts V4's 48.85e9 static MACs a token at their own shapes. With mHC's F32 GEMMs uncredited, V4's coverage
  is 96.4% on the H200 and 95.7% on the RTX PRO 6000.
- The γ column is D6 v1.1's. At NOFOLD-OPT's picks (see "The segment and the noise period"), the static calls' γ is
  1.05–1.15 on the RTX PRO 6000 with B pinned and 1.40–2.15 on the H200; the headline isn't restated at them yet.
- What v3.2 changes and D6 v1.1 doesn't price:
  - E with the answers' proofs raises ε_net by at most 0.0006%.
  - Hashing the operand roots (A for the residual-stream calls, P for PV) adds to γ and lowers the credited share.
    A row is K bytes against N(2S + 4) bytes of its call's values, so it adds little to the static calls. P is m bytes
    a head per attended position, more than PV's own values at long contexts, so the 07:04Z rule may serve PV plain.

**Per monitoring period, V4 as the world's request, credit at μ°** (P22's reading: μ°-priced device capacity, as FP8
FLOP-equivalents, not FP8 FLOP of other work). The bound is the union over the claim's replicas, and over the period's
one-week periods.

| card | period | F (FP8 FLOP) | credited | ε_skip·W_reg, δ_coin = 1e-6 | ε_skip·W_reg / F |
|---|---|---|---|---|---|
| H200 | week | 1.80e29 | 1.09e29 | 2.5e26 | 0.14% |
| H200 | month | 7.80e29 | 4.73e29 | 1.1e27 | 0.14% |
| H200 | year | 9.36e30 | 5.68e30 | 1.4e28 | 0.15% |
| RTX PRO 6000 | week | 1.80e29 | 1.52e29 | 3.5e26 | 0.19% |
| RTX PRO 6000 | month | 7.80e29 | 6.57e29 | 1.6e27 | 0.21% |
| RTX PRO 6000 | year | 9.36e30 | 7.89e30 | 2.0e28 | 0.21% |

What is scaled: the fleet is taken to be one card serving one model, and the claim to register all of it. So every
figure is that card's share of F, and G is F's GPUs at the card's FP8 rate over the replica's size. What is proved is the
per-replica bound, in Lean, under the premises below.

Pooled over the replicas, and with each replica's periods as one sequence (POOL-THM, not yet proved), RT-POOL's run D
gives ε_skip·W_reg of 1.17e26 a week and 5.24e26 a year at H200 V4, and 1.62e26 and 7.32e26 at RTX PRO 6000 V4.

## Open

- **The pooled bound over replicas** is POOL-THM's (CMP-152): tier A on the accepted set, weighted by each W_reg,u, with
  the fleet registered before any member's coin (see "The fleet"), and each replica's periods as one sequence. With it,
  ε_S = ln(1/δ_coin)/(ηq) at a fleet whose replicas all accept, and G leaves ε_skip. A headline taking the minimum of
  two tiers holds at δ_coin only with each tier at δ_coin/2.
- **κ's values** (see "Its budget") and the headline's single κ go to Daniel: κ_dmm and κ_dsha are `SAKMatmul`'s and
  `SAKSha`'s error terms.
- **The Lean** (WORKSOUND-5, stacked on #4787):
  - the dither run's commit phase: `reg b` read at the operand roots' arrival, `∀ b, opArrive ob b ≤ t ob b` in
    `Timed`, and the phase's oracle calls logged before root b;
  - the budget registration: `reg` adapted, fixed before each round's coins, with k padded to a fixed maximum by empty
    rounds;
  - RP20 (the answers' oracle calls under one budget), the joint outsourcing premise over the period's programs, and
    `SAKMatmul`'s reuse form over the challenger's draw for the weight-side coins (P17's κ_re). The coin stream's exact
    exclusion and the multi-node boundary are stated in `CertifierSeesImports` (RP5, RP9);
  - the Lean's word is 32 bits, while a proof block checks a 16-bit T_s (RP18, in `WorkSound`'s Open). Until the Lean
    is restated at v3.2's words, η is priced at the measured p ≤ 2.8e-5, and POOL-THM's ζ at 2^-16.
- **The challenger's and the certifier's specs** (network-accounting, infra and proofs): the challenger on the
  certifier's device (the MVP), the exact exclusion, re-clocking, one clock and the multi-node private links, as
  implemented.
- **Keyed transport**, in place of `WitnessPathPrivate`'s private links (pre-shared keys, ML-KEM for transport only, or
  keys set at provisioning), is the secure-transport workstream's. Whether ML-KEM may bend the one-hash rule is
  Daniel's.
- **The firewall** (@proofs, with C-Flock's ZK server):
  - that it proves a replica's answers concurrently, and its release schedule. If it can't, RT-SIZE's queued check
    coins are the fix (CMP-158, RS5), and `Timed` must then read the check coin's time apart from the selection's;
  - main's batch rule fixes every staged statement before any run's challenge phase. Dither's statements are staged
    through the period, each fixed before its own run's challenge phase but after earlier answers' releases, so the rule
    is to be read per statement for a stream (RS6).
- **τ_ans**, from a passing measured run (see "Timing").
- **The coin root** (see "Coins"): how a certifier checks a forwarded frame's derivation from its opened c is the
  challenger's spec, as remote bandwidth's is on #4968 and the network certifier's coin link's on #4946.
- **Credit becomes size** (the Glossary's Size, CMP-158): a sampling block's credit becomes its size. Then ε_skip is a
  share of the registered size, and the channel term stays in bits, k_infil and k_exfil, without ε_net's β.
- **μ° or a fixed published price** (Daniel's open Ask).
  - At μ°, plain int8 tensor work beside the credited sampling blocks is valued at little or nothing. Let x be its plain MACs
    for each credited MAC. At x = 0.25 it is 5.1–7.3% of the claim on the H200 and 10.8–13.0% on the RTX PRO 6000, and
    neither ε_skip·W_reg nor the uncredited column bounds it.
  - μ¹, the honest period's duals at x = 1, published as a parameter, removes 37–40% of that on the H200 and 78–80% on
    the RTX PRO 6000. Its cost is the credited share: pub_μ¹ is 92.3–96.4% of pub_μ° on the H200 and 69.8–85.2% on the
    RTX PRO 6000.
- **The premises**, named and unproved:
  - `SAKMatmul` (`pouw-sak-matmul`): no program computes the committed values more cheaply at μ than the public best.
    It is false wherever a known cheaper route is missing from the menu.
  - `SAKSha` (`pouw-sha512-speed`): nobody hashes meaningfully faster than the pinned public SHA-512.
  - Beside them are `OutsourceWithinEgress`, `CertifierSeesImports`, `RootAfterCoin` and `ProofSystemSound`,
    `ShaRealize` for the extraction (ASK-16, pending), and POOL-THM's `CoinsConfidential` (which `WitnessPathPrivate`
    discharges) and `IndependentCoins`.
  - There is no `SharedNoiseLucks` and no `UnitFixed` (NOFOLD-OPT): a word's 2^-16 guess sits in SAK-LB's λ.
  - `PeriodInventory` is held pending Daniel: whether host CPUs and DPUs are devices of the period.
- **The RTX PRO 6000's cells at x > 0 are upper bounds.** 𝓜's sm_120 joint issue rows are closed rows (A36), and they
  stay bounds until CALIB moves them to a current vector.
- **Perplexity** isn't run for V4, or for 70B at 8K.
- **Not written yet:** the proof block's Definition and its sidecar, the within-block weighting in Lean, the
  `poisson:Q` law, and D6 v2's prices of the operand roots' hashing.
