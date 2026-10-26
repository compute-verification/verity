# ACCEL: protocol v3.1

ACCEL credits the int8 matmuls of a served model. Each call is computed on dithered and noised operands, its running
sums are committed while it runs, and live coins pick which of them the developer proves. An accepted monitoring period
is credited at the public best price of its registered work, and the work the developer could have done instead is
bounded by εW_reg. This is v3.1 (CMP-108's D1): v3 as adopted on 26 Oct, its L′ picked by the 07:04Z rule, with
RT-PROTO-2's P21 (registration), P23 (replicas) and P25 (timed answers).

The protocol is experimental. Its guarantee is stated in Lean on open PRs: `Pouw.Guarantees.Dither.WorkSound` and its
card instances `Pouw.Guarantees.DitherInstance.WorkSoundAt` on
[#4665](https://github.com/compute-verification/verity/pull/4665), over the machine model `Accel` on
[#4662](https://github.com/compute-verification/verity/pull/4662). The PR whose Lean covers this document deletes it.
The numbers are D6 v1.1 (art:6926be1c98b7855aca699278551040538e83843d28b72fdf85044cfd291bf3ef, CMP-118).

Under "One proof service" (`AGENTS.md`), this document states only what ACCEL owns: what is committed, which proof
blocks need proofs, the public Program each must pass, and what ACCEL does with the outcome. Coins, selection, proving,
verifying and the record are the proof service's.

## Terms

The Glossary's terms (`ARCHITECTURE.md`), and these:

- **The machine model 𝓜**: a period's devices as the Lean's `Accel` reads them: their resources (tensor cores, issue
  slots, shared-memory and register ports, device memory, links), each at its data-sheet rate, and each run's use of
  each. On the H200 and the RTX PRO 6000 it is read at 𝓜's floors (`Card.floors`, #4662). A price μ ≥ 0 is per unit of
  each resource; a run's cost at μ is Σ_r μ_r u_r.
- **pub_μ(b)**: sampling block b's least cost at μ over D5 v2.1's menu of public methods, which holds the named matmul
  programs (with route (iii) on the RTX PRO 6000) and the pinned SHA-512 schedules (the Lean's `PubBest`).
- **μ°**: the honest served period's dual prices, which follow from the period's public shapes.
- **Sampling block**: one batch of matmul calls whose coins go out together, with its noise, its commitment and its
  draw. Unlike the Glossary's default, its draw is not stratified by subcircuit (see the sampling design below).
- **Proof block**: what one check proves with C-Flock `--zk`: one segment of one output cell of a call. This is the
  Glossary's proof unit, which TERMS-BLOCK (CMP-150) is renaming.
- **W_b**: sampling block b's public cost, its honest cost at μ° from its public shape. W is work at the period's
  prices, as in the Glossary's W_cert.
- **Registration**: before a monitoring period's first coin, the developer registers every sampling block of the
  period, and the registration declares **W_reg** = Σ_b W_b.
- **q**: the expected number of sampling blocks selected in a replica's monitoring period.
- **Replica**: the devices serving one copy of the model, behind one network certifier boundary, whose coins the
  challenger serializes. **G** is the number of replicas the guarantee covers.
- **W_ans**: the honest cost at μ° of the period's timed answers, the selected proof blocks' proofs.
- **F**: the adversary resource model's total over a period: census `world-inference-2027`, 9.36e30 FP8 FLOP a year
  (#4731).

The Lean predates these names. Its round is a sampling block, and its word is a proof block's committed value. Its `k`
counts a period's sampling blocks and `k_draw` is q. Its `C` is W_reg, `H` is W_b and `H_o` is W_ans. Its fleet's unit
is a replica, and θ is 𝓜. A rename follows #4665 and #4662's landing.

## What is committed

**A call** computes O = (A + d_A)(B + d_B)ᵀ over the integers, with int8 operands: A is m×K activations and B is N×K
weights.

**Noise.** The live coins give each call its dithers d_A ∈ {0,1}^{m×K} and d_B ∈ {0,1}^{N×K}, and HYB noise:
- U is N×L′ and V is m×L′, both uniform on ±80, periodic along K with period L′.
- U's columns are permuted afresh (π_τ) in every noise tile τ.
- A segment is t tiles, so L = t·L′: t = 2 on the H200 and t = 3 on the RTX PRO 6000.

**The tensor operands** are X = wrap8(A + d_A + V) and W′ = wrap8(B + d_B + U π_τ), tile by tile, with wrap indicators
w_X and w_W.

**Per output cell (i, j)**, the committed values are:
- the prefix sums T_s = Σ over the first s·L inner indices of X W′ᵀ, mod 2^16, for s = 1..S with S = ⌈K/L⌉;
- O at 32 bits, bound to them by O = T_S + 256(w_X W′ᵀ + X w_Wᵀ + 256 w_X w_Wᵀ) − Z_X Uᵀ − V Y_Wᵀ. Here
  Z_X = Σ_τ (A + d_A + V)_τ π_τᵀ, Y_W = Σ_τ (B + d_B)_τ, and T_S is taken at full width.

**Leaves and roots.** Values go in SHA-512 hiding leaves (8 blocks, one 128-bit salt, 7,935 data bits) in a 16-ary
frame-v3-sha512 tree, one root per call.
- B is committed once a monitoring period.
- A is not committed again: it is the previous call's committed O under the model's public elementwise map, plus
  committed per-row scalars.

**What is credited.** Static matmuls and attention's PV, with O bound and a fresh dither bit on P. QKᵀ and V4's
indexer are served plain and uncredited.

**L′ per class of sampling block** follows the 07:04Z rule. It maximizes credited value per honest second,
pub_μ°(L′)/T_honest(L′), subject to γ ≤ 10 for that class. A class with no L′ at γ ≤ 10 is credited at its least-γ L′
only where that adds credited value per honest second; otherwise it is served plain. D6 v1.1 lists the picks:
- static, H200: L′ 512 at most shapes, 384 on V4's K 1536;
- static, RTX PRO 6000: L′ 256;
- PV: an L′ for each attended length.

**A sampling block** is a set of independent calls whose operand roots are in before its coins. A replica's sampling
blocks are one sequence: block b + 1's coins go out only after every root of block b is in. Its roots are due within
Δ of its coins, and a late root fails the period.

## Which proof blocks need proofs

**The sampling design.** Each registered sampling block b is selected independently with probability
π_b = q·W_b/W_reg (Poisson sampling, P15 and P20). The challenger draws its coin live, after b's roots are in. This
needs q·W_b ≤ W_reg for every b. A replica's period selects q = 13,816 blocks on average, and 14,378 at the
1 − 1e-6 quantile on the H200. One more live coin then picks one proof block of the selected block to prove, each with
probability its share of W_b. That weighting is proposed here; the Lean's `NoFold.cells` takes any rule. Each check is
against its sampling block's own public cost, so ε is a share of the claim at every price (P20).

**This is a new sampling design**, proposed name `poisson:Q`. None of main's designs expresses it:
- `bernoulli:NUM/DEN` and `rate:NUM/DEN[:T]` give every unit one flat rate (`rate` as a subset of size ⌈qN⌉), not a
  rate proportional to its cost.
- `work:K` and `stratified:K` draw a fixed-size subset per template stratum once, over the whole registration. ACCEL
  has no strata, a random count (q ± about √q = 118), and a draw interleaved with serving.

It needs four things:
- W_b for every b, and W_reg, public in the registration before the first coin;
- a live coin for each sampling block after its own roots, from the challenger, in the replica's order;
- a law object beside `draw.parse_law`'s, and the Lean verifier's `Flock.Draw`;
- its escape bound, which is `WorkSound`'s sampling term (a moment bound at rate ηq/W_reg a unit of cost) rather than
  `work_escape_le`.

**A proof block** is one cell (i, j) and one segment s of a call:
- Its inputs are A's and B's row segments s (for s = S, the whole rows A[i,·] and B[j,·]), T_{s−1} (with T_0 = 0) and
  the call's public coins.
- Its outputs are T_s, 16 bits. For s = S, O opens with T_S: 48 bits, in the 64-bit size class.

O is never a block of its own. At A = B = 0 its 32 bits alone are guessed with probability 0.1145 at K = 64, and its
high word with probability 1.

## The public Program each proof block passes

One Definition for each (K, L′, t, s = S or not), with the call's coins as public inputs. It rebuilds the segment's X
and W′ from the committed operands and the coins, and checks T_s − T_{s−1} ≡ Σ X W′ᵀ (mod 2^16) over the segment. For
s = S it also checks:
- O ≡ Σ_ℓ (A + d_A)_{iℓ}(B + d_B)_{jℓ} (mod 2^32);
- the binding identity above, mod 2^16.

Its inputs are opened from their hiding leaves inside the proof, and C-Flock `--zk` proves it on the committed values.
The Definition and its circuit-check sidecar are not written yet.

## Coins and timing

The challenger issues every live coin. ACCEL needs these from it, for each sampling block in order:
1. the dither and noise seeds, after the block's operand roots are in;
2. after its roots, the selection coin;
3. if the block is selected, the check coin.

**Seeds.** Every credited coin comes from streamed 512-bit seeds, each covering at most 512·N_max MACs, that is, at
least 1/N_max unpredictable bits a credited MAC. That is 0.97 to 2.38 GB/s a GPU, and 7.7 to 19.0 GB/s an 8-GPU node.
Weight-side coins (d_B, U and the permutations) may be reused across decode calls as part of the registered instance
(P14, P17).

**The network certifier** watches the replica's boundary, signs arrival times, and counts the bits that cross it in
either direction. Its import count excludes the challenger's signed coin stream.

**Timing (P25).**
- T = 1 week, Δ = 30 s and τ_ans = 10 s.
- An answer is one C-Flock `--zk` proof of the selected proof block: about 1 GPU-s with an 8 s latency floor
  (r20261024-092255-2edd). It runs beside the replica's later sampling blocks.
- A replica's selected blocks go through the firewall one at a time, so q(Δ + τ_ans) ≤ T. This gives
  13,816 × 40 s = 6.40 days, and 14,378 draws (the 1 − 1e-6 quantile) take 6.66 days, 95.1% of T.
- **The margin is thin.** At that quantile Δ + τ_ans ≤ 42.1 s, so with Δ = 30 s an answer has 12.1 s against its 8 s
  floor. A slower answer needs a longer T or a shorter Δ.
- No coins go out in a period's last Δ + τ_ans, which leaves 0.0066% of the period uncredited.
- If answers stalled the replica instead of running beside it, they would cost 22.8% of its time.

## The outcome

**Acceptance.** The period is accepted only if all of these hold:
- every selected proof block's proof verifies;
- the certifier's signed times have every root within Δ of its coins and every answer within τ_ans of its check coin;
- its signed counts into and out of the replica are within E_in and E_out, caps set before the period at served
  traffic.

Otherwise the period is refused. A failed proof is a visible failure and is never retried.

**Credit.** An accepted period credits Σ_b pub_μ°(b) over its registered sampling blocks. It publishes W_reg, q and ε at
δ, which is what its draw established.

**The bound.** Except with probability δ + κ, the credited work plus any other work the period's devices did, valued at
μ°, is at most the honest optimum plus εW_reg (`WorkSound`; κ is the service's and the premises' error, stated there).
A developer that registers more than it serves only raises its own bound.

ε = (ε_S + ε_net)(1 + W_ans/W_reg), with:
- ε_S = ln(G/δ)/(ηq), each replica at δ/G. Here η = 1 − p, where p is the most predictable proof block's measured
  free-guess rate: p ≤ 2.79e-5 on the H200 (T_1 at L = 64) and p ≤ 2.02e-5 on the RTX PRO 6000 (T_1 at L = 384). D6
  takes η = 1 − 2.8e-5 on both.
- ε_net = β·E/W_reg (P18). β = n_max·W_mac, where W_mac is the honest cost of one MAC of the block (hashing included),
  and n_max = N, 128,256 for the Llamas and 129,280 for V4. E = E_in + E_out, the certifier's caps, set at 4 bytes a
  token crossing the replica's boundary (4,500-token requests; about 6.5 times that if HTTP/SSE crosses it).
- Over P one-week periods, P26 adds ln(P)/(ηq): +0.011% a month and +0.029% a year.

## D6 v1.1's headline

The H200 at measured hashing (4.79 G/s), the RTX PRO 6000 at 7.13 G/s, credit at μ°, ε at δ = 1e-6 for one period.

| card, model | GPUs a replica | ε | γ | coverage | perplexity | credited share at μ° |
|---|---|---|---|---|---|---|
| H200, 8B | 8 | 0.275% | 5.11 | 96.4% | +1.38% at 2K, +3.39% at 8K | 59.9% |
| H200, 70B | 8 | 0.228% | 4.40 | 98.0% | +3.01% at 2K (8K not run) | 59.0% |
| H200, V4 | 12 | 0.225% | 6.06 | 96.6% | not run | 60.7% |
| RTX PRO 6000, 8B | 8 | 0.282% | 1.58 | 96.3% | +1.38% at 2K, +3.39% at 8K | 84.0% |
| RTX PRO 6000, 70B | 8 | 0.233% | 1.35 | 98.0% | +3.01% at 2K (8K not run) | 85.5% |
| RTX PRO 6000, V4 | 17 | 0.228% | 1.61 | 95.9% | not run | 84.3% |

| measure | target | worst | ratio to target |
|---|---|---|---|
| ε at δ = 1e-6, one period | ≤ 1% | 0.282% | 0.28 |
| γ, overlapped, with the answers | ≤ 10 | 6.06 | 0.61 |
| coverage of matmul MACs | ≥ 95% | 95.9% | 1.009 |
| perplexity | < 5% | +3.39% | 0.68 |

Notes on the headline:
- ε's terms, across the six rows:
  - ε_S is 0.218–0.226%, at G of 1.15e7–3.68e7 replicas;
  - ε_net is 0.0058–0.0546%;
  - W_ans/W_reg is 0.16–0.48%.
- Without the registration (P21's variant, the draw normalised by wall-clock time), ε would be 0.270–0.424%.
- Coverage counts V4's 48.85e9 static MACs a token at their own shapes. With mHC's F32 GEMMs uncredited, V4's coverage
  is 96.4% on the H200 and 95.7% on the RTX PRO 6000.

**Per monitoring period, V4 as the world's request, credit at μ°** (P22's reading: μ°-priced device capacity, as FP8
FLOP-equivalents, not FP8 FLOP of other work). The bound is pooled over the period, with the union over its one-week
periods.

| card | period | F (FP8 FLOP) | credited | εW_reg, δ = 1e-6 | εW_reg / F |
|---|---|---|---|---|---|
| H200 | week | 1.80e29 | 1.09e29 | 2.5e26 | 0.14% |
| H200 | month | 7.80e29 | 4.73e29 | 1.1e27 | 0.14% |
| H200 | year | 9.36e30 | 5.68e30 | 1.4e28 | 0.15% |
| RTX PRO 6000 | week | 1.80e29 | 1.52e29 | 3.5e26 | 0.19% |
| RTX PRO 6000 | month | 7.80e29 | 6.57e29 | 1.6e27 | 0.21% |
| RTX PRO 6000 | year | 9.36e30 | 7.89e30 | 2.0e28 | 0.21% |

What is scaled: the fleet is taken to be one card serving one model, so every figure is that card's share of F, and G
is F's GPUs at the card's FP8 rate over the replica's size, not a census. What is proved is the per-replica bound, in
Lean, under the premises below.

## Open

- **The union bound over replicas.** ε_S pays ln(G) for a union over the world's replicas (`FleetSound`). A pooled
  bound over replicas is POOL-THM's (CMP-152).
- **μ° or a fixed published price** (Daniel's open Ask).
  - At μ°, plain int8 tensor work beside the credited blocks is valued at little or nothing. Let x be its plain MACs
    for each credited MAC. At x = 0.25 it is 5.1–7.3% of the claim on the H200 and 10.8–13.0% on the RTX PRO 6000, and
    neither εW_reg nor the uncredited column bounds it.
  - μ¹, the honest period's duals at x = 1, published as a parameter, removes 37–40% of that on the H200 and 78–80% on
    the RTX PRO 6000. Its cost is the credited share: pub_μ¹ is 92.3–96.4% of pub_μ° on the H200 and 69.8–85.2% on the
    RTX PRO 6000.
- **The algorithmic premises**, named and unproved:
  - `SAKMatmul` (`pouw-sak-matmul`): no program computes the committed values more cheaply at μ than the public best.
    It is false wherever a known cheaper route is missing from the menu.
  - `ShaSpeed` (`pouw-sha512-speed`): nobody hashes meaningfully faster than the pinned public SHA-512.
  - Beside them are `OutsourceWithinEgress`, `CertifierSeesImports`, `RootAfterCoin`, `AnswerAfterCoin` and
    `ServiceSound`.
  - `PeriodInventory` is held pending Daniel: whether host CPUs and DPUs are devices of the period.
- **The Lean's period boundary counts every sampling block.** `WorkSoundAtX` assumes k(Δ + τ_ans) ≤ T over all k of a
  period's sampling blocks, with each answer before the next block's coins. v3.1 times only the q selected blocks'
  answers, and runs them beside later blocks. A served period has far more than T/(Δ + τ_ans) = 15,120 sampling
  blocks, so the instance doesn't yet cover it.
- **The RTX PRO 6000's cells at x > 0 are upper bounds.** 𝓜's sm_120 joint issue rows are closed rows (A36), and they
  stay bounds until CALIB moves them to a current vector.
- **Perplexity** isn't run for V4, or for 70B at 8K.
- **Not written yet:** the proof block's Definition and its sidecar, the within-block weighting in Lean, and the
  `poisson:Q` law.
