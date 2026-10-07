import Models.Pouw.Assumptions.PearlC.TTOut
import Definitions.Pouw.PearlC.DeviceRev1

/-!
# The named assumption TT_OUT, rev1 (trusted; `Prop`s taken as hypotheses, never `axiom`s)

TT_OUT(1/400) for Pearl-C with every promotion into a +0 total out of the credit (`DeviceRev1.lean`): the first of every
word out of the credit and `W_ref`, any later one debited by replay. It holds at any device record, per unit and per audit
tile. It asserts less than the granted rows: a unit correct under rev1 is correct under them, and earns at most their
credit (`ttOutPearlCDevRev1_of_granted`). That is an implication only: the granted rows are falsified by the
first-promotion skip (rating D), so it carries no evidence for rev1, whose rating B rests on the assessor's alone
(07:40Z, `repair-rows-rev1-and-v2-cap.md`).
-/

namespace Pouw.PearlC.Assumptions

variable {Q R S : Type}

/-- **TT_OUT(1/400) for Pearl-C at a device record `d`, rev1**: `TTOut` at `pearlCProtocolDevRev1 d sem ρ`,
`pearlCDomainDev d`, `γ₀ = 1/400` and `εPearlC`. It is `TTOutPearlCDev` with every promotion into a +0 total,
`RNE(+0 + S_g) = S_g`, left out of the credit: the first of every word out of the credit and `W_ref` (`k/(32·G) − 1` adds
per word, not `k/(32·G)`), and any later one debited (`laterZeroPromos`). The game, error and falsifiers are
`TTOutPearlCDev`'s. Without promotion (`G = 0`) it is `TTOutPearlCDev`.

- **Why they go.** Every program outputs the same ticket words with them skipped (the first: 3,360 of 3,360 words across
  30 in-domain cells, `r20260930-061318-99f0`), and neither the flags nor the cap remove them. Crediting the first
  falsifies the granted row wherever `add·m·n − qa·m·k > γ₀·creditDev`: at `m = n = 8,192`, for every in-domain
  `k ≤ 8,192` at H100 prices and `k ≤ 2,048` at sm_120's measured FADD. A later +0 total is rare (≤ 3.9·10⁻⁶ of later
  promotions on the H100 atom, none measured on sm_120) and moves no γ.
- **The other promotion adds stay credited.** Folding add `g ≥ 2` into the tensor-core chain (window `g` started from
  `T_{g−1}` instead of `+0`) changes the word on 1.6–85% of words at sm_120 and 40–100% at H100 (six families, one
  64 × 64 tile each at `k = 256` and `1,024`, `pearlc_promotion_fusion.py`), so no tile stays correct. Telling which
  words it leaves right costs a per-word FP32 or INT32 operation, no cheaper than the add. `TT_OUT` asserts that nothing
  else skips them.
- **The cost model prices a tensor-core instruction by its whole fragment** (`w1-complete/<device>`; working decision
  W1). `CM` charges an `mma.sync` or `wgmma` of any kind, shape, precision or sparsity its full shape's MACs at the
  record's price, however few of its output words or lanes a program uses: on sm_120 an E4M3 `m16n8k32` is
  16·8·32 = 4,096 units, and a BF16 `m16n8k16` is 2,048 MACs at 2. `CM` never prices a tensor-core instruction per output
  word. Under per-word pricing, per-word joint skip sets would count: 10–16% of a word just inside v2's cap, 2.8–9.4%
  inside v1's (`repair-rows-rev1-and-v2-cap.md`). This assumption would then be a different one, which those sets
  falsify. Every other instruction is charged its class price per instruction. `CostModel` is abstract, so this rule is
  part of which `CM` the assumption is stated at, not a property Lean checks.
- **The prices are the cheapest bit-exact implementation's** (C3's charging rule). On sm_120 an FP32 add is 8.00 per word
  issue-bound, 8.376 per FADD in the measured loop (`r20260930-063213-c94a`). sm_120a has no packed FP32 add (no FADD2):
  `add.rn.f32x2` compiles to two scalar FADDs (`r20260930-081301-747b`), so it does not halve the add. -/
def TTOutPearlCDevRev1 [Fintype Q] [DecidableEq Q] [Fintype R] [Fintype S] (CM : CostModel Q R S) (d : PearlCDevice)
    (sem : PearlCSem Q R S) (ρ : ℚ) : Prop :=
  TTOut CM (pearlCProtocolDevRev1 d sem ρ) (pearlCDomainDev d) (1 / 400) εPearlC

/-- **TT_OUT(1/400) per audit tile at a device record `d`, rev1**: `TTOutTile` at `pearlCProtocolDevRev1 d sem ρ` and
`pearlCTilesDevRev1 d sem ρ`. It is `TTOutTilePearlCDev` with rev1's credit, cap and `W_ref`. -/
def TTOutTilePearlCDevRev1 [Fintype Q] [DecidableEq Q] [Fintype R] [Fintype S] (CM : CostModel Q R S)
    (d : PearlCDevice) (sem : PearlCSem Q R S) (ρ : ℚ) : Prop :=
  TTOutTile CM (pearlCProtocolDevRev1 d sem ρ) (pearlCTilesDevRev1 d sem ρ) (pearlCDomainDev d) (1 / 400) εPearlC

/-- **`tt-out/pearl-c-h100-rev1`**: rev1 at the H100 record, the repair of `TTOutPearlC · Costs.adopted`, with `CM` the
H100's W1 accounting, a `wgmma` priced by its whole shape. -/
def TTOutPearlCH100Rev1 [Fintype Q] [DecidableEq Q] [Fintype R] [Fintype S] (CM : CostModel Q R S)
    (sem : PearlCSem Q R S) (ρ : ℚ) : Prop :=
  TTOutPearlCDevRev1 CM devH100 sem ρ

/-- **`tt-out-tile/pearl-c-h100-rev1`**: rev1 per audit tile at the H100 record, the repair of
`TTOutTilePearlC · Costs.adopted`. -/
def TTOutTilePearlCH100Rev1 [Fintype Q] [DecidableEq Q] [Fintype R] [Fintype S] (CM : CostModel Q R S)
    (sem : PearlCSem Q R S) (ρ : ℚ) : Prop :=
  TTOutTilePearlCDevRev1 CM devH100 sem ρ

end Pouw.PearlC.Assumptions
