import Definitions.Pouw.PearlC.Defs
import Definitions.Pouw.Basic.Prob
import Definitions.Pouw.Basic.QTree
import Mathlib.Algebra.BigOperators.Group.Finset.Basic

/-!
# Pearl-C's γ-gap game and accounting (trusted)

The game of the integer statement (§3.3, `Pouw.Game.Defs`), for a protocol whose bound words are output words: per unit,
the ticket words (Pearl-C's pre-peel accumulator `C̃`) and the useful output words (`U`, after the checked clean-up and
before the unscale), both FP32 bit patterns, both checked bit for bit. Π's semantics (the forming, the promoted chain, the
peel) are `Pouw.PearlC.Defs`'; here the protocol's reference functions are fields, so the named instance takes them from
`Defs` once its bit-exact chain is fixed. On top: Pearl-C's accounting and domain, on `Defs`' `Params` and `Shape`.

~~~text
0. Oracle.      H : Q → R uniform (Q, R finite).
1. Workload.    U = (layout, weight codes): N ≤ 2^64 units, quantified outside the probability.
2. Preprocess.  σ = A.pre.eval H, never a function of the salt; its queries count in q.
3. Salt.        s : S uniform, independent of H.
4. Online.      τ = CM.run A.online H U A.act σ s, cost ≤ T and calls ≤ q for every H and s.
5. Score.       K = correctSet, the units whose committed words are correct and whose replayed debit is within the
                cap (units above the cap are rejected, so they enter neither sum); A wins if T / (1 − γ) < ∑_{u ∈ K} W_ref(u).
~~~
-/

namespace Pouw.PearlC

open Finset Pouw.Fp8Atom

/-! ## Workloads, transcripts, cost models -/

/-- The declared layout: `N` units `0, …, N − 1` (unit `u`'s index is `u`), their shapes and weight ids. -/
structure Layout where
  N : ℕ
  shape : ℕ → Shape
  wid : ℕ → ℕ

/-- A matrix of bit patterns: operand codes, or output FP32 words. -/
abbrev Codes : Type := ℕ → ℕ → ℕ

/-- A declared workload: its layout and its weight codes, fixed before the salt. -/
structure Workload where
  layout : Layout
  weight : ℕ → Codes

/-- The preprocessed state: any finite data. -/
abbrev State : Type := List ℤ

/-- One unit's commitment (ideal: the values themselves): activations, ticket words, useful output words. -/
structure Commit where
  act : Codes
  ticket : Codes
  useful : Codes

/-- A committed transcript: unit `u`'s commitment, or a blank, which is never correct. -/
abbrev Transcript : Type := ℕ → Option Commit

/-- An abstract cost model for the online phase (the named instance: W1 at the measured H100 prices, one unit = one dense
FP8 multiply-accumulate). The model fixes `cost` and `calls`; an adversary only chooses a program. -/
structure CostModel (Q R S : Type) where
  Prog : Type
  run : Prog → (Q → R) → Workload → (ℕ → Codes) → State → S → Transcript
  cost : Prog → (Q → R) → Workload → (ℕ → Codes) → State → S → ℝ
  calls : Prog → (Q → R) → Workload → (ℕ → Codes) → State → S → ℕ

/-- The model's programs reach `H` only through counted queries. -/
def CostModel.Queries {Q R S : Type} (CM : CostModel Q R S) : Prop :=
  ∀ (prog : CM.Prog) (U : Workload) (act : ℕ → Codes) (σ : State) (s : S), ∃ t : QTree Q R Transcript,
    ∀ H : Q → R, CM.run prog H U act σ s = t.eval H ∧ CM.calls prog H U act σ s = t.calls H

/-- An output-ticket protocol: the FP32 values of its reference ticket and useful words (`none` for a non-finite word),
its domain rules on committed activations and on each weight row, and its costs.
- `weightOK sh w j`: row `j` of the weight `w` of a unit of shape `sh` passes the rules. A correct unit and a good tile
  have every weight row they read passing (`Correct`, `TileGood`); a tile's check asks it of the tile's columns, on the
  weight rows it opens (`TileCheck.Accepts`).
- `credit H s U u act`: what the named assumption credits for unit `u`. The reference computes it from the committed
  activations `act`, the weights and the realized noise `(H, s)`, by the same replay that checks the unit; it never reads
  the transcript's claimed ticket or useful words.
- `capOK H s U u act`: the unit's replayed debit is within the cap. It is part of `Correct`: a unit above the cap is
  rejected, so it is in neither TT_OUT's credit sum nor `Gγ`'s win sum. `Correct` is the game's idealized check of every
  unit; a verifier that checks drawn tiles decides the cap per tile (`Tile.lean`: `TileGood`, `GγSampled`).
- `Wref`: the honest reference's cost of the unit, read from the layout (excluding hash calls; the model's own clean
  quantizer is outside the unit, so neither program is charged for it). -/
structure Protocol (Q R S : Type) where
  ticket : (Q → R) → S → Workload → ℕ → Codes → ℕ → ℕ → Option ℚ
  useful : (Q → R) → S → Workload → ℕ → Codes → ℕ → ℕ → Option ℚ
  actOK : Shape → Codes → Prop
  weightOK : Shape → Codes → ℕ → Prop
  credit : (Q → R) → S → Workload → ℕ → Codes → ℝ
  capOK : (Q → R) → S → Workload → ℕ → Codes → Prop
  Wref : Layout → ℕ → ℝ

variable {Q R S : Type}

/-- A workload is in the domain `D`: at most `2^64` units, and `D` holds of its layout. Its weights may fail the rules:
a unit or tile that reads a failing weight row is neither correct nor good, so it earns nothing. -/
def Workload.InDomain (_P : Protocol Q R S) (D : Layout → Prop) (U : Workload) : Prop :=
  U.layout.N ≤ 2 ^ 64 ∧ D U.layout

/-- Every row of the workload's weights passes the rules: the honest workload's admission, which completeness and
`AdmitsRef` take beside admissible activations. -/
def Workload.WeightsOK (P : Protocol Q R S) (U : Workload) : Prop :=
  ∀ u < U.layout.N, ∀ j < (U.layout.shape u).n, P.weightOK (U.layout.shape u) (U.weight (U.layout.wid u)) j

/-- A correct unit: admissible activations, its replayed debit within the cap, every row of its weight passing the
rules, and every ticket and useful word on its `m × n` entries has the reference FP32 value (bit for bit up to the sign
of zero). -/
def Correct (P : Protocol Q R S) (U : Workload) (H : Q → R) (s : S) (u : ℕ) (c : Commit) : Prop :=
  P.actOK (U.layout.shape u) c.act ∧ P.capOK H s U u c.act ∧
    (∀ j < (U.layout.shape u).n, P.weightOK (U.layout.shape u) (U.weight (U.layout.wid u)) j) ∧
    ∀ i < (U.layout.shape u).m, ∀ j < (U.layout.shape u).n,
      wordQ (c.ticket i j) = P.ticket H s U u c.act i j ∧ wordQ (c.useful i j) = P.useful H s U u c.act i j

open Classical in
/-- `K`: the correct units of `τ`. -/
noncomputable def correctSet (P : Protocol Q R S) (U : Workload) (H : Q → R) (s : S) (τ : Transcript) : Finset ℕ :=
  (range U.layout.N).filter fun u => ∃ c, τ u = some c ∧ Correct P U H s u c

/-- The activations unit `u` committed in `τ` (zero for a blank, which is never correct). -/
def actOf (τ : Transcript) (u : ℕ) : Codes :=
  match τ u with
  | some c => c.act
  | none => fun _ _ => 0

open Classical in
/-- An FP32 word with value `v` (some word, when one exists; `0` otherwise). -/
noncomputable def wordOf (v : Option ℚ) : ℕ := if h : ∃ w, wordQ w = v then Classical.choose h else 0

/-- The honest transcript on activations `act`. -/
noncomputable def honestTranscript (P : Protocol Q R S) (H : Q → R) (s : S) (U : Workload) (act : ℕ → Codes) :
    Transcript :=
  fun u => if u < U.layout.N then
    some ⟨act u, fun i j => wordOf (P.ticket H s U u (act u) i j), fun i j => wordOf (P.useful H s U u (act u) i j)⟩
  else none

/-- The cost model runs the honest reference: one program outputs the honest transcript on every in-domain workload
whose weights pass the rules and admissible activations, with no preprocessing, at cost at most `∑_u W_ref(u)` and at
most `qref U` calls. -/
def AdmitsRef (CM : CostModel Q R S) (P : Protocol Q R S) (D : Layout → Prop) (qref : Workload → ℕ) : Prop :=
  ∃ prog : CM.Prog, ∀ U : Workload, U.InDomain P D → U.WeightsOK P → ∀ act : ℕ → Codes,
    (∀ u < U.layout.N, P.actOK (U.layout.shape u) (act u)) → ∀ (H : Q → R) (s : S),
      CM.run prog H U act [] s = honestTranscript P H s U act ∧
      CM.cost prog H U act [] s ≤ ∑ u ∈ range U.layout.N, P.Wref U.layout u ∧
      CM.calls prog H U act [] s ≤ qref U

/-- An adversary, chosen after the workload: a preprocessing oracle program, an online program, its activation input. -/
structure Adversary (CM : CostModel Q R S) where
  pre : QTree Q R State
  online : CM.Prog
  act : ℕ → Codes

/-- The transcript the adversary commits on oracle `H` and salt `s`. -/
def Adversary.transcript {CM : CostModel Q R S} (A : Adversary CM) (U : Workload) (H : Q → R) (s : S) : Transcript :=
  CM.run A.online H U A.act (A.pre.eval H) s

/-- The online phase costs at most `T`, and all oracle calls number at most `q`, for every oracle and salt. -/
def Adversary.Bounded {CM : CostModel Q R S} (A : Adversary CM) (U : Workload) (T : ℝ) (q : ℕ) : Prop :=
  ∀ (H : Q → R) (s : S), CM.cost A.online H U A.act (A.pre.eval H) s ≤ T ∧
    A.pre.calls H + CM.calls A.online H U A.act (A.pre.eval H) s ≤ q

/-- The adversary wins: the honest work of its correct units exceeds `T / (1 − γ)`. -/
def Wins (P : Protocol Q R S) (U : Workload) (γ T : ℝ) (τ : Transcript) (H : Q → R) (s : S) : Prop :=
  T / (1 - γ) < ∑ u ∈ correctSet P U H s τ, P.Wref U.layout u

/-- **The γ-gap property** `G_γ` with error `η(q, N)`. -/
def Gγ [Fintype Q] [DecidableEq Q] [Fintype R] [Fintype S] (CM : CostModel Q R S) (P : Protocol Q R S)
    (D : Layout → Prop) (γ : ℝ) (η : ℕ → ℕ → ℝ) : Prop :=
  ∀ U : Workload, U.InDomain P D → ∀ T : ℝ, 0 ≤ T → ∀ q : ℕ, q ≤ 2 ^ 64 → ∀ A : Adversary CM, A.Bounded U T q →
    pr (fun x : (Q → R) × S => Wins P U γ T (A.transcript U x.1 x.2) x.1 x.2) ≤ η q U.layout.N

/-! ## Pearl-C's accounting and domain (H100 units) -/

/-- The forming costs per activation element (H100 units): `fs` depends on the salt (credited), `qa` on the activations
alone (not credited: an adversary that chose its activations can skip it). The unit's input is the model's clean
quantizer output; whatever that quantizer computes is outside the unit for both programs, and every operation inside the
unit is charged at its price to every program. -/
structure Costs where
  fs : ℕ
  qa : ℕ

/-- §5 with Pearl's headroom scale `α = 448/(s + 4δρ)` adopted as the model's quantizer, even with PoUW off: the
model's FP8 path uses exactly this α (the kernel's pins for s, ρ's order, α's rounding and `mul.rn(α, x)`) whether
or not PoUW runs; otherwise `Costs.current` applies. The input is `α·x`; the noise atom (32) and the salt-dependent
cast `A′ = e4m3(D)` (32) are credited; the sampled statistic ρ (4) is A-only. -/
def Costs.adopted : Costs := ⟨64, 4⟩

/-- As `adopted`, with the model's quantizer also emitting ρ. -/
def Costs.adoptedFull : Costs := ⟨64, 0⟩

/-- Under the model's current plain max scale: `α·x` (an FP32 multiply, 32) and ρ (4) are A-only protocol work. -/
def Costs.current : Costs := ⟨64, 36⟩

/-- The credited work of one unit: the ticket chain `m·k·n`, its promotion adds `m·k·n/G`, the fused `A′·F_B` (`m·k·r`),
the peel combine (BF16, depth `2r`, 2 units per MAC: `4·r·m·n`), and the forming that depends on the salt (`fs·m·k`). The
per-job share of `B̃·F_A` would enter credit and reference alike, and leaving it out only raises γ. -/
def creditOf (p : Params) (c : Costs) (s : Shape) : ℚ :=
  let mkn : ℚ := (s.m : ℚ) * s.k * s.n
  mkn + (if p.G = 0 then 0 else mkn / p.G) + (s.m : ℚ) * s.k * p.r + 4 * (p.r : ℚ) * s.m * s.n + (c.fs : ℚ) * s.m * s.k

/-- A unit's replayed debit: `atoms`, its flagged atom-words (P1's
`jointFlags` over every output word, on the noisy codes under the ticket's promotion schedule: every atom of a one-group
joint skip that leaves the word unchanged); `elems`, its forming elements that are salt-dead or needed by no unflagged
atom (P3). Cross-group joint skips are in neither: no replay-local count bounds them (`joint_cross_group`), and the named
assumption carries them. -/
structure Debit where
  atoms : ℕ
  elems : ℕ

/-- The debit in H100 units: per flagged atom-word its 32 MACs and its share `32/G` of the promotion adds, and `fs` per
debited element. -/
def debitOf (p : Params) (c : Costs) (d : Debit) : ℚ :=
  (32 + (if p.G = 0 then 0 else 32 / p.G)) * d.atoms + (c.fs : ℚ) * d.elems

/-- The honest reference's work: the credited work plus the forming that depends on the activations alone. Neither this
nor `creditOf` has a weight-side term (`B̃·F_A`, `P_B`, the forming of B̃), whether that work runs per call or once per
weight and epoch, so a verifier's `W_ref` and its draw's tickets must use exactly `wrefOf`. -/
def wrefOf (p : Params) (c : Costs) (s : Shape) : ℚ := creditOf p c s + (c.qa : ℚ) * s.m * s.k

/-- The credited work when forming is not credited at all. -/
def creditChainOnly (p : Params) (c : Costs) (s : Shape) : ℚ := creditOf p c s - (c.fs : ℚ) * s.m * s.k

/-- The reduction's gap: `γ = 1 − (1 − γ₀)/Ω`. -/
def gammaOf (γ₀ Ω : ℚ) : ℚ := 1 - (1 - γ₀) / Ω

/-- `Π`'s costs are Pearl-C's accounting at `(p, c)` with cap `ρ`: `W_ref` is `wrefOf`, and a unit within the cap is
credited at least `(1 − ρ)·creditOf`. -/
def PearlCAccounting (P : Protocol Q R S) (p : Params) (c : Costs) (ρ : ℚ) : Prop :=
  (∀ (L : Layout) (u : ℕ), P.Wref L u = (wrefOf p c (L.shape u) : ℝ)) ∧
    ∀ (H : Q → R) (s : S) (U : Workload) (u : ℕ) (act : Codes), P.capOK H s U u act →
      (((1 - ρ) * creditOf p c (U.layout.shape u) : ℚ) : ℝ) ≤ P.credit H s U u act

/-- Pearl-C's shape domain: positive dimensions, a whole number of promotion groups, `k ≤ 2^16`, and promotion every
4 atoms, which confines freezing to one group (P1). What freezing remains is debited per unit, and the cap on the debit is
in `Correct`, since it depends on the activations and the noise (the aligned-spike census). The
noise law, the seeds and the chain live in the protocol's reference functions.

**The channel split** is part of the model's own FP8 path, adopted as `Costs.adopted` adopts Pearl's α.
- **The set.** Each linear layer ℓ has a fixed set `S_ℓ` of input channels, at most 1% of its k. They are computed by the
  model's BF16 side GEMM with PoUW on and off, so they are the model's work (glue), outside the unit and outside `W_ref`.
- **The unit's k channels** are the kept channels, padded with zero channels to a whole number of promotion groups.
- **How `S_ℓ` is chosen.** Once per model, before any salt or request, by one pinned implementation of the greedy rule:
  `split_channels` in `benchmarks/pouw/pearlc_headline.py` (at `f6f05bd8`), with its batch of 4 channels per
  step, its tie-break (equal offence counts go to the lower channel index) and its budget of `⌊k/100⌋` channels, which S
  never exceeds. It moves the input channels that offend the most failing rows until every row of the calibration
  activations and of the weights passes Π's rules, or the budget is spent. The input is the committed calibration data: a
  fixed token set named by its hash in the model's commitment, with the weights. Another implementation of the same
  greedy idea is a different rule: the plan's two gave 38 and 35 channels on the same data.
- **Why it is fixed.** `S_ℓ` is committed with the weights before any salt, so it is the same for every request, and with
  the one pinned implementation it is the same for every prover. A prover can't widen or move it per request, so it can't
  steer which channels the unit leaves out.
- **Rows that still fail.** An activation row that fails Π's rules with the fixed `S_ℓ` (a token-specific massive
  activation, which no fixed set covers: Qwen2.5-7B's early `down_proj` layers) stays in its unit: its words are computed
  and bound, it earns no credit, and the verifier checks that it fails. Excluding a
  row only lowers the prover's credit, so γ needs no check that the exclusion was forced. The per-tile statement
  (`Tile.lean`) credits a tile's passing rows; this unit-level game, whose `actOK` reads the whole unit, credits nothing of
  a unit with such a row. -/
def pearlCDomain (p : Params) (L : Layout) : Prop :=
  ∀ u < L.N, 0 < (L.shape u).m ∧ 0 < (L.shape u).n ∧ 0 < (L.shape u).k ∧ Shape.Tiled p (L.shape u) ∧
    (L.shape u).k ≤ 2 ^ 16 ∧ p.G = 4

/-- The domain restricted to one shape. -/
def pearlCDomainAt (p : Params) (s : Shape) (L : Layout) : Prop := pearlCDomain p L ∧ ∀ u < L.N, L.shape u = s

/-- The headline shapes (`m n k`). -/
def sh8192 : Shape := ⟨8192, 8192, 8192⟩

def sh16384 : Shape := ⟨16384, 16384, 16384⟩

end Pouw.PearlC
