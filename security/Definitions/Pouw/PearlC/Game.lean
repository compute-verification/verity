module

public import Definitions.Pouw.PearlC.Defs
public import Definitions.Pouw.Basic.Prob
public import Definitions.Pouw.Basic.QTree

/-! Pearl-C's gap game, whose bound words are its ticket and useful-output words checked bit for bit, with Pearl-C's accounting and domain. -/

@[expose] public section

namespace Pouw.PearlC

open Finset Pouw.Fp8Atom

/-- The declared layout: `N` units `0, …, N − 1` (unit `u`'s index is `u`), their shapes and the id of each unit's
fixed operand. -/
structure Layout where
  N : ℕ
  shape : ℕ → Shape
  fixedId : ℕ → ℕ

/-- A matrix of bit patterns: operand codes, or output FP32 words. -/
abbrev Codes : Type := ℕ → ℕ → ℕ

/-- A declared workload: its layout and its fixed operands' codes. -/
structure Workload where
  layout : Layout
  /-- Declared before the salt and reusable across units. -/
  fixed : ℕ → Codes

/-- The preprocessed state: any finite data. -/
abbrev State : Type := List ℤ

/-- One unit's commitment (ideal: the values themselves): activations, ticket words, useful output words. -/
structure Commit where
  act : Codes
  ticket : Codes
  useful : Codes

/-- A committed trace: unit `u`'s commitment, or a blank, which is never correct. -/
abbrev Trace : Type := ℕ → Option Commit

/-- An abstract cost model for the online phase (the named instance: W1 at the measured H100 prices, one unit = one dense
FP8 multiply-accumulate). The model fixes `cost` and `calls`; an adversary only chooses a program. -/
structure CostModel (Q R S : Type) where
  Prog : Type
  run : Prog → (Q → R) → Workload → (ℕ → Codes) → State → S → Trace
  cost : Prog → (Q → R) → Workload → (ℕ → Codes) → State → S → ℝ
  calls : Prog → (Q → R) → Workload → (ℕ → Codes) → State → S → ℕ

/-- The model's programs reach `H` only through counted queries. -/
def CostModel.Queries {Q R S : Type} (CM : CostModel Q R S) : Prop :=
  ∀ (prog : CM.Prog) (U : Workload) (act : ℕ → Codes) (σ : State) (s : S), ∃ t : QTree Q R Trace,
    ∀ H : Q → R, CM.run prog H U act σ s = t.eval H ∧ CM.calls prog H U act σ s = t.calls H

/-- An output-ticket protocol: the FP32 values of its reference ticket and useful words (`none` for a non-finite word),
its domain rules on committed activations and on each fixed-operand row, and its costs.
- `fixedOK sh w j`: row `j` of the fixed operand `w` of a unit of shape `sh` passes the rules. A correct unit and a good
  tile have every fixed-operand row they read passing (`Correct`, `TileGood`); a tile's check asks it of the tile's
  columns, on the fixed-operand rows it opens (`TileCheck.Accepts`).
- `credit H s U u act`: what the named assumption credits for unit `u`. The reference computes it from the committed
  activations `act`, the fixed operands and the realized noise `(H, s)`, by the same replay that checks the unit; it never reads
  the trace's claimed ticket or useful words.
- `inertBoundOK H s U u act`: the unit's replayed debit is within the cap. It is part of `Correct`: a unit above the cap is
  rejected, so it is in neither TT_OUT's credit sum nor `GGap`'s win sum. `Correct` is the game's idealized check of every
  unit; a verifier that checks sampled tiles decides the cap per tile (`Tile.lean`: `TileGood`, `GGapSampled`).
- `Wref`: the honest reference's cost of the unit, read from the layout (excluding hash calls; the model's own clean
  quantizer is outside the unit, so neither program is charged for it). -/
structure Protocol (Q R S : Type) where
  ticket : (Q → R) → S → Workload → ℕ → Codes → ℕ → ℕ → Option ℚ
  useful : (Q → R) → S → Workload → ℕ → Codes → ℕ → ℕ → Option ℚ
  actOK : Shape → Codes → Prop
  fixedOK : Shape → Codes → ℕ → Prop
  credit : (Q → R) → S → Workload → ℕ → Codes → ℝ
  inertBoundOK : (Q → R) → S → Workload → ℕ → Codes → Prop
  Wref : Layout → ℕ → ℝ

variable {Q R S : Type}

/-- A workload is in the domain `D`: at most `2^64` units, and `D` holds of its layout. Its fixed operands may fail the
rules: a unit or tile that reads a failing fixed-operand row is neither correct nor good, so it earns nothing. -/
def Workload.InDomain (_P : Protocol Q R S) (D : Layout → Prop) (U : Workload) : Prop :=
  U.layout.N ≤ 2 ^ 64 ∧ D U.layout

def Workload.FixedOK (P : Protocol Q R S) (U : Workload) : Prop :=
  ∀ u < U.layout.N, ∀ j < (U.layout.shape u).n, P.fixedOK (U.layout.shape u) (U.fixed (U.layout.fixedId u)) j

/-- A correct unit: admissible activations, its replayed debit within the cap, every row of its fixed operand passing
the rules, and every ticket and useful word on its `m × n` entries has the reference FP32 value (bit for bit up to the sign
of zero). -/
def Correct (P : Protocol Q R S) (U : Workload) (H : Q → R) (s : S) (u : ℕ) (c : Commit) : Prop :=
  P.actOK (U.layout.shape u) c.act ∧ P.inertBoundOK H s U u c.act ∧
    (∀ j < (U.layout.shape u).n, P.fixedOK (U.layout.shape u) (U.fixed (U.layout.fixedId u)) j) ∧
    ∀ i < (U.layout.shape u).m, ∀ j < (U.layout.shape u).n,
      wordQ (c.ticket i j) = P.ticket H s U u c.act i j ∧ wordQ (c.useful i j) = P.useful H s U u c.act i j

open Classical in
/-- `K`: the correct units of `τ`. -/
noncomputable def correctSet (P : Protocol Q R S) (U : Workload) (H : Q → R) (s : S) (τ : Trace) : Finset ℕ :=
  (range U.layout.N).filter fun u => ∃ c, τ u = some c ∧ Correct P U H s u c

/-- The activations unit `u` committed in `τ` (zero for a blank, which is never correct). -/
def actOf (τ : Trace) (u : ℕ) : Codes :=
  match τ u with
  | some c => c.act
  | none => fun _ _ => 0

open Classical in
/-- Some word with value `v` when one exists, `0` otherwise. -/
noncomputable def wordOf (v : Option ℚ) : ℕ := if h : ∃ w, wordQ w = v then Classical.choose h else 0

noncomputable def honestTrace (P : Protocol Q R S) (H : Q → R) (s : S) (U : Workload) (act : ℕ → Codes) :
    Trace :=
  fun u => if u < U.layout.N then
    some ⟨act u, fun i j => wordOf (P.ticket H s U u (act u) i j), fun i j => wordOf (P.useful H s U u (act u) i j)⟩
  else none

def AdmitsRef (CM : CostModel Q R S) (P : Protocol Q R S) (D : Layout → Prop) (qref : Workload → ℕ) : Prop :=
  ∃ prog : CM.Prog, ∀ U : Workload, U.InDomain P D → U.FixedOK P → ∀ act : ℕ → Codes,
    (∀ u < U.layout.N, P.actOK (U.layout.shape u) (act u)) → ∀ (H : Q → R) (s : S),
      CM.run prog H U act [] s = honestTrace P H s U act ∧
      CM.cost prog H U act [] s ≤ ∑ u ∈ range U.layout.N, P.Wref U.layout u ∧
      CM.calls prog H U act [] s ≤ qref U

/-- An adversary, chosen after the workload: a preprocessing oracle program, an online program, its activation input. -/
structure Adversary (CM : CostModel Q R S) where
  pre : QTree Q R State
  online : CM.Prog
  act : ℕ → Codes

/-- The trace the adversary commits on oracle `H` and salt `s`. -/
def Adversary.trace {CM : CostModel Q R S} (A : Adversary CM) (U : Workload) (H : Q → R) (s : S) : Trace :=
  CM.run A.online H U A.act (A.pre.eval H) s

/-- The online phase costs at most `T`, and all oracle calls number at most `q`, for every oracle and salt. -/
def Adversary.Bounded {CM : CostModel Q R S} (A : Adversary CM) (U : Workload) (T : ℝ) (q : ℕ) : Prop :=
  ∀ (H : Q → R) (s : S), CM.cost A.online H U A.act (A.pre.eval H) s ≤ T ∧
    A.pre.calls H + CM.calls A.online H U A.act (A.pre.eval H) s ≤ q

/-- The adversary wins: the honest work of its correct units exceeds `T / (1 − g)`. -/
def Wins (P : Protocol Q R S) (U : Workload) (g T : ℝ) (τ : Trace) (H : Q → R) (s : S) : Prop :=
  T / (1 - g) < ∑ u ∈ correctSet P U H s τ, P.Wref U.layout u

/-- **The gap property** `G_g` at the gap `g` (an adversary's advantage `ε = g/(1 − g)`) with failure probability
`δ(q, N)`. -/
def GGap [Fintype Q] [DecidableEq Q] [Fintype R] [Fintype S] (CM : CostModel Q R S) (P : Protocol Q R S)
    (D : Layout → Prop) (g : ℝ) (δ : ℕ → ℕ → ℝ) : Prop :=
  ∀ U : Workload, U.InDomain P D → ∀ T : ℝ, 0 ≤ T → ∀ q : ℕ, q ≤ 2 ^ 64 → ∀ A : Adversary CM, A.Bounded U T q →
    pr (fun x : (Q → R) × S => Wins P U g T (A.trace U x.1 x.2) x.1 x.2) ≤ δ q U.layout.N

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

def Costs.adoptedFull : Costs := ⟨64, 0⟩

/-- Under the model's current plain max scale: `α·x` (an FP32 multiply, 32) and ρ (4) are A-only protocol work. -/
def Costs.current : Costs := ⟨64, 36⟩

/-- The credited work of one unit: the ticket chain `m·k·n`, its promotion adds `m·k·n/G`, the fused `A′·F_B` (`m·k·r`),
the peel combine (BF16, depth `2r`, 2 units per MAC: `4·r·m·n`), and the forming that depends on the salt (`fs·m·k`). The
per-job share of `B̃·F_A` would enter credit and reference alike, and leaving it out only raises g. -/
def creditOf (p : Params) (c : Costs) (s : Shape) : ℚ :=
  let mkn : ℚ := (s.m : ℚ) * s.k * s.n
  mkn + (if p.G = 0 then 0 else mkn / p.G) + (s.m : ℚ) * s.k * p.r + 4 * (p.r : ℚ) * s.m * s.n + (c.fs : ℚ) * s.m * s.k

/-- A unit's replayed debit: `atoms`, its flagged atom-words (P1's
`jointFlags` over every output word, on the noisy codes under the ticket's promotion schedule: every atom of a one-group
joint skip that leaves the word unchanged); `elems`, its forming elements that are salt-dead or needed by no unflagged
atom (P3). Cross-group joint skips are in neither: no replay-local count bounds them (`joint_cross_group`), and the named
assumption carries them. -/
structure InertWork where
  atoms : ℕ
  elems : ℕ

/-- The debit in H100 units: per flagged atom-word its 32 MACs and its share `32/G` of the promotion adds, and `fs` per
debited element. -/
def inertWorkOf (p : Params) (c : Costs) (d : InertWork) : ℚ :=
  (32 + (if p.G = 0 then 0 else 32 / p.G)) * d.atoms + (c.fs : ℚ) * d.elems

/-- The honest reference's work: the credited work plus the forming that depends on the activations alone. Neither this
nor `creditOf` has a weight-side term (`B̃·F_A`, `P_B`, the forming of B̃), whether that work runs per call or once per
weight and epoch, so a verifier's `W_ref` and its sample's tickets must use exactly `wrefOf`. -/
def wrefOf (p : Params) (c : Costs) (s : Shape) : ℚ := creditOf p c s + (c.qa : ℚ) * s.m * s.k

def creditChainOnly (p : Params) (c : Costs) (s : Shape) : ℚ := creditOf p c s - (c.fs : ℚ) * s.m * s.k

def gapOf (g₀ Ω : ℚ) : ℚ := 1 - (1 - g₀) / Ω

def PearlCAccounting (P : Protocol Q R S) (p : Params) (c : Costs) (ρ : ℚ) : Prop :=
  (∀ (L : Layout) (u : ℕ), P.Wref L u = (wrefOf p c (L.shape u) : ℝ)) ∧
    ∀ (H : Q → R) (s : S) (U : Workload) (u : ℕ) (act : Codes), P.inertBoundOK H s U u act →
      (((1 - ρ) * creditOf p c (U.layout.shape u) : ℚ) : ℝ) ≤ P.credit H s U u act

/-- Promotion every 4 atoms confines freezing to one group (P1), and the debit's cap is in `Correct`. The channel split is
committed with the weights before any salt by one pinned rule, so a prover can't steer which channels a unit leaves
out; a unit with a row that still fails Π's rules earns no credit here. -/
def pearlCDomain (p : Params) (L : Layout) : Prop :=
  ∀ u < L.N, 0 < (L.shape u).m ∧ 0 < (L.shape u).n ∧ 0 < (L.shape u).k ∧ Shape.Tiled p (L.shape u) ∧
    (L.shape u).k ≤ 2 ^ 16 ∧ p.G = 4

def pearlCDomainAt (p : Params) (s : Shape) (L : Layout) : Prop := pearlCDomain p L ∧ ∀ u < L.N, L.shape u = s

/-- The headline shapes (`m n k`). -/
def sh8192 : Shape := ⟨8192, 8192, 8192⟩

def sh16384 : Shape := ⟨16384, 16384, 16384⟩

end Pouw.PearlC
