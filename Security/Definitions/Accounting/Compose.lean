import Definitions.Accounting.TS
import Definitions.Accounting.HMu

/-!
# PoUW as a compile pass over an arbitrary declared circuit (trusted)

The vocabulary of `Models.Accounting.Guarantees.Compose` (note:20261018T0346Z-draft-theorem-compose). The compile pass
`C → C′` replaces each credited subcircuit of a declared circuit by its kind's template, and each subcircuit's noise is
drawn after its operands are committed (decisions 101 and 102: one seed a subcircuit). A round of the window is one run
of `C′`: one stage a subcircuit, in the circuit's order, then `s` draws of whole units, each proved with the generation
of its own noise (`Accounting.TS.StageWindow`).

* `Scheme κ K`: per credited kind `κ` (a matmul at its shape, or any kind with a template), one instance's floor `♭_κ`
  over the classes `K`, its proof units (whole dot products), each unit's committed words, the fresh noise bits a
  skipped value of the kind carries and its receipt's fibre `Z`.
* `Declared κ`: the declared circuit, `N` subcircuits in order, each credited at a kind or not (`none`). Nothing else
  about the circuit is read: an uncredited subcircuit is a stage with no unit.
* `UnitOf`, `TicketOf`: the circuit's proof units, and its committed words (one ticket a word, so a uniform ticket is a
  work-weighted draw). `stageOf u` counts the stages after `u`'s, as `CoinStages` does; `workWeight u` is `u`'s words
  over all of them.
* `floorOf`: the circuit's floor, the sum of its credited subcircuits' floors, so its credit `creditOf` is
  `B(Σ_j ♭_j)/f_max` on the declared machine (decision 100), not `Σ_j B(♭_j)/f_max`.
* `epsFlat`, `epsMu`, `epsLambda`, `epsOf`: NOTATION's `ε_♭ = ε_skip + ε_cert + ε_route`, `ε_μ = ε_mand` and
  `ε_Λ = ε_jl` at the circuit's floor, and their sum.
* `Fits`, `Scope`, `PlayOf`: the model is the circuit's on the data sheet (`N_w` its committed words); the parameters'
  ranges; the play's ideal game is the stage window's at work-weighted draws.
* `luckAt b Z = Z·C(b, ⌊b/2⌋)/2^b`, the receipt luck of `b` fresh bits at fibre `Z` (`HMu.ReceiptLuckOfCoins`), and
  `luckOf u` a unit's.
* `windowLuck p stg u W`: in every round, at `u`'s stage, whatever the past, `u` skipped passes with probability at
  most `p`; `TemplateLuck` is that at every unit's own `luckOf`.
* `ReadsOffNoise skip ok u b Z`: a stage's coin splits into `b` fresh bits and the rest, and `u`'s skip decision and its
  committed receipt are a `HMu.CoinRound` on the rest whose skipped value is the bits' sum; `windowNoise` is that at
  `u`'s stage in every round.
-/

namespace Accounting.Compose

open Accounting.TS Accounting.HMu Pouw.RelativeSecurity

/-- **A scheme's templates**, per credited kind `κ`: one instance's floor `♭_κ`, its proof units, unit `i`'s committed
words, the fresh noise bits a skipped value carries and the receipt's fibre `Z`. -/
structure Scheme (κ K : Type) where
  floor : κ → Labelled K
  units : κ → ℕ
  words : κ → ℕ → ℕ
  bits : κ → ℕ
  fibre : κ → ℕ

/-- **A declared circuit**: `N` subcircuits in order, each credited at a kind (`some k`) or not (`none`). -/
structure Declared (κ : Type) where
  N : ℕ
  kind : Fin N → Option κ

/-- The receipt luck of `b` fresh bits at fibre `Z`: `Z·C(b, ⌊b/2⌋)/2^b`. -/
noncomputable def luckAt (b Z : ℕ) : ℝ := Z * ((b.choose (b / 2) : ℝ) / 2 ^ b)

/-- A kind's receipt luck. -/
noncomputable def Scheme.luck {κ K : Type} (Sc : Scheme κ K) (k : κ) : ℝ := luckAt (Sc.bits k) (Sc.fibre k)

section Circuit

variable {κ K : Type} (Sc : Scheme κ K) (Cd : Declared κ)

/-- Subcircuit `j`'s proof units: its kind's, or none when it isn't credited. -/
def unitsAt (j : Fin Cd.N) : ℕ := (Cd.kind j).elim 0 Sc.units

/-- The circuit's proof units: unit `i` of subcircuit `j`. -/
abbrev UnitOf : Type := Σ j : Fin Cd.N, Fin (unitsAt Sc Cd j)

/-- A unit's committed words. -/
def wordsOf (u : UnitOf Sc Cd) : ℕ := (Cd.kind u.1).elim 0 fun k => Sc.words k u.2.val

/-- A unit's fresh noise bits. -/
def bitsOf (u : UnitOf Sc Cd) : ℕ := (Cd.kind u.1).elim 0 Sc.bits

/-- A unit's receipt fibre. -/
def fibreOf (u : UnitOf Sc Cd) : ℕ := (Cd.kind u.1).elim 0 Sc.fibre

/-- A unit's receipt luck. -/
noncomputable def luckOf (u : UnitOf Sc Cd) : ℝ := luckAt (bitsOf Sc Cd u) (fibreOf Sc Cd u)

/-- The circuit's committed words, one ticket each. -/
abbrev TicketOf : Type := Σ u : UnitOf Sc Cd, Fin (wordsOf Sc Cd u)

/-- The unit a ticket names. -/
def ticketUnit (t : TicketOf Sc Cd) : UnitOf Sc Cd := t.1

/-- A unit's stage, counted as `CoinStages` does: the stages after its subcircuit's. -/
def stageOf (u : UnitOf Sc Cd) : ℕ := Cd.N - 1 - u.1.val

/-- A unit's work weight: its committed words over the circuit's. -/
noncomputable def workWeight (u : UnitOf Sc Cd) : ℝ := (wordsOf Sc Cd u : ℝ) / Fintype.card (TicketOf Sc Cd)

/-- Subcircuit `j`'s floor: its kind's, or zero when it isn't credited. -/
def floorAt (j : Fin Cd.N) : Labelled K := (Cd.kind j).elim (⟨0, 0, 0, 0⟩ : Labelled K) Sc.floor

/-- **The circuit's floor** `♭ = Σ_j ♭_j`, part by part. -/
def floorOf : Labelled K where
  nw k := ∑ j, (floorAt Sc Cd j).nw k
  nf k := ∑ j, (floorAt Sc Cd j).nf k
  nx k := ∑ j, (floorAt Sc Cd j).nx k
  ns k := ∑ j, (floorAt Sc Cd j).ns k

/-- **The play is the compiled circuit's**: its ideal game is the stage window's, one stage a subcircuit, at
work-weighted draws (a uniform ticket names the unit holding its word). -/
def PlayOf {C : Type} [Fintype C] {s S : ℕ} {T L U : Type}
    (Pl : Play (Paths C Cd.N × (Fin s → TicketOf Sc Cd)) S K T L U)
    (W : StageWindow C (UnitOf Sc Cd) (TicketOf Sc Cd) Cd.N s S) : Prop :=
  Pl.game = W.game (stageOf Sc Cd) (ticketUnit Sc Cd) (workWeight Sc Cd)

variable {R T L U F X : Type} [Fintype K] [Fintype R] [Fintype T] [Fintype L] [Fintype U] [Fintype F] [Fintype X]

/-- **The credit**: `T_cred = B(♭)/f_max`, the circuit's floor on the model's machine. -/
noncomputable def creditOf (Z : Model K R T L U F X) : ℝ := Z.credit (floorOf Sc Cd)

/-- `ε_♭ = ε_skip + ε_cert + ε_route` at the circuit's floor. -/
noncomputable def epsFlat (Z : Model K R T L U F X) (lam : R → ℝ) : ℝ :=
  Z.epsSkip (floorOf Sc Cd) lam + Z.epsCert (floorOf Sc Cd) lam + Z.epsRoute (floorOf Sc Cd) lam

/-- `ε_μ = ε_mand` at the circuit's floor. -/
noncomputable def epsMu (Z : Model K R T L U F X) (lam : R → ℝ) : ℝ := Z.epsMand (floorOf Sc Cd) lam

/-- `ε_Λ = ε_jl` at the circuit's floor, at luck `p`, confidence `δs` and `S` rounds. -/
noncomputable def epsLambda (Z : Model K R T L U F X) (lam : R → ℝ) (p δs : ℝ) (S : ℕ) : ℝ :=
  Z.epsJl (floorOf Sc Cd) lam p δs S

/-- **The circuit's `ε`**: `ε_♭ + ε_μ + ε_Λ`. -/
noncomputable def epsOf (Z : Model K R T L U F X) (lam : R → ℝ) (p δs : ℝ) (S : ℕ) : ℝ :=
  epsFlat Sc Cd Z lam + epsMu Sc Cd Z lam + epsLambda Sc Cd Z lam p δs S

/-- **The model is the circuit's on sheet `Sh`**: its machine is the sheet's, and its committed words are the credited
units' (`N_w = |TicketOf|`). -/
def Fits {P : Type} [DecidableEq P] (Sh : Sheet K P) (Z : Model K (SheetRow P) T L U F X) : Prop :=
  Z.G = Sh.machine ∧ Z.Nw = (Fintype.card (TicketOf Sc Cd) : ℝ)

/-- **The parameters' ranges**: `π` a certificate, the model well formed at the circuit's floor, `S > 0` rounds of
`s ≥ 2` draws, `p ∈ [1/e, 1)` at least every unit's receipt luck, `δs > 0`, and SCALAR-ROUTE's floors at the ½ the
data sheet gives (`q, q_T ≤ ½`, `b_T ≤ 0`). -/
def Scope (Z : Model K R T L U F X) (lam : R → ℝ) (S s : ℕ) (p δs : ℝ) : Prop :=
  IsCert lam ∧ Z.Wf (floorOf Sc Cd) ∧ 0 < S ∧ 2 ≤ s ∧ Real.exp (-1) ≤ p ∧ p < 1 ∧ 0 < δs ∧
    (∀ u, luckOf Sc Cd u ≤ p) ∧ Z.q ≤ 1 / 2 ∧ Z.qT ≤ 1 / 2 ∧ Z.bT ≤ 0

end Circuit

section Stages

variable {C V D : Type}

/-- **Unit `u`'s luck at `p` in a round**: at its stage, whatever the past, `u` skipped passes with probability at
most `p` over the stage's coin. -/
def stagesLuck [Fintype C] (p : ℝ) (stg : V → ℕ) (u : V) : {m : ℕ} → CoinStages C V m → Prop
  | _, .done => True
  | _, .stage (m := m) skip ok next =>
      (stg u = m → prob (fun c => skip c u ∧ ok c u) ≤ p * prob (fun c => skip c u)) ∧
        ∀ c, stagesLuck p stg u (next c)

/-- Unit `u`'s luck at `p` in every round of a window. -/
def windowLuck [Fintype C] {M s : ℕ} (p : ℝ) (stg : V → ℕ) (u : V) : {n : ℕ} → StageWindow C V D M s n → Prop
  | _, .done => True
  | _, .round T next => stagesLuck p stg u T ∧ ∀ a, windowLuck p stg u (next a)

/-- **A unit's skip and receipt read off its fresh noise**: the stage's coin splits as `b` fresh bits and the rest
(`e`), and there is a `CoinRound` on the rest whose skipped value is the bits' sum, whose deficiency is the skip
decision, and on which `u` skipped and passing is lucky with fibre `Z` and nonzero weights. -/
def ReadsOffNoise (skip ok : C → V → Prop) (u : V) (b Z : ℕ) : Prop :=
  ∃ (Ω₀ J β : Type) (_ : Fintype Ω₀) (_ : Fintype J) (_ : DecidableEq J) (e : C ≃ Ω₀ × (J → Bool))
    (R : CoinRound Ω₀ J ℤ β),
    Fintype.card J = b ∧ ∀ c, (skip c u ↔ 0 < R.d (e c).1) ∧
      (skip c u → ok c u → R.Inj Z (e c).1 ∧ (∀ j, R.weight (e c).1 j ≠ 0) ∧ R.lucky (e c))

/-- `ReadsOffNoise` at unit `u`'s stage of a round, whatever the past. -/
def stagesNoise (stg : V → ℕ) (u : V) (b Z : ℕ) : {m : ℕ} → CoinStages C V m → Prop
  | _, .done => True
  | _, .stage (m := m) skip ok next =>
      (stg u = m → ReadsOffNoise skip ok u b Z) ∧ ∀ c, stagesNoise stg u b Z (next c)

/-- `ReadsOffNoise` at unit `u`'s stage in every round of a window. -/
def windowNoise {M s : ℕ} (stg : V → ℕ) (u : V) (b Z : ℕ) : {n : ℕ} → StageWindow C V D M s n → Prop
  | _, .done => True
  | _, .round T next => stagesNoise stg u b Z T ∧ ∀ a, windowNoise stg u b Z (next a)

end Stages

/-- **Every unit is lucky at most its template's receipt luck**, in every round. -/
def TemplateLuck {κ K C D : Type} [Fintype C] {M s n : ℕ} (Sc : Scheme κ K) (Cd : Declared κ)
    (W : StageWindow C (UnitOf Sc Cd) D M s n) : Prop :=
  ∀ u, windowLuck (luckOf Sc Cd u) (stageOf Sc Cd) u W

end Accounting.Compose
