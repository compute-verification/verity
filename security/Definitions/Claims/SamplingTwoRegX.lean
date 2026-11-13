module

public import Definitions.Claims.SamplingGenericX
public import Definitions.Claims.SamplingTwoReg

/-!
# Claims: the two-stage audit's committed form, at a boundary trace read at the statement

`CompiledTwo.KSRegTwo`, `CompiledTwo.HeadlineTwo` and `CompiledTwo.MissTwo` read the boundary at `Compiled.regVal`, the
committed trace at the prover's tape, so whatever the tape holds reaches the replay blocks wrong at the boundary
(`wrongAtBoundaryAt`). This file states the three at the boundary trace read at the statement (`Compiled.regValX`),
which reads the statement and the prover's strategy and no tape, beside the three, which it leaves as they stand:

1. **The incorrect set** (`FregTwoX`): `FregTwo` at `Compiled.regValX` and each replay block's committed interior
   (`iregVal`). The boundary is read at the statement; the interior stays at the tape and the first stage's coins,
   which fix it before the second stage's.
2. **The error** (`εregTwoX`): `Compiled.εregX` and the interior's binding across runs, `N·√Pr[B_conf,2]`.
3. **The claims**: `CompiledTwo.KSRegTwoX`, `CompiledTwo.HeadlineTwoX` and `CompiledTwo.MissTwoX`, whose replay blocks
   wrong at the boundary are one set at every tape; and the compiler's statements.

Statements only: definitions, no proofs.
-/

@[expose] public section

namespace Claims.SamplingGeneric

open Claims
open Claims.Sampling (AuditStatement Partition wrongInAt footprintAt wrongAtBoundaryAt)

namespace CompiledTwo

variable {Coin : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin] {A : Arg Coin}

/-! ## 1. The incorrect set -/

open Classical in
/-- **The incorrect set at the boundary trace read at the statement**, at the prover's tape `ω` and the run's coins
`r`: over the grid of replay blocks and strata, the proof blocks incorrect at that trace (`Compiled.regValX`) and their
replay block's committed interior (`iregVal`); empty on a statement that doesn't parse or whose partitions don't
evaluate. It is `FregTwo` with the boundary read at the statement: it reads the tape and `r` through the interiors
only, and `r` off the second stage only. -/
noncomputable def FregTwoX (H : Msg → Msg) (L : CompiledTwo Coin A) {Ω : Type} [Fintype Ω] (P : Prover Ω) (x : Msg)
    (ω : Ω) (r : Fin (L.V.k x) → Coin) : Finset ℕ :=
  match L.parse x with
  | none => ∅
  | some s =>
    match L.partition s, L.two s with
    | some Pt, some d =>
      (Finset.range d.part.blocks.length ×ˢ Finset.range Pt.strata.length).biUnion fun p =>
        wrongInAt s Pt d.part (L.regValX H P x) p.1 p.2 (L.iregVal H P x ω r p.1)
    | _, _ => ∅

/-! ## 2. The error -/

/-- **The committed form's error at two stages, at the boundary trace read at the statement**: `Compiled.εregX`'s four
terms and the committed interior's binding across runs,
`ε_reg,2,X = κ_s + √(k(x)·γ_s·Pr[B]) + N·√Pr[B_conf,X] + Pr[B_conf,X] + N·√Pr[B_conf,2]`, `N` the program's gates. The
interior is read as in `εregTwo`, at the finder whose two runs share a tape and the first stage's coins
(`BconfTwo`). -/
noncomputable def εregTwoX (H : Msg → Msg) (L : CompiledTwo Coin A) (x : Msg) {Ω : Type} [Fintype Ω] [Nonempty Ω]
    (P : Prover Ω) : ℝ :=
  L.εregX H x P + L.nG x * Real.sqrt ((BconfTwo H L Ω P x).Pr H)

/-! ## 3. The claims -/

/-- **KS, the committed form at two stages, at the boundary trace read at the statement**, on the statements in scope,
at every prover: the run accepts with a sampled proof block in that trace's incorrect set (`FregTwoX`) with probability
at most `ε_reg,2,X` (`εregTwoX`); so, by the two stages' samples, at every `δ ≥ 0` it accepts with no interiors
putting that trace's footprint in `M̄_δ` with probability at most `δ + ε_reg,2,X`. It is `KSRegTwo` with the boundary
read at the statement: the footprints it speaks of are those of one boundary trace at every tape. -/
def KSRegTwoX (H : Msg → Msg) (L : CompiledTwo Coin A) : Prop :=
  ∀ x, L.Scope x → ∀ (Ω : Type) [Fintype Ω] [Nonempty Ω] (P : Prover Ω),
    Pr (fun p : Ω × (Fin (L.V.k x) → Coin) => (run H P L.V x p.1 p.2).out = .acc ∧
        ∃ u ∈ L.checked x p.2, u ∈ L.FregTwoX H P x p.1 p.2) ≤ L.εregTwoX H x P ∧
      ∀ s Pt d, L.parse x = some s → L.partition s = some Pt → L.two s = some d → ∀ δ : ℝ, 0 ≤ δ →
        Pr (fun p : Ω × (Fin (L.V.k x) → Coin) => (run H P L.V x p.1 p.2).out = .acc ∧
          ∀ ι : ℕ → ℕ → Bool, footprintAt s Pt d.part (L.regValX H P x) ι ∉ d.M Pt δ) ≤
          δ + L.εregTwoX H x P

/-- **The headline at two stages, at the boundary trace read at the statement**: on a statement in scope, at every
prover, the run accepts with probability at most the two-stage design's `P̄_a` at that trace's footprint at its best
interiors, plus `ε_reg,2,X`. A bound `pb` on `P̄_a` at every interior stands for the best interiors'. The trace is one
at every tape, so the bound is one number where `HeadlineTwo` takes an expectation over the tape. -/
def HeadlineTwoX (H : Msg → Msg) (L : CompiledTwo Coin A) : Prop :=
  ∀ x s Pt d, L.Scope x → L.parse x = some s → L.partition s = some Pt → L.two s = some d →
    ∀ (Ω : Type) [Fintype Ω] [Nonempty Ω] (P : Prover Ω) (pb : ℝ),
      (∀ ι, d.Pa Pt (footprintAt s Pt d.part (L.regValX H P x) ι) ≤ pb) →
        Pr (fun p : Ω × (Fin (L.V.k x) → Coin) => (run H P L.V x p.1 p.2).out = .acc) ≤ pb + L.εregTwoX H x P

/-- **The miss at the replay blocks wrong at the boundary trace read at the statement**: on a statement in scope whose
first stage catches each replay block at rate `r₁` and whose second stage samples each cell at rate `r₂`, at every
prover and tape, the run's sample holds no proof block of the incorrect set (`FregTwoX`) with probability at most
`(1 − r₁·r₂)^{|W|}`, `W` the replay blocks wrong at that trace (`wrongAtBoundaryAt`). `W` reads the statement and the
prover's strategy and no tape: whatever fixes those two fixes it, where `MissTwo`'s is fixed by the tape. -/
def MissTwoX (H : Msg → Msg) (L : CompiledTwo Coin A) (r₁ r₂ : ℝ) : Prop :=
  ∀ x s Pt d, L.Scope x → L.parse x = some s → L.partition s = some Pt → L.two s = some d →
    d.FirstRate r₁ → d.SecondRate Pt r₂ → ∀ (Ω : Type) [Fintype Ω] (P : Prover Ω) (ω : Ω),
      Pr (fun r : Fin (L.V.k x) → Coin => ∀ u ∈ L.checked x r, u ∉ L.FregTwoX H P x ω r) ≤
        (1 - r₁ * r₂) ^ (wrongAtBoundaryAt s Pt d.part (L.regValX H P x)).card

end CompiledTwo

/-- **The two-stage compiler's committed form at the boundary trace read at the statement**: at `CompilerKSRegTwo`'s
premises, every compiled two-stage verifier meets `CompiledTwo.KSRegTwoX`. -/
def CompilerKSRegTwoX : Prop :=
  ∀ (Coin : Type) [Fintype Coin] [Nonempty Coin] [Inhabited Coin] (A : Arg Coin) (L : CompiledTwo Coin A)
    (H : Msg → Msg), A.KS H → L.commit.Binds H → L.Sessions H → L.KSRegTwoX H

/-- **The two-stage miss bound at the boundary trace read at the statement**: every compiled two-stage verifier that
meets the contract meets `CompiledTwo.MissTwoX` at all rates `0 ≤ r₁ ≤ 1` and `0 ≤ r₂ ≤ 1`, from its draw laws. -/
def CompilerMissTwoX : Prop :=
  ∀ (Coin : Type) [Fintype Coin] [Nonempty Coin] [Inhabited Coin] (A : Arg Coin) (L : CompiledTwo Coin A)
    (H : Msg → Msg), L.Sessions H → ∀ r₁ r₂ : ℝ, 0 ≤ r₁ → r₁ ≤ 1 → 0 ≤ r₂ → r₂ ≤ 1 → L.MissTwoX H r₁ r₂

/-- **The two-stage headline at the boundary trace read at the statement**: at `CompilerKSRegTwo`'s premises, every
compiled two-stage verifier meets `CompiledTwo.HeadlineTwoX`. `CompilerHeadlineTwo` takes the committed form as a
hypothesis; this one is stated at the premises that prove it, so it reads no premise the committed form doesn't. -/
def CompilerHeadlineTwoX : Prop :=
  ∀ (Coin : Type) [Fintype Coin] [Nonempty Coin] [Inhabited Coin] (A : Arg Coin) (L : CompiledTwo Coin A)
    (H : Msg → Msg), A.KS H → L.commit.Binds H → L.Sessions H → L.HeadlineTwoX H

end Claims.SamplingGeneric
