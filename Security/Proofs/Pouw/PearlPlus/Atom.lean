module

public import Evaluate.TC.Dots
public import Definitions.Pouw.PearlC.TileCircuit

/-!
# Pearl+'s bridge, part (a): the check's FP8 atom is the Definitions' step

`PouwCheck.PearlPlus` runs the H100's E4M3 atom as the evaluator computes it (`Evaluate.TC.hopperE4m3Dot32`, on
`Evaluate.TC.Fp8Atom`'s grouped-sum model), in the chain, the forming and the debit replay. The game reads
`Pouw.Fp8Atom.Pipeline.step` at `hopperE4m3K32`. Code Lean may not read `Security/`, so the two models are one text in
two packages; the bridge is one commuting map per definition, as `Proofs.Properties.Core.Evaluate` does for the BF16
tensor-core spec.
-/

@[expose] public section

namespace Pouw.PearlPlus.Bridge

/-- The evaluator's grouped-sum pipeline as the Definitions' (the same three fields). -/
def pipelineToDef (p : Evaluate.TC.Fp8Atom.Pipeline) : Pouw.Fp8Atom.Pipeline := ⟨p.groups, p.width, p.zeroExp⟩

/-- **(a) The evaluator's FP8 step is the Definitions' step**, on every pipeline, accumulator word and operand codes. -/
def StepIsTheStep : Prop :=
  ∀ (p : Evaluate.TC.Fp8Atom.Pipeline) (acc : ℕ) (a b : List (Fin 256)),
    Evaluate.TC.Fp8Atom.Pipeline.step p acc a b = Pouw.Fp8Atom.Pipeline.step (pipelineToDef p) acc a b

/-- **(a) The H100 atom the check runs is `hopperE4m3K32`'s step** on its operands' codes (each word mod 256). -/
def AtomIsTheStep : Prop :=
  ∀ (acc : ℕ) (a b : List ℕ), Evaluate.TC.hopperE4m3Dot32 acc a b =
    Pouw.Fp8Atom.hopperE4m3K32.step acc (a.map Pouw.PearlC.TileCircuit.code) (b.map Pouw.PearlC.TileCircuit.code)

/-- The evaluator's term as the Definitions' (the same three fields). -/
def termToDef (t : Evaluate.TC.Fp8Atom.Term) : Pouw.Fp8Atom.Term := ⟨t.neg, t.exp, t.mag⟩

/-- The evaluator's value as the Definitions'. -/
def valueToDef : Evaluate.TC.Fp8Atom.Value → Pouw.Fp8Atom.Value
  | .nan => .nan
  | .inf s => .inf s
  | .fin t => .fin (termToDef t)

theorem valueToDef_nan : valueToDef .nan = .nan := rfl

theorem valueToDef_inf (s : Bool) : valueToDef (.inf s) = .inf s := rfl

theorem valueToDef_fin (t : Evaluate.TC.Fp8Atom.Term) : valueToDef (.fin t) = .fin (termToDef t) := rfl

theorem product_toDef (a b : Fin 256) :
    termToDef (Evaluate.TC.Fp8Atom.product a b) = Pouw.Fp8Atom.product a b := by
  simp only [Evaluate.TC.Fp8Atom.product, Pouw.Fp8Atom.product, apply_ite termToDef]
  rfl

theorem productTotal_toDef (a b : Fin 256) :
    valueToDef (Evaluate.TC.Fp8Atom.productTotal a b) = Pouw.Fp8Atom.productTotal a b := by
  simp only [Evaluate.TC.Fp8Atom.productTotal, Pouw.Fp8Atom.productTotal, apply_ite valueToDef,
    valueToDef_nan, valueToDef_fin, product_toDef]
  rfl

theorem fp32Value_toDef (w : Nat) : valueToDef (Evaluate.TC.Fp8Atom.fp32Value w) = Pouw.Fp8Atom.fp32Value w := by
  simp only [Evaluate.TC.Fp8Atom.fp32Value, Pouw.Fp8Atom.fp32Value, apply_ite valueToDef]
  rfl

theorem maxExp_toDef (z : Int) (ts : List Evaluate.TC.Fp8Atom.Term) :
    Evaluate.TC.Fp8Atom.maxExp z ts = Pouw.Fp8Atom.maxExp z (ts.map termToDef) := by
  simp only [Evaluate.TC.Fp8Atom.maxExp, Pouw.Fp8Atom.maxExp, List.foldl_map]
  rfl

theorem alignedSum_toDef (w : Nat) (E : Int) (ts : List Evaluate.TC.Fp8Atom.Term) :
    Evaluate.TC.Fp8Atom.alignedSum w E ts = Pouw.Fp8Atom.alignedSum w E (ts.map termToDef) := by
  simp only [Evaluate.TC.Fp8Atom.alignedSum, Pouw.Fp8Atom.alignedSum, List.map_map]
  rfl

theorem normalize_toDef (w : Nat) (E total : Int) :
    termToDef (Evaluate.TC.Fp8Atom.normalize w E total) = Pouw.Fp8Atom.normalize w E total := by
  simp only [Evaluate.TC.Fp8Atom.normalize, Pouw.Fp8Atom.normalize, apply_ite termToDef]
  rfl

theorem groupSum_toDef (w : Nat) (z : Int) (ts : List Evaluate.TC.Fp8Atom.Term) :
    termToDef (Evaluate.TC.Fp8Atom.groupSum w z ts) = Pouw.Fp8Atom.groupSum w z (ts.map termToDef) := by
  simp only [Evaluate.TC.Fp8Atom.groupSum, Pouw.Fp8Atom.groupSum, ← maxExp_toDef, ← alignedSum_toDef,
    normalize_toDef]

theorem groupStepTotal_toDef (w : Nat) (z : Int) (acc : Evaluate.TC.Fp8Atom.Value)
    (ps : List Evaluate.TC.Fp8Atom.Value) :
    valueToDef (Evaluate.TC.Fp8Atom.groupStepTotal w z acc ps) =
      Pouw.Fp8Atom.groupStepTotal w z (valueToDef acc) (ps.map valueToDef) := by
  have hcons : valueToDef acc :: ps.map valueToDef = (acc :: ps).map valueToDef := rfl
  have hnan : ((acc :: ps).map valueToDef).any Pouw.Fp8Atom.Value.isNaN =
      (acc :: ps).any Evaluate.TC.Fp8Atom.Value.isNaN := by
    rw [List.any_map]; congr 1; funext v; cases v <;> rfl
  have hinf : ∀ s, ((acc :: ps).map valueToDef).any (Pouw.Fp8Atom.Value.isInf s) =
      (acc :: ps).any (Evaluate.TC.Fp8Atom.Value.isInf s) := by
    intro s; rw [List.any_map]; congr 1; funext v; cases v <;> rfl
  have hfin : ((acc :: ps).map valueToDef).filterMap Pouw.Fp8Atom.Value.finite? =
      ((acc :: ps).filterMap Evaluate.TC.Fp8Atom.Value.finite?).map termToDef := by
    rw [List.filterMap_map, List.map_filterMap]; congr 1; funext v; cases v <;> rfl
  simp only [Evaluate.TC.Fp8Atom.groupStepTotal, Pouw.Fp8Atom.groupStepTotal, hcons, hnan, hinf, hfin,
    ← groupSum_toDef, apply_ite valueToDef,
    valueToDef_nan, valueToDef_inf, valueToDef_fin]
  rfl

theorem packValue_toDef (v : Evaluate.TC.Fp8Atom.Value) :
    Evaluate.TC.Fp8Atom.packValue v = Pouw.Fp8Atom.packValue (valueToDef v) := by
  cases v <;> rfl

theorem fold_toDef (p : Evaluate.TC.Fp8Atom.Pipeline) (gs : List Nat) (v : Evaluate.TC.Fp8Atom.Value)
    (ab : List (Fin 256 × Fin 256)) :
    valueToDef (p.fold gs v ab) = (pipelineToDef p).fold gs (valueToDef v) ab := by
  induction gs generalizing v ab with
  | nil => rfl
  | cons size gs ih =>
    simp only [Evaluate.TC.Fp8Atom.Pipeline.fold, Pouw.Fp8Atom.Pipeline.fold, ih, groupStepTotal_toDef, List.map_map]
    congr 2
    apply List.map_congr_left
    intro x _
    exact productTotal_toDef x.1 x.2

/-- **(a)**: the evaluator's FP8 step is the Definitions' step. -/
theorem stepIsTheStep : StepIsTheStep := by
  intro p acc a b
  simp only [Evaluate.TC.Fp8Atom.Pipeline.step, Pouw.Fp8Atom.Pipeline.step, packValue_toDef, fold_toDef,
    fp32Value_toDef]
  rfl

/-- **(a)**: the H100 atom the check runs is `hopperE4m3K32`'s step. -/
theorem atomIsTheStep : AtomIsTheStep := fun acc a b =>
  stepIsTheStep Evaluate.TC.Fp8Atom.hopperE4m3K32 acc (a.map Evaluate.TC.Fp8Atom.code)
    (b.map Evaluate.TC.Fp8Atom.code)

end Pouw.PearlPlus.Bridge
