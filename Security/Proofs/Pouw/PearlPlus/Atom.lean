import Evaluate.TC.Dots
import Definitions.Pouw.PearlC.TileCircuit

/-!
# Pearl+'s bridge, part (a): the check's FP8 atom is the Definitions' step

`PouwCheck.PearlPlus` runs the H100's E4M3 atom as the evaluator computes it (`Evaluate.TC.hopperE4m3Dot32`, on
`Evaluate.TC.Fp8Atom`'s grouped-sum model), in the chain, the forming and the debit replay. The game reads
`Pouw.Fp8Atom.Pipeline.step` at `hopperE4m3K32`. Code Lean may not read `Security/`, so the two models are one text in
two packages; the bridge is one commuting map per definition, as `Proofs.Properties.Evaluate` does for the BF16
tensor-core spec.
-/

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

end Pouw.PearlPlus.Bridge
