module

public import Definitions.Claims.SamplingTwoReg
public import Proofs.Flock.SamplingGeneric.Reg
public import Proofs.Flock.SamplingGeneric.TwoCount

/-!
# The two-stage committed form's event

The primary conjunct of `CompiledTwo.KSRegTwo`, from the sessions' bound: an accepted run whose sample holds a proof
block of the committed incorrect set (`CompiledTwo.FregTwo`) has its session unfound, or its extractions disagree with
themselves or open a gate off the committed boundary trace (`Compiled.regVal`, the one-stage form's events at
`L.toCompiled`) or off the committed interior (`CompiledTwo.iregVal`).

* `correct_refIn_congr`: a proof block's correctness on its replay block's values reads the boundary values only below
  the program's gates and the interior values only at the replay block's gates.
-/

@[expose] public section

namespace Claims.Sampling

/-- **What a proof block's correctness reads of its replay block's values**: on a topological program that `Pt`
partitions, a proof block of `Pt` is correct at `refIn rb val ι` exactly when it is at `refIn rb val' ι'`, for boundary
values that agree below the program's gates and interior values that agree at `rb`'s gates below them. -/
theorem correct_refIn_congr {C : Prog} {Pt : Partition} (htop : C.Topological) (hP : Pt.Partitions C) {v : ℕ}
    {u : List ℕ} (hu : Pt.blocks[v]? = some u) (rb : List ℕ) {val val' : ℕ → Option Bool} {ι ι' : ℕ → Bool}
    (hval : ∀ g < C.gates.length, val g = val' g) (hι : ∀ g < C.gates.length, g ∈ rb → ι g = ι' g) :
    C.Correct (C.refIn rb val ι) u ↔ C.Correct (C.refIn rb val' ι') u := by
  have href : ∀ g < C.gates.length, C.refIn rb val ι g = C.refIn rb val' ι' g := by
    intro g hg
    unfold Prog.refIn
    split_ifs with h
    · rw [hι g hg h.1]
    · exact hval g hg
  have hlt : ∀ g ∈ u, g < C.gates.length := fun g hg => hP.2.1 g (mem_gates hu hg)
  exact C.correct_congr (fun g hg => href g (hlt g hg)) fun g hg x hx i hi _ =>
    href i ((htop.1 g x hx i hi).trans (hlt g hg))

end Claims.Sampling
