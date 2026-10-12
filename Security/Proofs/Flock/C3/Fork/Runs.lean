import Proofs.Flock.C3.Fork.Sparse
import Proofs.Flock.C3.Fork.RegOpen

/-!
# Piece (a), the assembly: candidate 3's table on one prover's runs

`holo_runs_sound`, `compiled_runs_sound`'s analogue at `Model.holoTableCL`: two runs of the prover's own strategy that
clash (`ClashHL`), or the table accepts with probability at most `holoError`. Its proof composes three counts:

* the reps, as `compiled_runs_sound` counts them (the link points and `repError`, every off-plurality opening a clash
  of two runs on the reps' streams), at each table `R` decoded from the registered word;
* each rep's sparse phase, `sparse_sound` (`C3/Fork/Sparse.lean`) at `z = R.bits`: where the rep's hidden fold `h` is
  `R`'s (`sparseFold`), the rep is a rep of `R.toStatement`, which no witness satisfies, and otherwise its registered
  claim fails on `R.bits` but with probability `sparseError`. That `sparseFold` at `R.bits` is the hidden part of the
  lincheck's fold needs every unit slot in range (`hunits`: `U` reads a unit's low `k_log − ℓ` bits);
* R's opening, `regOpen_sound` (`C3/Fork/RegOpen.lean`) at `fast100`'s schedule for `mReg` (`hR`'s right side; on its
  left, the empty schedule, Ligerito rejects): within the unique-decoding radius at most one message is near `w`, every
  table is some message's (`RegTable.bits`), and a level-0 opening off `w` is a `RegClash`.
-/

namespace FlockC3.Fork

open FlockSoundness FlockSoundness.Discharge FlockSoundness.Discharge.Recursion
open Game Model Accounting Exec StrictCR
open scoped ENNReal

variable {F K : Type} [Field F] [Fintype F] [DecidableEq F] [Field K] [Fintype K] [DecidableEq K] {D Sl : Type}
variable {nT r : ℕ} (T : Fin nT → Type) (CT : Fin r → Type) [∀ k, Fintype (CT k)] [∀ k, Nonempty (CT k)]

/-- **Candidate 3's table on one prover's runs.** At a registration whose registrant opens every level-0 position and
no table near whose word makes the hidden statement true, for every `G` and every prover `s` of `holoInnerCL`: two
transcripts `s` reaches that satisfy `G` play the table to outcomes that clash (`ClashHL`), or `s` makes the inner
verifier accept a transcript satisfying `G` with probability at most `holoError`. -/
theorem holo_runs_sound (A : Arith F K) (hA : A.Correct) [Inhabited D] [Inhabited Sl]
    (H : List UInt8 → D) (E : Merkle.Enc (List F) D) (Lf : Merkle.Leaf (List F) Sl D H) (e : ExecHolo)
    (hsch : Accounting.fast100 e.cls.m = some e.sch)
    (hlay : ∀ R : RegTable e.cls, (R.toStatement e.τ e.regs).LinkLayout e.cls.ptLocal e.cls.mPts)
    (hR : e.rsch.sch.levels = [] ∨ Accounting.fast100 e.cls.shape.mReg = some e.rsch.sch)
    (hunits : ∀ σ, ∀ u ∈ e.cls.units σ, u < 2 ^ (e.cls.kLog - e.cls.slotLog))
    (x : {x : HoloReg F D // x.Opens H E Lf e.rsch.sch.level₀}) (hno : ¬ (holoInnerCL T CT A H E Lf e).holds x)
    (G : List (List (SMsg T) × ((k : Fin r) → CT k)) → Prop) (s : Strategy (holoInnerCL T CT A H E Lf e).game) :
    (∃ t t' o o', Game.Outcome (holoInnerCL T CT A H E Lf e).game s t ∧
      Game.Outcome (holoInnerCL T CT A H E Lf e).game s t' ∧ G t ∧ G t' ∧
      PlaysT (holoTableCL A H E Lf (e.cls.pubStatement e.regs) e.sch e.cls.hm e.cls.ptLocal e.cls.mPts e.cls.shape
        e.cls.units (fun σ => (e.τ σ).val) e.rsch.sch x.1.cap) (schedSteps T CT 0 t) o ∧
      PlaysT (holoTableCL A H E Lf (e.cls.pubStatement e.regs) e.sch e.cls.hm e.cls.ptLocal e.cls.mPts e.cls.shape
        e.cls.units (fun σ => (e.τ σ).val) e.rsch.sch x.1.cap) (schedSteps T CT 0 t') o' ∧
      ClashHL H E Lf e.sch e.rsch.sch x.1 o o') ∨
    prob (fun t => (holoInnerCL T CT A H E Lf e).accepts x t ∧ G t) (holoInnerCL T CT A H E Lf e).game s ≤
      ENNReal.ofReal (holoError e) := by
  sorry

end FlockC3.Fork
