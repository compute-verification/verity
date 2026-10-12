import Proofs.Flock.Recursive.HoloExec
import Proofs.Flock.Recursive.ClashFree
import Proofs.Flock.C3.Fork.CouplingLock

/-!
# a1l, step 5: no accepted run in `W` disagrees with `W`'s tables

The coupling's oracle prover sends `W`'s tables (`chH`): at a rep's slot, the rows the reps' part of an outcome of `W`
opens there (`tabRW` at `WC W`); at R's, the rows R's part opens (`tabRW` at `WR W x`, R's stream as rep 0 of a
compiled table at the registered cap). `holo_bad_zero`: where the tables a run records are those, an accepted run in
`W` that disagrees with them clashes with a run in `W` (`ClashHL`): on the reps as in `bad_zero_of_fixed`, on R's
levels `ℓ ≥ 1` the same way at R's part, and on R's level 0, whose table is the registered word, by a `RegClash`.
-/

namespace FlockC3.Fork

open FlockSoundness FlockSoundness.Game FlockSoundness.Model FlockSoundness.Accounting
open FlockSoundness.Discharge.Exec FlockSoundness.Discharge.Recursion

variable {F K : Type} [Field F] [Fintype F] [DecidableEq F] [Field K] [Fintype K] [DecidableEq K] {D Sl : Type}

/-- The reps' parts of `W`'s outcomes, as compiled-table outcomes. -/
def WC (W : OutHL F D Sl → Prop) (c : OutCL F D Sl) : Prop := ∃ o, W o ∧ o.circ = c

/-- R's parts of `W`'s outcomes, as compiled-table outcomes whose rep 0 is R's stream at the registered cap. -/
def WR (W : OutHL F D Sl → Prop) (x : HoloReg F D) (c : OutCL F D Sl) : Prop := ∃ o, W o ∧ o.reg x.cap = c

/-- **`W`'s tables**, at the lockstep's slots: a rep's at the even ones, R's at the odd ones. -/
noncomputable def chH (H : List UInt8 → D) (E : Merkle.Enc (List F) D) (Lf : Merkle.Leaf (List F) Sl D H)
    (sch rsch : Schedule) (W : OutHL F D Sl → Prop) (x : HoloReg F D) (s : ℕ) (d : Cap D) : Oracle F :=
  if s % 2 = 0 then tabRW H E Lf sch (WC W) (s / 2) d else tabRW H E Lf rsch (WR W x) (s / 2) d

/-- **An accepted run in `W` agrees with `W`'s tables**: at the level-0 table `W`'s at the run's cap, R's level 0 the
registered word, and the tables the run records `W`'s (`chH`), disagreeing would be a clash of the run with a run of
`W`. -/
theorem holo_bad_zero (H : List UInt8 → D) (E : Merkle.Enc (List F) D) (Lf : Merkle.Leaf (List F) Sl D H)
    (sch rsch : Schedule) (x : HoloReg F D) (W : OutHL F D Sl → Prop)
    (hW : ∀ o o', W o → W o' → ¬ ClashHL H E Lf sch rsch x o o') (cap₀ : Cap D)
    (i : (IRep F D × IRep F D) × IRep F D) (v : Bool) (b : OutHL F D Sl)
    (hR : TopRelH H E Lf sch rsch (tab₀W H E Lf sch (WC W) cap₀) x.w cap₀ x.cap i v b)
    (hfix : ∀ s t d, getEO (Lock.getPair (getRep 0) (getRep 1)) (getRep 0) i s = some (t, d) →
      t = chH H E Lf sch rsch W x s d) :
    ind (fun out : OutHL F D Sl => out.accepted = true ∧ W out ∧
      ¬ (AgreeRep 0 (tab₀W H E Lf sch (WC W) cap₀) i.1.1 out.pend.1.reqs out.opens ∧
        AgreeRep 1 (tab₀W H E Lf sch (WC W) cap₀) i.1.2 out.pend.2.reqs out.opens ∧
        AgreeRep 2 x.w i.2 out.pendR.reqs out.opens)) b ≤ 0 := by
  sorry

end FlockC3.Fork
