import Proofs.Flock.C3.Fork.Sparse
import Proofs.Flock.C3.Fork.RegOpen

/-!
# Piece (a), the oracle count: candidate 3's table with its oracles sent whole

`holoTable_sound`, `table_sound`'s analogue at `Model.holoTable` (`holoTableOf`): every prover makes the table accept
while no table close to the registered word has its hidden statement satisfied by a codeword near the committed level-0
table, with probability at most `holoError`. a1s and a1r are used here:

* within the unique-decoding radius at most one message `M` is near `w` (`card_closeL_ud_le`), and its cells are a table
  `R` (`RegSchedule.Close`); no claim reads the padding (`RegClaim.coord` is zero there), so a claim holds on
  `witnessOf M` iff it holds on `R.bits`;
* a rep whose hidden fold `h` is `R`'s (`sparseFold` at `R.bits`, which needs `hunits`) is a plain rep of
  `R.toStatement`, counted as `rep_sound` counts one from a doomed start (`repError`), after the link points as
  `link_sound` counts them (`linkError`);
* a rep whose `h` is not `R`'s: its sparse phase hands on a claim that holds on `R.bits` with probability at most
  `sparseError` (`sparse_sound`);
* a claim that fails on `R.bits`, or no message near `w`: R's opening accepts with probability at most `regOpenError`
  (`regOpen_sound`, at `fast100`'s schedule for `mReg`; at the empty schedule Ligerito rejects);
* each rep with its sparse phase plays on its own coins (the sparse phase reads only its rep's outputs), so the two
  multiply as `value_interleave_le` multiplies the reps: `(repError + sparseError)^2`.
-/

namespace FlockC3.Fork

open FlockSoundness FlockSoundness.Discharge FlockSoundness.Discharge.Recursion
open Game Model Accounting
open scoped ENNReal

variable {F K : Type} [Field F] [Fintype F] [DecidableEq F] [Field K] [Fintype K] [DecidableEq K]

/-- **(a1o) Candidate 3's table is sound with its oracles sent whole.** At the reps' schedule `fast100`'s, a link
layout at every table, R's schedule empty or `fast100`'s at `mReg` and every unit slot within the table: for every
registered word `w` and every prover, the table accepts while no table close to `w` has a codeword near the committed
level-0 table packing a witness of its hidden statement with probability at most `holoError`. -/
theorem holoTable_sound (A : Arith F K) (hA : A.Correct) (e : ExecHolo)
    (hsch : Accounting.fast100 e.cls.m = some e.sch)
    (hlay : ∀ R : RegTable e.cls, (R.toStatement e.τ e.regs).LinkLayout e.cls.ptLocal e.cls.mPts)
    (hR : e.rsch.sch.levels = [] ∨ Accounting.fast100 e.cls.shape.mReg = some e.rsch.sch)
    (hunits : ∀ σ, ∀ u ∈ e.cls.units σ, u < 2 ^ (e.cls.kLog - e.cls.slotLog)) (w : Oracle F)
    (σ : Strategy (holoTableOf A e.cls e.regs e.sch e.rsch e.τ w)) :
    prob (fun out => out.1 = true ∧ ¬ ∃ R : RegTable e.cls, e.rsch.Close A w R ∧
        Committed A (R.toStatement e.τ e.regs) e.sch out.2)
      (holoTableOf A e.cls e.regs e.sch e.rsch e.τ w) σ ≤ ENNReal.ofReal (holoError e) := by
  sorry

end FlockC3.Fork
