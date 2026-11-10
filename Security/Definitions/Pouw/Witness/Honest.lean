module

public import Definitions.Pouw.Basic.QTree
public import Definitions.Pouw.Game.Deadline
public import Definitions.Pouw.M1
public import Definitions.Pouw.Witness.Model

/-!
# The witnesses' instances (trusted definitions; used only by the satisfiability witnesses)

What the witnesses of PoUW's guarantees, and of the compute guarantees PoUW gives, are stated at:

* the milestone's workload `M1.sixteen`: sixteen milestone-shape units on one zero weight;
* the honest reference as an oracle program (`honestProg`: each unit queries its noise, then commits) and as an
  adversary of the output-priced model (`pricedHonest`);
* the deadline's instances (`Witness.Deadline`): the run that delivers the honest reference at the deadline, and a
  protocol that compares nothing, on one unit of shape `0×0×0`, with an adversary that commits it at no cost on a run
  that spends nothing.
-/

@[expose] public section

namespace Pouw.Proofs

open QTree

variable {Q R S : Type}

section Honest

variable (P : D1.Params Q R S)

/-- Unit `u`'s honest commitment on activations `A`, from the answers `a` to `qAct` and `b` to `qWt` (the formulas
of `D1.Params.refChecked` and `refOut` with the oracle's answers as inputs). -/
def commitAns (U : Workload) (u : ℕ) (A : Mat) (a b : List R) : Commit :=
  let sh := U.layout.shape u
  let E := P.expAct sh a b
  let F := P.expWt sh b
  let act' : Mat := fun i l => wrap32 (A i l + mmW P.r E.1 E.2 i l)
  let wt' : Mat := fun l j => wrap32 (U.fixed (U.layout.fixedId u) l j + mmW P.r F.1 F.2 l j)
  ⟨A, fun t => mmW (D1.blockEnd P.d sh.k t) act' wt',
    fun i j => wrap32 (wrap32 (mmW sh.k act' wt' i j - mmW P.r (mmW sh.k A F.1) F.2 i j)
      - mmW P.r E.1 (mmW sh.k E.2 wt') i j)⟩

/-- Unit `u`'s honest program: query its noise, then commit. -/
def unitProg (U : Workload) (act : ℕ → Mat) (s : S) (u : ℕ) : QTree Q R Commit :=
  (queryAll (P.qAct s u (act u) (U.layout.fixedId u))).bind fun a =>
    (queryAll (P.qWt s u (U.layout.fixedId u))).bind fun b => done (commitAns P U u (act u) a b)

/-- The honest reference: every unit's honest program, in order. -/
def honestProg (U : Workload) (act : ℕ → Mat) (s : S) : QTree Q R Transcript :=
  upTo (unitProg P U act s) U.layout.N

end Honest

/-- The honest reference as an adversary of the witness model. -/
noncomputable def pricedHonest {Q R : Type} (P : D1.Params Q R Salt) : Adversary (outputPriced Q R Salt) :=
  ⟨done [], fun U act _ s => honestProg P U act s, fun _ => 0⟩

namespace M1

/-- Sixteen milestone-shape units on one zero weight: one full group of `2^16` rows, the least the domain allows. -/
def sixteen : Workload := ⟨⟨16, fun _ => milestoneShape, fun _ => 0⟩, fun _ => 0⟩

end M1

end Pouw.Proofs

namespace Pouw.Witness.Deadline

/-- The honest run: the salt at `0`, the transcript delivered at `1`, and the milestone's `16·2^44` spent at a constant
rate over `(0, 1]`. -/
noncomputable def honestRun : TimedRun D1.M1Q D1.M1R Salt := ⟨0, fun _ _ => 1, fun _ _ t => t * (16 * 2 ^ 44)⟩

/-- A protocol that compares nothing, every unit owning one ticket. -/
def acceptAll : Protocol Unit Unit Unit :=
  ⟨fun _ _ _ _ _ => 0, fun _ => ∅, fun _ _ _ _ _ => 0, fun _ _ _ _ _ => 0, fun _ _ => 1⟩

/-- One unit of shape `0×0×0`: under `acceptAll` every commitment to it is correct. -/
def dot : Workload := ⟨⟨1, fun _ => ⟨0, 0, 0⟩, fun _ => 0⟩, fun _ => 0⟩

/-- Commits every unit, with no preprocessing, in the free model. -/
def commitAll : Adversary (freeModel Unit Unit Unit) :=
  ⟨QTree.done [], fun _ _ _ _ => QTree.done fun _ => some ⟨0, 0, 0⟩, fun _ => 0⟩

/-- The run that delivers at `0` and spends nothing. -/
noncomputable def idleRun : TimedRun Unit Unit Unit := ⟨0, fun _ _ => 0, fun _ _ _ => 0⟩

instance instNonemptyDotTicket : Nonempty (Ticket (acceptAll.Wref dot.layout) dot.layout.N) :=
  ⟨⟨⟨0, by decide⟩, ⟨0, by decide⟩⟩⟩

end Pouw.Witness.Deadline
