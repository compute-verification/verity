import Proofs.Flock.Claims.Complete.Reps
import Proofs.Flock.Claims.Complete.Replay.RingSwitchAlgebra

/-!
# The replay goals from the session walk alone

At the deployed relation, the skeleton's `Zerocheck`, `Lincheck` and `RingSwitch` need only what the session walk
establishes for each rep: `ZcAt`, `LcWalk` and `RsAt`.

* `Run.fold_ok`: the statement's circuit fold at the lincheck's `α` returns at every rep. The scope's untyped row
  circuit passed `HmRow.check`, so each range's slot type folds, and the zerocheck's `ρ` has the block's size
  (`JsonRT.fold_ok`).
* `LcWalk`: `LcAt` without the fold, which `Run.lcAt` adds back.
* `replay_goals`: the three goals at `Readable.relation` from `ZcAt`, `LcWalk` and `RsAt`.
-/

namespace FlockComplete.Replay

open Flock FlockDeploy Transcript

variable {H : Claims.Msg → Claims.Msg} {rel : Readable.Parsed → Readable.Relation}

/-- **The statement's circuit fold returns** at the lincheck's `α`, at every table and rep of every honest run. -/
theorem _root_.FlockComplete.Run.fold_ok (o : Run H rel) {i : ℕ} (hi : i < o.tabs.size) (r : ℕ) :
    o.tabs[i]!.fold.fold ((o.lcCoins i r).ca 0) (skipWeights (o.zcOut i r).z)
      ((o.zcOut i r).rho.extract 0 (o.tabs[i]!.fold.kLog - 6)) = .ok (o.full i r) := by
  refine ok_val ?_
  obtain ⟨st, hsts, -, -, -, ⟨tags, file, tables, hty, hcirc⟩, hm, hk⟩ := o.table
  obtain ⟨ls, hcheck⟩ := JsonRT.circuitOf_check hty hcirc
  have hi0 : i = 0 := by
    have : o.tabs.size = 1 := by simp [Run.tabs, tabsOf, hsts]
    omega
  subst hi0
  have htab : o.tabs[0]! = Setup.ofCircuit st := by simp [Run.tabs, tabsOf, hsts]
  have hρ : ((o.zcOut 0 r).rho.extract 0 (o.tabs[0]!.fold.kLog - 6)).size = st.c.kLog - 6 := by
    simp only [Run.zcOut, Zc.out, Array.size_extract, pushes_size, htab, Setup.ofCircuit, Stmt.circuitFold]
    omega
  rw [htab] at hρ ⊢
  exact JsonRT.fold_ok st hk (JsonRT.hm_check_kinds hcheck) _ (FlockSoundness.Refine.skipWeights_size _) hρ

/-- **What the session walk establishes for table `i`'s rep `r`'s lincheck**: the decoded proof's lincheck is the
prover's, and the recorded stream carries the prover's rounds through lincheck. -/
def LcWalk (o : Run H rel) (i r : ℕ) : Prop :=
  (o.proof i r).lincheck =
      Lc.proof o.tabs[i]!.fold o.tabs[i]!.m (o.trace i) (o.zcOut i r) (o.full i r) (o.lcCoins i r) ∧
    Agrees (o.stream i r).rounds
      (Lc.tEnd o.tabs[i]!.fold o.tabs[i]!.m (o.trace i) (o.zcOut i r) (o.full i r) (o.lcCoins i r)
        (o.zcEnd i r)).rounds 0

theorem _root_.FlockComplete.Run.lcAt (o : Run H rel) {i : ℕ} (hi : i < o.tabs.size) {r : ℕ} (h : LcWalk o i r) :
    LcAt o i r :=
  ⟨o.fold_ok hi r, h⟩

/-- **The replay goals** at the deployed relation, at every rep the session walk aligns. -/
theorem replay_goals (H : Claims.Msg → Claims.Msg)
    (hzc : ∀ o : Run H Readable.relation, ∀ i < o.tabs.size, ∀ r < 2, ZcAt o i r)
    (hlc : ∀ o : Run H Readable.relation, ∀ i < o.tabs.size, ∀ r < 2, LcWalk o i r)
    (hrs : ∀ o : Run H Readable.relation, ∀ i < o.tabs.size, ∀ r < 2, RsAt o i r) :
    Zerocheck H Readable.relation ∧ Lincheck H Readable.relation ∧ RingSwitch H Readable.relation :=
  have hlc' : ∀ o : Run H Readable.relation, ∀ i < o.tabs.size, ∀ r < 2, LcAt o i r :=
    fun o _ hi _ hr => o.lcAt hi (hlc o _ hi _ hr)
  ⟨zerocheck_goal' H hzc, lincheck_relation H hzc hlc', ringSwitch_relation H hzc hlc' hrs⟩

end FlockComplete.Replay
