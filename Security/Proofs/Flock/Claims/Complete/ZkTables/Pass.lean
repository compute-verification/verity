import Proofs.Flock.Claims.Complete.ZkTables.Files
import Proofs.Flock.Claims.Complete.ZkTables.Table
import Proofs.Flock.Claims.Complete.Zk.Exact
import Proofs.Flock.Soundness.Discharge.ZkExec.RunsSeams

/-!
# A masked session that returns passed every refusal at its slot

`sessionPZk_pass`: one induction over the session's reps. Where the pure masked session at the replies `resp` returns,
each rep `q` started at `repStartZk q` sends and passed its refusals there (`PassAt`): level 0's pair coins, the
replies at `rhoSlotsZk`'s slot and the next, are independent, and the inner `β`, the reply at `betaSlotsZk`'s slot, is
not `0`. Each rep that returns sent exactly its `repRounds` (`repPZk_exact`), so the next starts where `repStartZk`
says. Read the other way, at replies where some rep's slot refuses, the session does not return: it stops at that
refusal or at an earlier one.
-/

namespace FlockComplete.ZkTables

open Flock Transcript FlockComplete FlockDeploy JsonRT

/-- **Rep `q` passed its refusals at its slots**: at its table's `--zk` schedule, level 0's pair coins (`rhoSlotsZk`'s
slot and the next) are independent and its inner `β` (`betaSlotsZk`'s slot) is not `0`. -/
def PassAt (resp : ℕ → Claims.Msg) (spec : SessionSpec) (sts : Array Flock.Stmt) (q : ℕ) : Prop :=
  ∀ v : Schedule × ℕ, Zk.schedule sts[q / 2]!.m = .ok v →
    Zk.rhoIndependent (coinKOf (resp (repStartZk spec sts q + piopRoundsZk spec sts[q / 2]! + (v.1.k0 + 2))),
      coinKOf (resp (repStartZk spec sts q + piopRoundsZk spec sts[q / 2]! + (v.1.k0 + 2) + 1))) = true ∧
    (coinOf (resp (repStartZk spec sts q + piopRoundsZk spec sts[q / 2]! +
      (v.1.k0 + 7 + 11 * (v.1.levels.size - 1)) + 2)) 0).isZero = false

/-- **A masked session that returns passed every rep's refusals at their slots** (`PassAt`), at tables of `14 ≤ m`
and `6 ≤ k_log`, one spec table per statement. -/
theorem sessionPZk_pass (resp : ℕ → Claims.Msg) (h : ByteArray → ByteArray) (spec : SessionSpec)
    (sts : Array Flock.Stmt) (words : Array (List ℕ)) (trs : Array Trace) (tape : TapeZk)
    (hspec : spec.tables.size = sts.size) (hsts : ∀ j (hj : j < sts.size), 14 ≤ sts[j].m ∧ 6 ≤ sts[j].c.kLog)
    {r : String × Array ByteArray} {bodies : Array Claims.Msg}
    (hrun : (sessionPZk (asksS resp) (Hashes.ofSha512 h) spec (tabsOf sts) words trs tape).go #[] =
      (.ok r, bodies)) :
    ∀ q < 2 * sts.size, PassAt resp spec sts q := by
  unfold sessionPZk at hrun
  obtain ⟨schs, l1, hA, hrun⟩ := go_bind_ok hrun
  obtain ⟨hssz, hsf⟩ := Ret.go (Ret.mapM_array _ (fun st' v => Zk.schedule st'.m = .ok v) _ fun a _ =>
    Ret.monadLift (Zk.schedule a.m)) hA
  have hl1 := Still.go (Still.mapM_array (fun a l => by rw [monadLift_go]) _) hA
  subst hl1
  have htabs : (tabsOf sts).size = sts.size := tabsOf_size sts
  have htab : ∀ i (hi : i < sts.size), (tabsOf sts)[i]! = Setup.ofCircuit sts[i]! := fun i hi => by
    rw [getElem!_pos (tabsOf sts) i (by omega), getElem!_pos sts i hi]
    simp [tabsOf]
  have hv : ∀ i (hi : i < sts.size), Zk.schedule sts[i]!.m = .ok schs[i]! := fun i hi => by
    have := hsf i (by omega) (by omega)
    rw [← getElem!_pos (tabsOf sts) i (by omega), htab i hi, ← getElem!_pos schs i (by omega)] at this
    exact this
  dsimp only at hrun
  obtain ⟨cms, l2, hB, hrun⟩ := go_bind_ok hrun
  have hl2 := Still.go (Still.mapM_array (fun i => by
    exact Still.of_pureS (commitPZk_pureS _ _ _ _ _ _ _ _)) _) hB
  subst hl2
  obtain ⟨c₁, l₃, hL, hrun⟩ := go_bind_ok hrun
  rw [link_go] at hL
  simp only [Prod.mk.injEq, Except.ok.injEq, Array.size_empty] at hL
  obtain ⟨rfl, rfl⟩ := hL
  obtain ⟨⟨ss, ps⟩, l₄, hF, -⟩ := go_bind_ok hrun
  have hinv := forIn_array_idx _ _ (fun k (_ : Array (Array Round) × Array ByteArray) (l : Array Claims.Msg) =>
      l.size = repStartZk spec sts k ∧ ∀ q < k, PassAt resp spec sts q)
    ?_ (by exact ⟨by simp [repStartZk], fun q hq => absurd hq (Nat.not_lt_zero _)⟩) hF
  · intro q hq
    exact hinv.2 q (by rw [repStreams_size, hspec]; exact hq)
  · rintro k a ⟨ss, ps⟩ l res l' ha ⟨hl, hpass⟩ e
    obtain ⟨hk, rfl⟩ := Array.getElem?_eq_some_iff.1 ha
    have hk' : k < 2 * sts.size := by rw [repStreams_size, hspec] at hk; exact hk
    have hi : k / 2 < sts.size := by omega
    rw [repStreams_getElem _ hk] at e
    obtain ⟨extras, l₁, e1, e⟩ := go_bind_ok e
    rw [monadLift_go] at e1
    simp only [Prod.mk.injEq] at e1
    obtain ⟨hex, rfl⟩ := e1
    obtain ⟨ic, l₀, e0, e3⟩ := go_bind_ok e
    rw [Still.go (Still.of_pureS (innerCommitP_pureS _ _ _)) e0] at e3
    obtain ⟨⟨pf, tr'⟩, l₂, e2, e⟩ := go_bind_ok e3
    simp only [go_pure, Prod.mk.injEq, Except.ok.injEq] at e
    obtain ⟨rfl, rfl⟩ := e
    rw [htab _ hi] at e2 hex
    obtain ⟨hm, hk6⟩ := hsts (k / 2) hi
    rw [← getElem!_pos sts (k / 2) hi] at hm hk6
    obtain ⟨h25, -, hsm, hk0, -, hdim, hlv, -, -⟩ :=
      FlockSoundness.Discharge.ZkExec.zk_schedule_seams sts[k / 2]!.m schs[k / 2]!.1 schs[k / 2]!.2 (hv _ hi)
    obtain ⟨hn7, h2, -, -, -, -, -⟩ := hdim
    obtain ⟨hl₂, hrho, hb, -⟩ := repPZk_exact (sch := schs[k / 2]!.1) (hm := hm) (hn := by rw [hn7, hsm]; omega) (hk0 := hk0) (hL := h2)
      (hq := fun i hi => (hlv i hi).q_ne_zero) (h := e2)
    have hx : extras.size = spec.points * sts[k / 2]!.regions.size := by
      rw [JsonRT.extraClaims_size hex]
      simp [Nat.mul_comm]
    have hp : piopZk (Setup.ofCircuit sts[k / 2]!) extras = piopRoundsZk spec sts[k / 2]! := by
      have e6 : (Setup.ofCircuit sts[k / 2]!).fold.kLog = sts[k / 2]!.c.kLog := rfl
      have e7 : (Setup.ofCircuit sts[k / 2]!).m = sts[k / 2]!.m := rfl
      unfold piopZk piopRoundsZk
      rw [e6, e7, hx]
      omega
    refine ⟨_, rfl, ?_, fun q hq => ?_⟩
    · rw [repStart_succ, ← repRoundsT_eq (hv _ hi) hx, hl₂, hl]
      have e6 : (Setup.ofCircuit sts[k / 2]!).fold.kLog = sts[k / 2]!.c.kLog := rfl
      have e7 : (Setup.ofCircuit sts[k / 2]!).m = sts[k / 2]!.m := rfl
      unfold repRoundsT piopZk
      rw [e6, e7, hx]
      omega
    · rcases Nat.lt_succ_iff_lt_or_eq.1 hq with hq | rfl
      · exact hpass q hq
      · intro v hv'
        rw [hv _ hi] at hv'
        cases hv'
        rw [← hl, ← hp]
        refine ⟨?_, ?_⟩
        · rw [show l.size + piopZk (Setup.ofCircuit sts[q / 2]!) extras + (schs[q / 2]!.1.k0 + 2) =
              l.size + piopZk (Setup.ofCircuit sts[q / 2]!) extras + 2 + schs[q / 2]!.1.k0 by omega,
            show l.size + piopZk (Setup.ofCircuit sts[q / 2]!) extras + 2 + schs[q / 2]!.1.k0 + 1 =
              l.size + piopZk (Setup.ofCircuit sts[q / 2]!) extras + 3 + schs[q / 2]!.1.k0 by omega]
          exact hrho
        · exact hb

end FlockComplete.ZkTables
