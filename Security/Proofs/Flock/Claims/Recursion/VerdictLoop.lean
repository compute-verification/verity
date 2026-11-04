import Proofs.Flock.Claims.Recursion.VerdictRowG
import Proofs.Flock.Claims.Recursion.RecordCanon
import Proofs.Flock.Claims.Recursion.VerdictRun
import Proofs.Flock.Claims.Complete.Verdict

/-!
# Row A's step 3: the verdict on the decoded proofs, from every table's runs

`hpieces` (`VerdictParts`) asks the verdict on the decoded proofs (`verifyJPWith`) to accept at a decision of the last
step, at SHA-512. Each table's reps run (`TableFiles.Runs`, `runs_assembled`); this file runs the rest of the verdict
around them.

* **The verdict accepts when each of its checks passes** (`verifyJPWith_ok`, `FlockComplete.verifyWith_ok` without
  decoding a proof file and without S17): one statement per table, the record decodes, each table's public words,
  each table's fast100 schedule (`schedsOf`), one proof per rep stream, and each table's two reps accept with one cap.
* **At a dense inner statement** (`tables_of_dense`, `points_eq`): the decision sets up one table, which the session
  spec names, and the session's link points are the first coin's (`decodeJWith_sub`).
* **At a decision** (`DecAt.verdict_of_runs`): with row G at each table (`DecAt.rowG`), each table's runs and row L's
  pinned parameters and committed cap, each rep accepts with the committed `root_B` (`verifyRepDWith_std_of_prog`),
  so both reps' caps agree, and the verdict accepts where the session's public words are empty.
-/

namespace RecDeploy.Rows

open Flock
open FlockDeploy (Coin FinalP tabsOf pointsOf verifyJPWith verifyRepDWith jAt)
open FlockComplete (schedsOf)
open FlockSoundness.Discharge.Recursion (HClass)

/-! ## The verdict on the decoded proofs -/

/-- **The verdict on the decoded proofs accepts** when each of its checks passes. -/
theorem verifyJPWith_ok (hs : Flock.Hashes Flock.V) (spec : SessionSpec) (tabs : Array Setup) (j : Lean.Json)
    (ps : Array Proof) {sess : RunRecord}
    (hT : (tabs.size != spec.tables.size || tabs.size == 0) = false)
    (hD : Record.decodeJWith hs spec j = .ok sess)
    (hU : ∀ q ∈ (tabs.zip sess.publics).toList, q.1.publics q.2 = .ok ())
    (hF : ∀ st ∈ tabs.toList, (fast100 st.m).isSome)
    (hN : (ps.size != spec.repStreams.size) = false)
    (hR : ∀ i < tabs.size, ∃ c,
      verifyRepDWith hs spec tabs[i]! (schedsOf tabs)[i]! sess i 0 ps[2 * i]! = .ok c ∧
      verifyRepDWith hs spec tabs[i]! (schedsOf tabs)[i]! sess i 1 ps[2 * i + 1]! = .ok c) :
    verifyJPWith hs spec tabs j ps = .ok () := by
  unfold verifyJPWith
  have hpub : forIn (tabs.zip sess.publics) PUnit.unit (fun x _ => do
      liftM (x.1.publics x.2)
      pure (ForInStep.yield PUnit.unit) : _ → _ → Flock.V _) = Except.ok () := by
    rw [← Array.forIn_toList]
    refine FlockComplete.forIn_ok _ _ fun a ha => ?_
    simp only [liftM, monadLift_self, hU a ha]
    rfl
  have hreps : forIn [:tabs.size] PUnit.unit (fun i _ => do
      let c0 ← verifyRepDWith hs spec tabs[i]! (schedsOf tabs)[i]! sess i 0 ps[2 * i]!
      let c1 ← verifyRepDWith hs spec tabs[i]! (schedsOf tabs)[i]! sess i 1 ps[2 * i + 1]!
      if (c0 != c1) = true then do
        throw (toString "S18/R1: " ++ toString spec.tables[i]! ++ toString ": the reps' caps differ")
        pure (ForInStep.yield PUnit.unit)
      else pure (ForInStep.yield PUnit.unit) : _ → _ → Flock.V _) = Except.ok () := by
    rw [Std.Legacy.Range.forIn_eq_forIn_range']
    refine FlockComplete.forIn_ok _ _ fun i hi => ?_
    have hi' : i < tabs.size := by
      simp only [Std.Legacy.Range.size, Nat.sub_zero, Nat.add_one_sub_one, Nat.div_one, List.mem_range'] at hi
      omega
    obtain ⟨c, h0, h1⟩ := hR i hi'
    have hcc : (c != c) = false := by
      show (!(c.data == c.data)) = false
      simp
    simp only [h0, h1, FlockComplete.ok_bind, hcc, Bool.false_eq_true, ite_false]
    rfl
  simp only [hT, Bool.false_eq_true, ite_false, hD, FlockComplete.ok_bind, hpub, hN]
  rw [FlockComplete.array_mapM_ok _ (fun st => (fast100 st.m).getD default) tabs ?_]
  · rw [show Array.map (fun st => (fast100 st.m).getD default) tabs = schedsOf tabs from rfl, FlockComplete.ok_bind,
      hreps]
    rfl
  · intro st hst
    have := hF st hst
    split
    · rename_i sch hf
      rw [hf]
      rfl
    · rename_i hf
      rw [hf] at this
      cases this

/-! ## At a decision -/

namespace DecAt

variable {rd : Claims.Msg → Option FinalP} {x : Claims.Msg} {ms : List Claims.Msg} {cs : List Coin}
  (D : DecAt rd x ms cs)

/-- **A dense inner statement's decision sets up one table**, and the session spec names one. -/
theorem tables_of_dense {xo : Claims.Msg} {sto : Claims.Recursion.CompiledStatement} (hst : parse xo = some sto)
    (hf : Claims.Recursion.Form.Dense sto.form) (hx : x = xIOf xo) :
    D.sts.size = 1 ∧ D.spec.tables.size = 1 := by
  obtain ⟨s, spec, st, sch, hset, htags, hpi, -⟩ := dense_inner hst hf
  rw [← hx] at hset
  obtain ⟨-, hsts⟩ := D.setup_eq
  obtain ⟨v, c₁, r₀, rss, fl, link, pf, hv, -, -, -, -, -, hd⟩ := FlockDeploy.decR_some D.dec
  rw [hset] at hv
  cases hv
  have hds : D.d.s = s := congrArg (·.s) hd
  have hdsts : D.d.sts = #[st] := congrArg (·.sts) hd
  have hsts1 : D.sts = #[st] := hsts.trans hdsts
  have hT : D.d.s.read.tags = RecVStar.innerTags ∧ D.d.s.read.publicInputs = none := by
    rw [hds]; exact ⟨htags, hpi⟩
  have h := D.read_setup
  unfold Readable.Read.setup at h
  rw [hT.1, hT.2] at h
  have h1 : D.sts.size = 1 := by rw [hsts1]; rfl
  refine ⟨h1, ?_⟩
  rcases FlockSoundness.Discharge.HiddenExec.setupTables_okH
      FlockSoundness.Discharge.Recursion.innerTags_hiddenOutputs h with
    ⟨-, st', -, hsp, -⟩ | ⟨hc, hsz, -⟩
  · rw [hsp]
    rfl
  · omega

/-- **The session's link points are the first coin's** (`decodeJWith_sub`). -/
theorem points_eq : D.sess.points = pointsOf D.spec D.d.c₁ := by
  have h := D.decode
  unfold FlockDeploy.jAt at h
  exact (FlockDeploy.decodeJWith_sub _ h).1

theorem schedsOf_getElem {tabs : Array Setup} {i : ℕ} (hi : i < tabs.size) :
    (schedsOf tabs)[i]! = (fast100 tabs[i]!.m).getD default := by
  unfold schedsOf
  rw [getElem!_pos _ i (by simpa using hi), Array.getElem_map, getElem!_pos _ i hi]

variable (cls : HClass) (acc : ℕ → Array F128)

/-- **Rep `r` of table `i` accepts with the committed `root_B`**, where its program runs and row L's parameters and
cap hold. -/
theorem rep_ok {i : ℕ} (hG : D.RowG cls acc i) (hti : i < D.spec.tables.size) (hrun : (D.files cls i).Runs)
    (hl : ∀ r, D.RowL cls acc i r) (r : Fin 2) :
    verifyRepDWith With.std D.spec (D.files cls i).st (D.files cls i).sch D.sess i r ((D.files cls i).proof r) =
      .ok D.sess.rootB[i]! := by
  obtain ⟨t₄, hp, hk⟩ := hrun r r.isLt
  have e := verifyRepDWith_std_of_prog D.spec (D.files cls i).st hG.named (D.files cls i).sch D.sess i r
    ((D.files cls i).stream r) (D.files cls i).extras ((D.files cls i).proof r) t₄ (D.finds cls hti r) hG.extra
    (hl r).params hp hk (hl r).cap
  rw [FlockSoundness.Refine.bytes_eq_of_beq (hl r).cap] at e
  exact e

/-- **Row A's step 3 at a decision**: with row G at each table, each table's runs and row L, the verdict on the decoded
proofs accepts where the session spec has one table per statement and the session's public words are empty. -/
theorem verdict_of_runs (hT : D.sts.size = D.spec.tables.size) (h0 : D.sts.size ≠ 0)
    (hU : (D.sess.publics.all fun p => p.size == 0) = true)
    (hG : ∀ i < D.sts.size, D.RowG cls acc i)
    (hrun : ∀ i < D.sts.size, (D.files cls i).Runs ∧ ∀ r, D.RowL cls acc i r) :
    verifyJPWith (Flock.Hashes.ofSha512 (Claims.hOf FlockDeploy.sha512) : Flock.Hashes Flock.V) D.spec (tabsOf D.sts)
      (jAt FlockDeploy.sha512 D.spec D.sts D.d.c₁ D.d.mids D.d.f) D.d.f.ps = .ok () := by
  have hsz : (tabsOf D.sts).size = D.sts.size := Array.size_map ..
  refine verifyJPWith_ok _ _ _ _ _ (sess := D.sess) ?_ D.decode ?_ ?_ ?_ ?_
  · rw [hsz, ← hT]
    simp [h0]
  · intro q hq
    obtain ⟨a, b⟩ := q
    obtain ⟨ha, hb⟩ := Array.of_mem_zip (Array.mem_toList_iff.1 hq)
    have ha' : a ∈ D.sts.map Setup.ofCircuit := ha
    obtain ⟨st, -, rfl⟩ := Array.mem_map.1 ha'
    have hp : b.size = 0 := by
      simpa using Array.all_eq_true_iff_forall_mem.1 hU b hb
    simp only [Setup.ofCircuit, hp, beq_self_eq_true, ↓reduceIte]
    rfl
  · intro st hst
    obtain ⟨i, hi, rfl⟩ := Array.mem_iff_getElem.1 (Array.mem_toList_iff.1 hst)
    have hf := (hG i (hsz ▸ hi)).fast
    change fast100 (tabsOf D.sts)[i]!.m = _ at hf
    rw [getElem!_pos (tabsOf D.sts) i hi] at hf
    rw [hf]
    rfl
  · obtain ⟨-, -, -, -, hps, -⟩ := FlockDeploy.decR_fits D.dec
    rw [hps, Claims.Bridge.Custody.repStreams_size, ← (D.setup_eq).2, ← hT]
    simp
  · intro i hi
    have hi' : i < D.sts.size := hsz ▸ hi
    have hti : i < D.spec.tables.size := hT ▸ hi'
    obtain ⟨hr, hl⟩ := hrun i hi'
    rw [schedsOf_getElem hi, FlockDeploy.hOf_sha512]
    exact ⟨D.sess.rootB[i]!, D.rep_ok cls acc (hG i hi') hti hr hl 0, D.rep_ok cls acc (hG i hi') hti hr hl 1⟩

end DecAt

end RecDeploy.Rows
