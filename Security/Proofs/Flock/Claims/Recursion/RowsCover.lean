import Proofs.Flock.Claims.Recursion.LayoutBTops
import Proofs.Flock.C3.Reps.RoundLinks

/-!
# Every absorbing stream round is a part's row (`RowsCover`), from the verifier's checks

`RowsCover` (`LayoutBCap.lean`) asks, at every compiled statement, that each stream round absorbing a message be held
by some part through a port under its `rec-c<k>` (`heldRows`). Two checks give it at every shape:

* **V\*'s list** (`RecVStar.checkRoundsListed`, in `RecVStar.statements`): each `k`-th message round of the inner
  shape's rep schedule is a message-round port of one of the list's statements, `rec-c<k>` at both reps' rows
  (`RecVStar.roundListed`). A list without one is refused.
* **The parts' circuits** (`portsNamed`, in `readX`): part `s`'s circuit has a port of each name its statement
  registers under a `rec-c<k>`. Each session refuses a part without (`checkOwn`, `checkPubRows`).

`rowsCover`: `RowsCover` at every shape. `ksLin2_deployed_rows` is `ksLin2_deployed_tops` with it discharged.
-/

namespace RecDeploy

open Flock
open Claims (Msg)
open FlockDeploy (Coin)

/-- **V\*'s list holds every message round a rep absorbs** wherever it builds (`RecVStar.checkRoundsListed`). -/
theorem roundsListed_of_statements {i : RecVStar.Inner} {ss : Array VStar.Stmt}
    (h : RecVStar.statements i = .ok ss) : RecVStar.checkRoundsListed i ss = .ok () := by
  unfold RecVStar.statements at h
  dsimp only at h
  split at h
  rotate_left
  · cases h
  split at h
  · cases h
  obtain ⟨ps, -, h⟩ := FlockC3.VStarList.eok_bind h
  obtain ⟨gs, -, h⟩ := FlockC3.VStarList.eok_bind h
  split at h
  · cases h
  split at h
  · cases h
  obtain ⟨K, -, h⟩ := FlockC3.VStarList.eok_bind h
  obtain ⟨t, -, h⟩ := FlockC3.VStarList.eok_bind h
  obtain ⟨A, -, h⟩ := FlockC3.VStarList.eok_bind h
  obtain ⟨C, -, h⟩ := FlockC3.VStarList.eok_bind h
  obtain ⟨K0, -, h⟩ := FlockC3.VStarList.eok_bind h
  obtain ⟨u, -, h⟩ := FlockC3.VStarList.eok_bind h
  obtain ⟨u', -, h⟩ := FlockC3.VStarList.eok_bind h
  obtain ⟨u'', hu, h⟩ := FlockC3.VStarList.eok_bind h
  cases h
  cases u''
  exact hu

/-- **Each rep's `k`-th message round is listed** where `checkRoundsListed` passes, at the shape's rep schedule. -/
theorem roundListed_of_check {i : RecVStar.Inner} {ss : Array VStar.Stmt} {rs : Array RecVStar.SchedRound}
    (hrs : repSched i = some rs) (h : RecVStar.checkRoundsListed i ss = .ok ()) {k : ℕ}
    (hk : k < (RecVStar.commitRounds rs).length) : RecVStar.roundListed ss k = true := by
  unfold repSched at hrs
  unfold RecVStar.checkRoundsListed at h
  simp only [hrs] at h
  cases hf : (List.range (RecVStar.commitRounds rs).length).find? (!RecVStar.roundListed ss ·) with
  | some k' =>
    rw [hf] at h
    cases h
  | none =>
    simpa using List.find?_eq_none.1 hf k (List.mem_range.2 hk)

/-- What `readX` read of V*'s list and the parts: the list builds at the verifier's own inner shape, the parts number
as its statements, each part's circuit names its statement's message-round ports (`portsNamed`), and the last part's
circuit has RecCap's port `top` (`capPorted`). -/
theorem readX_listed {x : Msg} {r : XR} (h : readX x = some r) :
    ∃ vs, RecVStar.statements r.shape = .ok vs ∧ r.w.parts.length = vs.size ∧ portsNamed r.w.parts vs = true ∧
      capPorted r.w.parts = true := by
  replace h := readXPre_of_readX h
  unfold readXPre at h
  simp only [Option.bind_eq_bind, Option.bind_eq_some_iff] at h
  obtain ⟨w, -, ri, -, sh, -, pa, -, dims, -, vs, hvs, h⟩ := h
  split_ifs at h with hl
  · simp at h
  have h' : (⟨w, ri, pa, sh, dims⟩ : XR) = r := by simpa using h
  subst h'
  obtain ⟨h1, h2⟩ := not_or.1 hl
  obtain ⟨h2, h3⟩ := not_or.1 h2
  exact ⟨vs, toOption_eq_some hvs, not_not.1 h1, by simpa using h2, by simpa using h3⟩

/-- **A part's circuit has each port its statement registers under a `rec-c<k>`**, where `portsNamed` passes. -/
theorem port_of_portsNamed {parts : List PartX} {vs : Array VStar.Stmt} (h : portsNamed parts vs = true) {s : ℕ}
    (hs : s < vs.size) {q : String × String × Array (Option ℕ)} (hq : q ∈ RecVStar.ownPorts vs[s])
    {k : ℕ} (hk : RecVStar.cIndex q.2.1 = some k) :
    ∃ c, parts[s]?.bind partCircuit = some c ∧ ∃ qi, c.ports.findIdx? (·.name == q.1) = some qi := by
  have hall := List.all_eq_true.1 h s (List.mem_range.2 hs)
  rw [getElem!_pos vs s hs] at hall
  have hneed : q ∈ (RecVStar.ownPorts vs[s]).filter fun q => (RecVStar.cIndex q.2.1).isSome :=
    List.mem_filter.2 ⟨hq, by simp [hk]⟩
  cases hc : parts[s]?.bind partCircuit with
  | none =>
    simp only [hc] at hall
    rw [List.isEmpty_iff] at hall; rw [hall] at hneed; cases hneed
  | some c =>
    simp only [hc] at hall
    have hany := List.all_eq_true.1 hall q hneed
    refine ⟨c, rfl, ?_⟩
    rw [← Option.isSome_iff_exists]
    rw [Array.findIdx?_isSome]; exact hany

/-- **`RowsCover` at every shape**: a stream round `j` that absorbs a message is rep `j / R`'s round `j % R` (`R` the
rep schedule's rounds), its `k`-th message round for some `k` (`RecVStar.commitRounds`). V*'s list registers `rec-c<k>`
at both reps' rows in some statement `s` (`roundsListed_of_statements`), and part `s`'s circuit has that port
(`readX_listed`), so part `s` holds rep `j / R`'s row of round `j` (`heldRows`). -/
theorem rowsCover (shape : Claims.Recursion.Form → Prop) : RowsCover shape := by
  intro x st _ _ r j sr hr hsr _ hA
  obtain ⟨vs, hvs, hlen, hpn, -⟩ := readX_listed hr
  unfold streamRound at hsr
  rw [hr, Option.bind_some] at hsr
  cases hrs : repSched r.shape with
  | none =>
    rw [hrs] at hsr
    cases hsr
  | some rs =>
    rw [hrs, Option.bind_some] at hsr
    by_cases hj : j < 2 * rs.size
    swap
    · rw [if_neg hj] at hsr
      cases hsr
    rw [if_pos hj] at hsr
    obtain rfl := Option.some.inj hsr
    have hpos : 0 < rs.size := by omega
    have ht : j % rs.size ∈ RecVStar.commitRounds rs := by
      simp only [RecVStar.commitRounds, List.mem_filter, List.mem_range]
      exact ⟨Nat.mod_lt _ hpos, by simpa using hA⟩
    obtain ⟨k, hk⟩ := List.mem_iff_getElem?.1 ht
    have hkl : k < (RecVStar.commitRounds rs).length := by
      by_contra hc
      have h0 : (RecVStar.commitRounds rs)[k]? = none := List.getElem?_eq_none_iff.2 (by omega)
      rw [h0] at hk
      cases hk
    have hl := roundListed_of_check hrs (roundsListed_of_statements hvs) hkl
    unfold RecVStar.roundListed at hl
    obtain ⟨s, hs, hl⟩ := Array.any_eq_true.1 hl
    obtain ⟨q, hq, hql⟩ := List.any_eq_true.1 hl
    simp only [Bool.and_eq_true] at hql
    have hqv : q.2.1 = s!"rec-c{k}" := beq_iff_eq.1 hql.1
    have hqr : q.2.2 = #[some 0, some 1] := beq_iff_eq.1 hql.2
    have hck : RecVStar.cIndex q.2.1 = some k := by rw [hqv]; exact FlockC3.Reps.cIndex_c k
    obtain ⟨c, hc, qi, hqi⟩ := port_of_portsNamed hpn hs hq hck
    have hsv : ((RecVStar.statements r.shape).toOption.bind fun vs => vs[s]?) = some vs[s] := by
      rw [hvs]
      exact Array.getElem?_eq_getElem hs
    have hrep : j / rs.size = 0 ∨ j / rs.size = 1 := by
      have : j / rs.size < 2 := (Nat.div_lt_iff_lt_mul hpos).2 (by omega)
      generalize j / rs.size = d at this ⊢
      omega
    have hrow : some (j / rs.size) ∈ q.2.2.toList := by
      rw [hqr]
      rcases hrep with h | h <;> simp [h]
    have hmr : msgRound rs (j / rs.size) k = some j := by
      unfold msgRound
      rw [hk]
      show some (j / rs.size * rs.size + j % rs.size) = some j
      rw [Nat.div_add_mod']
    refine ⟨s, by simp only [nS, hr]; omega, (j, q.1, qi, j / rs.size), ?_, rfl⟩
    unfold heldRows
    rw [hsv, hc, hrs]
    refine List.mem_flatMap.2 ⟨q, hq, ?_⟩
    simp only [hck]
    exact List.mem_filterMap.2 ⟨some (j / rs.size), hrow, by simp [hmr, hqi]⟩

/-- **`KSLin2` at the deployment's dense statements, at `layoutBTops`, with the rows' ports discharged**:
`ksLin2_deployed_tops` with `RowsCover` from `rowsCover`. What stays at the dense form is the schedule (`SchedMatches`),
RecCap's port (`CapPorts`), the last round's pieces (`FinPieces`, which holds at `finOpen`: `finPieces_finOpen`), and
`VstarShowsB`, `PartChecksBV`, `PartOpensB` and `DecompB`; then `RunPairs` and the readers' three facts. This bounds no
number by itself: those premises are open.
-/
theorem ksLin2_deployed_rows (b : Brings) (fin : Msg → List (ℕ × List (Msg × List HmNets.Lin))) (wE : Msg → ℝ)
    (join : List Msg → Msg) (final : Msg → Option (List (List (Msg × Salt)) × Msg)) (w : Msg → ℝ) :
    let L := layoutBTops fin
    let Bd := bdDeployed _root_.Flock.Readable.relation (FlockDeploy.V false).next
      (FlockDeploy.deployPlainTrim _root_.Flock.Readable.relation).cols wE join rowInDeployed final w
    let D := recBound _root_.Flock.Readable.relation b L Bd.IVStar
    let bL : Brings := { b with rd := L.readers }
    (∃ decC : Msg → Msg → Coin,
      (∀ decI, Claims.Recursion.VStar.RunPairs FlockDeploy.sha512 D
        (Claims.Recursion.Inner.ofFlock Bd.deploy decI) decC) ∧
      ∀ shape ∈ [Claims.Recursion.Form.Dense],
        SchedMatches shape ∧ CapPorts shape ∧ FinPieces fin shape ∧
        ∃ (chk : ℕ → Msg → (ℕ → Coin) → List Msg → Prop) (shows : ℕ → Msg → Msg → (ℕ → Coin) → Prop)
          (cmt : Msg → ℕ → Msg → Msg),
          (∀ decI, Claims.Recursion.VStarB.VstarShowsB D (Claims.Recursion.Inner.ofFlock Bd.deploy decI) L shape
            decC shows cmt) ∧
          (∀ s, Claims.Recursion.VStarB.PartChecksBV FlockDeploy.sha512 D (deployZkRec _root_.Flock.Readable.relation)
            L shape shows cmt chk s) ∧
          (∀ s, Claims.Recursion.VStarB.PartOpensB FlockDeploy.sha512 D (deployZkRec _root_.Flock.Readable.relation)
            L shape shows cmt s) ∧
          Claims.Recursion.VStarB.DecompB FlockDeploy.sha512 D Bd.IVStar L shape chk) →
    ReadersTrue FlockDeploy.sha512 Bd.IVStar bL (deployZkRec _root_.Flock.Readable.relation) →
    ReadersRunPlayed FlockDeploy.sha512 Bd.IVStar bL (deployZkRec _root_.Flock.Readable.relation) →
    Claims.Recursion.KSLin2On FlockDeploy.sha512 (recBoundAt _root_.Flock.Readable.relation b L) Bd
      (deployZkRec _root_.Flock.Readable.relation) FlockDeploy.KS.CpDep DenseAt := by
  intro L Bd D bL hP hRT hRP
  refine ksLin2_deployed_tops b fin wE join final w ?_ hRT hRP
  obtain ⟨decC, hRun, hS⟩ := hP
  refine ⟨decC, hRun, fun shape hsh => ?_⟩
  obtain ⟨hM, hC, hF, rest⟩ := hS shape hsh
  exact ⟨hM, hC, rowsCover shape, hF, rest⟩

end RecDeploy
