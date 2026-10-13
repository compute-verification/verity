import Proofs.Flock.Recursive.HoloSched

/-!
# Where `fixedRow` reads, by the schedule

`fixedRow_valueRound`: where the verifier holds row `x` of a value `RecVStar.fixedRow` names, `valueRound` names the
stream and round of the `c_k` (or top) it read, at streams whose held `c_k` sit at their schedules' message rounds
(`CommitsAt`) and, for R, whose held tops are its schedule's (`TopsAt`). The verifier checks both of R's stream round by
round (`checkStackedShape` holds each round's `schedOf`, its tops' sizes and whether it commits, to `stackedRounds`'), and
neither of the reps' or the sparse phases': `innerCommits` holds the reps' rounds to the record's coins, and `sparseRow`
checks only a sparse stream's count of `c_k`.
-/

namespace Flock.Guarantees

open Flock Flock.RecVStar

/-- **Hypothesis on a stream's held `c_k`: they sit at its schedule's message rounds**, in order: the `i`-th's round is
`rs`' `i`-th message round. -/
def CommitsAt (cm : Array (Nat × ByteArray)) (rs : Array SchedRound) : Prop :=
  cm.toList.map (·.1) = msgRounds rs

/-- **Hypothesis on R's held tops: they are the tops its schedule commits**, in order, each of its node count. -/
def TopsAt (tops : Array (Array ByteArray)) (rs : Array SchedRound) : Prop :=
  tops.toList.map (·.size) = (topRounds rs).map (·.2)

/-- The `c_k` the verifier holds of stream `s`, each with its round: a rep's from the prover's record
(`InnerCommits.reps`), a sparse phase's and R's from the coin server's (`InnerCommits.streams`). -/
def heldCommits (cs : InnerCommits) : HoloStream → Option (Array (Nat × ByteArray))
  | .rep x => cs.reps[x.val]?
  | .sparse x => (cs.stream? s!"{SPARSE}/rep{x.val}").map (·.commits)
  | .stacked => (cs.stream? s!"{STACKED}/rep0").map (·.commits)

/-- `c` is what the verifier holds at stream `s`'s round `k`, node `x` where the round commits a top: a `c_k` it holds at
round `k`, or node `x` of R's `l`-th top, which R's round `k` commits. -/
def HeldAt (cs : InnerCommits) (stacked : Option (Array SchedRound)) (s : HoloStream) (k x : Nat) (c : ByteArray) :
    Prop :=
  (∃ cm i, heldCommits cs s = some cm ∧ cm[i]? = some (k, c)) ∨
    ∃ rs st l n, s = .stacked ∧ stacked = some rs ∧ cs.stream? s!"{STACKED}/rep0" = some st ∧
      (topRounds rs)[l]? = some (k, n) ∧ (st.tops[l]?).bind (·[x]?) = some c

theorem msgRounds_getElem? {cm : Array (Nat × ByteArray)} {rs : Array SchedRound} (h : CommitsAt cm rs) (i : Nat) :
    (msgRounds rs)[i]? = (cm[i]?).map (·.1) := by
  unfold CommitsAt at h
  rw [← h, List.getElem?_map, Array.getElem?_toList]

theorem topRounds_getElem? {tops : Array (Array ByteArray)} {rs : Array SchedRound} (h : TopsAt tops rs) (l : Nat) :
    ((topRounds rs)[l]?).map (·.2) = (tops[l]?).map (·.size) := by
  unfold TopsAt at h
  rw [← List.getElem?_map, ← h, List.getElem?_map, Array.getElem?_toList]

theorem heldCommits_sparse {cs : InnerCommits} {x : Nat} (hx : x < 2) {st : InnerStream}
    (h : cs.stream? s!"{SPARSE}/rep{x}" = some st) : heldCommits cs (.sparse ⟨x, hx⟩) = some st.commits := by
  show (cs.stream? s!"{SPARSE}/rep{x}").map (·.commits) = some st.commits
  exact congrArg (Option.map (·.commits)) h

theorem heldCommits_stacked {cs : InnerCommits} {st : InnerStream} (h : cs.stream? s!"{STACKED}/rep0" = some st) :
    heldCommits cs .stacked = some st.commits := by
  show (cs.stream? s!"{STACKED}/rep0").map (·.commits) = some st.commits
  exact congrArg (Option.map (·.commits)) h

theorem claim_or_sparse (cs : InnerCommits) (sh : RecSparse.Shape) (p : String) (x : Nat) :
    (if p == "claim" then claimRow cs sh x else sparseRow cs sh p x) = sparseRow cs sh p x := by
  split
  next hp =>
    have hp' := eq_of_beq hp
    subst hp'
    rfl
  next => rfl

/-- **What `sparseRow` reads**: the `i`-th `c_k` `sparse/rep<x>` holds, `i` `p`'s place in `sparseMessages`. -/
theorem sparseRow_ok {cs : InnerCommits} {sh : RecSparse.Shape} {p : String} {x : Nat} {d : String} {c : ByteArray}
    (h : sparseRow cs sh p x = .ok (d, c)) :
    ∃ i st k, (sparseMessages sh).idxOf? p = some i ∧ cs.stream? s!"{SPARSE}/rep{x}" = some st ∧
      st.commits[i]? = some (k, c) := by
  cases hi : (sparseMessages sh).idxOf? p with
  | none => simp [sparseRow, hi] at h
  | some i =>
    cases hs : cs.stream? s!"{SPARSE}/rep{x}" with
    | none => simp [sparseRow, hi, hs] at h
    | some st =>
      obtain ⟨hil, -⟩ := List.idxOf?_eq_some_iff.1 hi
      by_cases hn : st.commits.size = (sparseMessages sh).length
      · have hix : i < st.commits.size := by omega
        cases hk : st.commits[i]'hix with
        | mk k c' =>
          have hc : c' = c := by
            simp [sparseRow, hi, hs, hn, getElem!_pos st.commits i hix, hk, bind, Except.bind, pure,
              Except.pure] at h
            exact h.2
          subst hc
          refine ⟨i, st, k, by first | rfl | exact hi, by first | rfl | exact hs, ?_⟩
          rw [Array.getElem?_eq_getElem hix, hk]
      · simp [sparseRow, hi, hs, hn, bind, Except.bind] at h

theorem fixedRow_sparse_eq {cs : InnerCommits} {sh : RecSparse.Shape} {v p : String} {x : Nat}
    (hci : cIndex v = none) (hoi : openingIndex v = none) (hti : stackedTopLevel v = none)
    (hpi : sparsePort v = some p) : fixedRow cs v x (some sh) = some (sparseRow cs sh p x) := by
  simp only [fixedRow, hci, hoi, hti, hpi]
  rw [claim_or_sparse]

section cases

variable {cs : InnerCommits} {rep : Array SchedRound} {sparse : Option RecSparse.Shape}
  {stacked : Option (Array SchedRound)} {v : String} {x : Nat} {d : String} {c : ByteArray}

theorem fixedRow_repRound {i : Nat} (hci : cIndex v = some i)
    (hrep : ∀ y cm, cs.reps[y]? = some cm → y < 2 ∧ CommitsAt cm rep)
    (h : fixedRow cs v x sparse = some (.ok (d, c))) :
    ∃ s k, valueRound rep sparse stacked v x = some (s, k) ∧ HeldAt cs stacked s k x c := by
  cases hr : cs.reps[x]? with
  | none => simp [fixedRow, hci, hr] at h
  | some cm =>
    cases hk : cm[i]? with
    | none => simp [fixedRow, hci, hr, hk] at h
    | some kc =>
      obtain ⟨k, c'⟩ := kc
      have hc : c' = c := by
        simp [fixedRow, hci, hr, hk] at h
        exact h.2
      subst hc
      obtain ⟨hx, hcm⟩ := hrep x cm hr
      refine ⟨.rep ⟨x, hx⟩, k, ?_, .inl ⟨cm, i, hr, hk⟩⟩
      simp [valueRound, hci, hx, msgRounds_getElem? hcm, hk]

theorem fixedRow_openingRound {i : Nat} (hci : cIndex v = none) (hoi : openingIndex v = some i)
    (hst : ∀ st, cs.stream? s!"{STACKED}/rep0" = some st →
      ∃ rs, stacked = some rs ∧ CommitsAt st.commits rs ∧ TopsAt st.tops rs)
    (h : fixedRow cs v x sparse = some (.ok (d, c))) :
    ∃ s k, valueRound rep sparse stacked v x = some (s, k) ∧ HeldAt cs stacked s k x c := by
  by_cases hx : x = 0
  · subst hx
    cases hs : cs.stream? s!"{STACKED}/rep0" with
    | none => simp [fixedRow, hci, hoi, hs] at h
    | some st =>
      cases hk : st.commits[i]? with
      | none => simp [fixedRow, hci, hoi, hs, hk] at h
      | some kc =>
        obtain ⟨k, c'⟩ := kc
        have hc : c' = c := by
          simp [fixedRow, hci, hoi, hs, hk] at h
          exact h.2
        subst hc
        obtain ⟨rs, hrs, hcm, -⟩ := hst st hs
        refine ⟨.stacked, k, ?_, .inl ⟨st.commits, i, heldCommits_stacked hs, hk⟩⟩
        simp [valueRound, hci, hoi, hrs, msgRounds_getElem? hcm, hk]
  · simp [fixedRow, hci, hoi, hx] at h

theorem fixedRow_topRound {l : Nat} (hci : cIndex v = none) (hoi : openingIndex v = none)
    (hti : stackedTopLevel v = some l)
    (hst : ∀ st, cs.stream? s!"{STACKED}/rep0" = some st →
      ∃ rs, stacked = some rs ∧ CommitsAt st.commits rs ∧ TopsAt st.tops rs)
    (h : fixedRow cs v x sparse = some (.ok (d, c))) :
    ∃ s k, valueRound rep sparse stacked v x = some (s, k) ∧ HeldAt cs stacked s k x c := by
  cases hs : cs.stream? s!"{STACKED}/rep0" with
  | none => simp [fixedRow, hci, hoi, hti, hs] at h
  | some st =>
    cases ht : st.tops[l]? with
    | none => simp [fixedRow, hci, hoi, hti, hs, ht] at h
    | some top =>
      cases hn : top[x]? with
      | none => simp [fixedRow, hci, hoi, hti, hs, ht, hn] at h
      | some c' =>
        have hc : c' = c := by
          simp [fixedRow, hci, hoi, hti, hs, ht, hn] at h
          exact h.2
        subst hc
        obtain ⟨rs, hrs, -, htp⟩ := hst st hs
        obtain ⟨hx, -⟩ := Array.getElem?_eq_some_iff.1 hn
        have e := topRounds_getElem? htp l
        rw [ht] at e
        cases hk : (topRounds rs)[l]? with
        | none =>
          rw [hk] at e
          simp at e
        | some kn =>
          obtain ⟨k, n⟩ := kn
          rw [hk] at e
          have hn' : n = top.size := Option.some.inj e
          subst hn'
          refine ⟨.stacked, k, ?_, .inr ⟨rs, st, l, top.size, rfl, hrs, hs, hk, ?_⟩⟩
          · simp [valueRound, hci, hoi, hti, hrs, hk, hx]
          · simp [ht, hn]

theorem fixedRow_sparseRound {p : String} (hci : cIndex v = none) (hoi : openingIndex v = none)
    (hti : stackedTopLevel v = none) (hpi : sparsePort v = some p)
    (hsp : ∀ y st sh, cs.stream? s!"{SPARSE}/rep{y}" = some st → sparse = some sh →
      y < 2 ∧ CommitsAt st.commits (sparseRounds sh))
    (h : fixedRow cs v x sparse = some (.ok (d, c))) :
    ∃ s k, valueRound rep sparse stacked v x = some (s, k) ∧ HeldAt cs stacked s k x c := by
  cases sparse with
  | none => simp [fixedRow, hci, hoi, hti, hpi] at h
  | some sh =>
    rw [fixedRow_sparse_eq hci hoi hti hpi] at h
    obtain ⟨i, st, k, hi, hs, hk⟩ := sparseRow_ok (Option.some.inj h)
    obtain ⟨hx, hcm⟩ := hsp x st sh hs rfl
    refine ⟨.sparse ⟨x, hx⟩, k, ?_, .inl ⟨st.commits, i, heldCommits_sparse hx hs, hk⟩⟩
    simp [valueRound, hci, hoi, hti, hpi, hx, hi, msgRounds_getElem? hcm, hk]

end cases

/-- **Where the verifier holds row `x` of a value `fixedRow` names, `valueRound` names the round it read**: at streams
whose held `c_k` sit at their schedules' message rounds (the reps' at `rep`, the sparse phases' at `sparseRounds` of the
verifier's RecSparse shape, R's at `stacked`) and whose held tops are R's schedule's, the `b ‖ c` `c` the verifier holds
there is what it holds at the stream and round `valueRound` names (`HeldAt`): the `c_k` it holds at that round, or node
`x` of the top R commits there. -/
theorem fixedRow_valueRound {cs : InnerCommits} {rep : Array SchedRound} {sparse : Option RecSparse.Shape}
    {stacked : Option (Array SchedRound)} {v : String} {x : Nat} {d : String} {c : ByteArray}
    (hrep : ∀ y cm, cs.reps[y]? = some cm → y < 2 ∧ CommitsAt cm rep)
    (hsp : ∀ y st sh, cs.stream? s!"{SPARSE}/rep{y}" = some st → sparse = some sh →
      y < 2 ∧ CommitsAt st.commits (sparseRounds sh))
    (hst : ∀ st, cs.stream? s!"{STACKED}/rep0" = some st →
      ∃ rs, stacked = some rs ∧ CommitsAt st.commits rs ∧ TopsAt st.tops rs)
    (h : fixedRow cs v x sparse = some (.ok (d, c))) :
    ∃ s k, valueRound rep sparse stacked v x = some (s, k) ∧ HeldAt cs stacked s k x c := by
  cases hci : cIndex v with
  | some i => exact fixedRow_repRound hci hrep h
  | none =>
    cases hoi : openingIndex v with
    | some i => exact fixedRow_openingRound hci hoi hst h
    | none =>
      cases hti : stackedTopLevel v with
      | some l => exact fixedRow_topRound hci hoi hti hst h
      | none =>
        cases hpi : sparsePort v with
        | some p => exact fixedRow_sparseRound hci hoi hti hpi hsp h
        | none => simp [fixedRow, hci, hoi, hti, hpi] at h

#print axioms msgRounds_getElem?
#print axioms topRounds_getElem?
#print axioms heldCommits_sparse
#print axioms heldCommits_stacked
#print axioms claim_or_sparse
#print axioms sparseRow_ok
#print axioms fixedRow_sparse_eq
#print axioms fixedRow_repRound
#print axioms fixedRow_openingRound
#print axioms fixedRow_topRound
#print axioms fixedRow_sparseRound
#print axioms fixedRow_valueRound

end Flock.Guarantees
