module

public import Proofs.Flock.Recursive.ZkServerTemplate
public import Proofs.Flock.Recursive.ZkServerLive
import all Lean.Data.Json.Basic

/-! The live zk_server at any requests sends the pinned schedule's transcript, its commit strings' digests and its stop read from the requests alone. -/

@[expose] public section

namespace FlockSoundness.Discharge.Recursion.ZkServerAny

open Flock Flock.ZkServer ZkServerRun ZkServerTemplate ZkServerLive

def liveOut (s : Schedule) : State → List (ByteArray × List ByteArray) → List Sent × Option State
  | st, [] => ([], some st)
  | st, (b, fresh) :: rest =>
    match liveStep s st b fresh with
    | .ok (sent, _, st') => (sent ++ (liveOut s st' rest).1, (liveOut s st' rest).2)
    | .error _ => ([], none)

def liveFinished (s : Schedule) (L : List (ByteArray × List ByteArray)) : Bool :=
  match (liveOut s { salts := [] } L).2 with
  | some t => t.finished
  | none => false

def shape (L : List (ByteArray × List ByteArray)) : List (ByteArray × Nat) := L.map fun x => (x.1, x.2.length)

theorem liveStep_eq (s : Schedule) (st : State) (b : ByteArray) (fresh : List ByteArray) :
    liveStep s st b fresh = match (stepOf s b).run { st with salts := st.salts ++ fresh, sent := #[] } with
      | .ok ((), t) => .ok (t.sent.toList, (st.salts ++ fresh).take ((st.salts ++ fresh).length - t.salts.length),
          { t with sent := #[] })
      | .error e => .error e :=
  rfl

theorem liveOut_ok {s : Schedule} {st t : State} {b : ByteArray} {fresh : List ByteArray}
    {rest : List (ByteArray × List ByteArray)}
    (h : (stepOf s b).run { st with salts := st.salts ++ fresh, sent := #[] } = .ok ((), t)) :
    liveOut s st ((b, fresh) :: rest) =
      (t.sent.toList ++ (liveOut s { t with sent := #[] } rest).1, (liveOut s { t with sent := #[] } rest).2) := by
  simp only [liveOut, liveStep_eq, h]

theorem liveOut_error {s : Schedule} {st : State} {b : ByteArray} {fresh : List ByteArray}
    {rest : List (ByteArray × List ByteArray)} {e : String}
    (h : (stepOf s b).run { st with salts := st.salts ++ fresh, sent := #[] } = .error e) :
    liveOut s st ((b, fresh) :: rest) = ([], none) := by
  simp only [liveOut, liveStep_eq, h]

theorem liveOut_liveRun (s : Schedule) : ∀ (L : List (ByteArray × List ByteArray)) (st : State),
    ∃ P t, liveRun s st P = .ok ((liveOut s st L).1, t) ∧
      ((liveOut s st L).2 = some t ∨ (liveOut s st L).2 = none)
  | [], st => ⟨[], st, rfl, Or.inl rfl⟩
  | (b, fresh) :: rest, st => by
    rcases hm : (stepOf s b).run { st with salts := st.salts ++ fresh, sent := #[] } with e | ⟨⟨⟩, t⟩
    · rw [liveOut_error hm]
      exact ⟨[], st, rfl, Or.inr rfl⟩
    · obtain ⟨P, u, hP, hu⟩ := liveOut_liveRun s rest { t with sent := #[] }
      rw [liveOut_ok hm]
      refine ⟨(b, fresh) :: P, u, ?_, hu⟩
      have hl : liveStep s st b fresh = .ok (t.sent.toList,
          (st.salts ++ fresh).take ((st.salts ++ fresh).length - t.salts.length), { t with sent := #[] }) := by
        simp only [liveStep_eq, hm]
      simp only [liveRun, hl, hP]

theorem liveFinished_eq (s : Schedule) (L : List (ByteArray × List ByteArray)) :
    liveFinished s L = ((liveOut s { salts := [] } L).2.map State.finished).getD false := by
  unfold liveFinished
  rcases (liveOut s { salts := [] } L).2 with _ | t <;> rfl

theorem roundT_null (s : Schedule) {i : Nat} (h : s.rounds.size ≤ i) : roundT s i = [] := by
  have e : s.rounds.getD i .null = .null := by
    rw [Array.getD_eq_getD_getElem?, Array.getElem?_eq_none h]
    rfl
  rw [roundT, e]
  rfl

theorem roundsT_stable (s : Schedule) : ∀ r, s.rounds.size ≤ r → roundsT s r = roundsT s s.rounds.size
  | 0, h => by rw [Nat.le_zero.mp h]
  | r + 1, h => by
    rcases Nat.lt_or_ge s.rounds.size (r + 1) with h' | h'
    · rw [roundsT_succ, roundT_null s (Nat.le_of_lt_succ h'), List.append_nil,
        roundsT_stable s r (Nat.le_of_lt_succ h')]
    · rw [Nat.le_antisymm h h']

theorem roundsT_mono (s : Schedule) {r : Nat} : ∀ r', r ≤ r' → roundsT s r <+: roundsT s r'
  | 0, h => by
    rw [Nat.le_zero.mp h]
    exact List.prefix_refl _
  | r' + 1, h => by
    rcases Nat.lt_or_ge r (r' + 1) with h' | h'
    · rw [roundsT_succ]
      exact (roundsT_mono s r' (Nat.le_of_lt_succ h')).trans (List.prefix_append _ _)
    · rw [Nat.le_antisymm h h']
      exact List.prefix_refl _

theorem roundsT_prefix (s : Schedule) (r : Nat) : roundsT s r <+: roundsT s s.rounds.size := by
  rcases Nat.le_total r s.rounds.size with h | h
  · exact roundsT_mono s _ h
  · rw [roundsT_stable s r h]
    exact List.prefix_refl _

theorem sofar_prefix {s : Schedule} {v : State} (hv : Inv s v) : sofar s v <+: tmpl s s.tables := by
  cases hh : v.hello
  · obtain ⟨hc, hl, hr, hf⟩ := hv.fresh hh
    simp [sofar, hh, hc, hl, hr, hf, roundsT]
  cases hc : v.committed
  · obtain ⟨hl, hr, hf⟩ := hv.uncommitted hc
    simp [sofar, tmpl, hh, hc, hl, hr, hf, roundsT]
  cases hl : v.linked
  · obtain ⟨hr, hf⟩ := hv.unlinked hl
    simp [sofar, tmpl, hh, hc, hl, hr, hf, roundsT]
  cases hf : v.finished
  · have e : sofar s v = [.hello s.hello, commitT s s.tables, .link s.y] ++ roundsT s v.rounds := by
      simp [sofar, hh, hc, hl, hf]
    rw [e, tmpl, List.append_assoc]
    exact (List.prefix_append_right_inj _).mpr ((roundsT_prefix s _).trans (List.prefix_append _ _))
  · obtain ⟨-, hr⟩ := hv.finished hf
    have e : sofar s v = tmpl s s.tables := by
      simp [sofar, tmpl, hh, hc, hl, hr, hf]
    rw [e]
    exact List.prefix_refl _

theorem sofar_finished {s : Schedule} {v : State} (hv : Inv s v) (hf : v.finished = true) :
    sofar s v = tmpl s s.tables := by
  obtain ⟨hl, hr⟩ := hv.finished hf
  have hc := hv.committed hl
  simp [sofar, tmpl, hv.hello hc, hc, hl, hr, hf]

theorem zipWith_append_of_le {α β γ : Type} (f : α → β → γ) :
    ∀ (a : List α) (b c : List β), a.length ≤ b.length → List.zipWith f a (b ++ c) = List.zipWith f a b
  | [], _, _, _ => by simp
  | _ :: _, [], _, h => by simp at h
  | x :: a, y :: b, c, h => by
    simp only [List.cons_append, List.zipWith_cons_cons, List.length_cons] at h ⊢
    rw [zipWith_append_of_le f a b c (by omega)]

theorem liveOut_rel (s : Schedule) : ∀ (L L' : List (ByteArray × List ByteArray)) (st : State) (z' : List ByteArray),
    shape L = shape L' → st.salts.length = z'.length →
    (liveOut s st L).1.map blank = (liveOut s { st with salts := z' } L').1.map blank ∧
      (liveOut s st L).2.map State.finished = (liveOut s { st with salts := z' } L').2.map State.finished ∧
      ∃ ds : List ByteArray, (∀ d ∈ ds, d.size = 64) ∧ ds.length ≤ (st.salts ++ L.flatMap (·.2)).length ∧
        (liveOut s st L).1.flatMap holes =
          List.zipWith Hm96.Default512.commitString ds (st.salts ++ L.flatMap (·.2)) ∧
        (liveOut s { st with salts := z' } L').1.flatMap holes =
          List.zipWith Hm96.Default512.commitString ds (z' ++ L'.flatMap (·.2))
  | [], [], _, _, _, _ => ⟨rfl, rfl, [], (fun _ h => nomatch h), Nat.zero_le _, rfl, rfl⟩
  | [], _ :: _, _, _, h, _ => by simp [shape] at h
  | _ :: _, [], _, _, h, _ => by simp [shape] at h
  | (b, f) :: rest, (b', f') :: rest', st, z', h, hz => by
    simp only [shape, List.map_cons, List.cons.injEq, Prod.mk.injEq] at h
    obtain ⟨⟨rfl, hf⟩, hr⟩ := h
    have H := stepOf_rel (ys := st.salts ++ f) (ys' := z' ++ f') s b { st with salts := st.salts ++ f, sent := #[] }
      (z' ++ f') #[] ⟨rfl, by simp only [List.length_append, hz, hf], [], (fun _ h => nomatch h), Nat.zero_le _,
        rfl, rfl, rfl, rfl⟩
    change Both _ ((stepOf s b).run _) ((stepOf s b).run { st with salts := z' ++ f', sent := #[] }) at H
    rcases hm : (stepOf s b).run { st with salts := st.salts ++ f, sent := #[] } with e | ⟨⟨⟩, t⟩ <;>
      rcases hm' : (stepOf s b).run { st with salts := z' ++ f', sent := #[] } with e' | ⟨⟨⟩, t'⟩ <;>
      rw [hm, hm'] at H
    · rw [liveOut_error hm, liveOut_error (st := { st with salts := z' }) (fresh := f') hm']
      exact ⟨rfl, rfl, [], (fun _ h => nomatch h), Nat.zero_le _, rfl, rfl⟩
    · exact H.elim
    · exact H.elim
    · obtain ⟨q, q', ⟨rfl, rfl⟩, z₁, x₁, ht', hb, hlen, ds₁, hds₁, hle₁, hz₁, hz₁', hh₁, hh₁'⟩ := H
      rw [liveOut_ok (rest := rest) hm, liveOut_ok (st := { st with salts := z' }) (fresh := f') (rest := rest') hm']
      have e1 : ({ t' with sent := #[] } : State) = { { t with sent := #[] } with salts := z₁ } := by
        rw [ht']
      have hx : t'.sent = x₁ := by rw [ht']
      rw [e1, hx]
      have hlen' : ({ t with sent := #[] } : State).salts.length = z₁.length := by
        show t.salts.length = z₁.length
        rw [hz₁, hz₁', List.length_drop, List.length_drop, hlen]
      obtain ⟨ihb, ihf, ds₂, hds₂, hle₂, hh₂, hh₂'⟩ := liveOut_rel s rest rest' { t with sent := #[] } z₁ hr hlen'
      have hle₂' : ds₂.length ≤ t.salts.length + (rest.flatMap (·.2)).length := by
        have := hle₂
        simp only [List.length_append] at this
        exact this
      rw [hz₁, List.length_drop, List.length_append] at hle₂'
      simp only [List.length_append] at hle₁
      have hh₁ : t.sent.toList.flatMap holes = List.zipWith Hm96.Default512.commitString ds₁ (st.salts ++ f) :=
        (List.append_nil _).symm.trans hh₁
      have hh₁' : x₁.toList.flatMap holes = List.zipWith Hm96.Default512.commitString ds₁ (z' ++ f') :=
        (List.append_nil _).symm.trans hh₁'
      have hh₂ : (liveOut s { t with sent := #[] } rest).1.flatMap holes =
          List.zipWith Hm96.Default512.commitString ds₂ ((st.salts ++ f).drop ds₁.length ++ rest.flatMap (·.2)) := by
        rw [← hz₁]
        exact hh₂
      have hh₂' : (liveOut s { { t with sent := #[] } with salts := z₁ } rest').1.flatMap holes =
          List.zipWith Hm96.Default512.commitString ds₂ ((z' ++ f').drop ds₁.length ++ rest'.flatMap (·.2)) := by
        rw [← hz₁']
        exact hh₂'
      have hl₁ : ds₁.length ≤ (st.salts ++ f).length := by
        simp only [List.length_append]
        exact hle₁
      have hl₁' : ds₁.length ≤ (z' ++ f').length := by
        simp only [List.length_append, ← hz, ← hf]
        exact hle₁
      refine ⟨?_, ihf, ds₁ ++ ds₂, fun d hd => (List.mem_append.mp hd).elim (hds₁ d) (hds₂ d), ?_, ?_, ?_⟩
      · rw [List.map_append, List.map_append, hb, ihb]
      · simp only [List.length_append, List.flatMap_cons]
        omega
      · rw [List.flatMap_append, hh₁, hh₂, List.flatMap_cons, ← List.append_assoc, zipWith_append_left,
          zipWith_append_of_le _ _ _ _ hl₁, List.drop_append_of_le_length hl₁]
      · rw [List.flatMap_append, hh₁', hh₂', List.flatMap_cons, ← List.append_assoc, zipWith_append_left,
          zipWith_append_of_le _ _ _ _ hl₁', List.drop_append_of_le_length hl₁']

theorem digests_eq {ds ds' ys : List ByteArray} (h : ∀ d ∈ ds, d.size = 64) (h' : ∀ d ∈ ds', d.size = 64)
    (hl : ds.length ≤ ys.length) (hl' : ds'.length ≤ ys.length)
    (e : List.zipWith Hm96.Default512.commitString ds ys = List.zipWith Hm96.Default512.commitString ds' ys) :
    ds = ds' := by
  have hlen : ds.length = ds'.length := by
    have := congrArg List.length e
    simp only [List.length_zipWith] at this
    omega
  apply List.ext_getElem hlen
  intro i hi hi'
  have := List.getElem_of_eq e (i := i) (by simp only [List.length_zipWith]; omega)
  simp only [List.getElem_zipWith] at this
  exact commitString_inj (h _ (List.getElem_mem _)) (h' _ (List.getElem_mem _)) this

theorem fillAll_take : ∀ (l : List Sent) (k : Nat) (cs : List ByteArray),
    fillAll (l.take k) cs = (fillAll l cs).take k
  | [], _, _ => by simp [fillAll]
  | _ :: _, 0, _ => by simp [fillAll]
  | m :: ms, k + 1, cs => by
    simp only [List.take_succ_cons, fillAll, fillAll_take ms k]

theorem length_salts : ∀ L : List (ByteArray × List ByteArray),
    (L.flatMap (·.2)).length = ((shape L).map (·.2)).sum
  | [] => rfl
  | x :: L => by
    rw [List.flatMap_cons, List.length_append, length_salts L]
    rfl

theorem shape_canon (R : List (ByteArray × Nat)) :
    shape (R.map fun x => (x.1, List.replicate x.2 ByteArray.empty)) = R := by
  induction R with
  | nil => rfl
  | cons x R ih =>
    simp only [shape, List.map_cons, List.length_replicate] at ih ⊢
    rw [ih]

theorem run_any (s : Schedule) (reqs : List ByteArray) (n : Nat) :
    ∃ (ds : List ByteArray) (k : Nat) (fin : Bool),
      (∀ d ∈ ds, d.size = 64) ∧ ds.length ≤ n ∧ k ≤ (tmpl s s.tables).length ∧
        (fin = true → k = (tmpl s s.tables).length) ∧
        ∀ ys : List ByteArray, ys.length = n →
          run s reqs ys =
            (((fillAll (tmpl s s.tables) (List.zipWith Hm96.Default512.commitString ds ys)).take k).toArray, fin) := by
  have hl₀ : (List.replicate n ByteArray.empty).length = n := List.length_replicate
  obtain ⟨-, -, -, -, -, -, ds, hds, hle, -, -, hh, -⟩ :=
    loop_rel s reqs (ys := List.replicate n ByteArray.empty) (ys' := List.replicate n ByteArray.empty) rfl
  have hI := loop_inv s reqs (List.replicate n ByteArray.empty)
  have hrun : ∀ ys : List ByteArray, ys.length = n →
      ∃ x, run s reqs ys = (x, (loop s reqs (List.replicate n ByteArray.empty)).1.finished &&
          !(loop s reqs (List.replicate n ByteArray.empty)).2) ∧
        x.toList.map blank = (loop s reqs (List.replicate n ByteArray.empty)).1.sent.toList.map blank ∧
        holesOf x = List.zipWith Hm96.Default512.commitString ds ys := by
    intro ys hys
    obtain ⟨h2, z, x, he, hb, -, ds₁, hds₁, hle₁, -, -, hh₁, hh₁'⟩ :=
      loop_rel s reqs (ys := List.replicate n ByteArray.empty) (ys' := ys) (by rw [hl₀, hys])
    have e1 : ds₁ = ds := digests_eq hds₁ hds hle₁ hle
      (((List.append_nil _).symm.trans hh₁).symm.trans ((List.append_nil _).symm.trans hh))
    refine ⟨x, ?_, hb.symm, ?_⟩
    · rw [run_eq, he, h2]
    · rw [← e1]
      exact (List.append_nil _).symm.trans hh₁'
  revert hI hrun
  generalize loop s reqs (List.replicate n ByteArray.empty) = r
  obtain ⟨u, stp⟩ := r
  intro hI hrun
  dsimp only at hI hrun
  refine ⟨ds, (sofar s u).length, u.finished && !stp, hds, hl₀ ▸ hle, (sofar_prefix hI).length_le,
    fun hf => ?_, fun ys hys => ?_⟩
  · simp only [Bool.and_eq_true] at hf
    rw [sofar_finished hI hf.1]
  · obtain ⟨x, hx, hb, hh'⟩ := hrun ys hys
    have hh'' : x.toList.flatMap holes = List.zipWith Hm96.Default512.commitString ds ys := hh'
    have e3 := fillAll_blank_holes x.toList []
    rw [List.append_nil, hb, hI.sent, List.prefix_iff_eq_take.mp (sofar_prefix hI), hh'', fillAll_take] at e3
    rw [hx, e3, Array.toArray_toList]

theorem liveOut_blank (s : Schedule) (L : List (ByteArray × List ByteArray)) :
    ∃ k : Nat, k ≤ (tmpl s s.tables).length ∧ (liveFinished s L = true → k = (tmpl s s.tables).length) ∧
      (liveOut s { salts := [] } L).1.map blank = (tmpl s s.tables).take k := by
  obtain ⟨P, t, hP, ht⟩ := liveOut_liveRun s L { salts := [] }
  have e := run_live s P _ t hP
  rw [run_eq] at e
  have hI := loop_inv s (P.map (·.1)) (P.flatMap (·.2))
  revert e hI
  generalize loop s (P.map (·.1)) (P.flatMap (·.2)) = r
  obtain ⟨u, stp⟩ := r
  intro e hI
  dsimp only at e hI
  obtain ⟨he1, he2⟩ := Prod.mk.inj e
  refine ⟨(sofar s u).length, (sofar_prefix hI).length_le, fun hf => ?_, ?_⟩
  · rw [liveFinished_eq] at hf
    rcases ht with ht | ht <;> rw [ht] at hf
    · have hu : (u.finished && !stp) = true := he2.trans hf
      simp only [Bool.and_eq_true] at hu
      rw [sofar_finished hI hu.1]
    · exact nomatch hf
  · rw [← List.prefix_iff_eq_take.mp (sofar_prefix hI), ← hI.sent, he1, List.toList_toArray]

end FlockSoundness.Discharge.Recursion.ZkServerAny

namespace Flock.SecurityProofs

open Flock.ZkServer FlockSoundness.Discharge.Recursion
open ZkServerRun ZkServerTemplate ZkServerAny

theorem liveOut_any (s : Schedule) (R : List (ByteArray × Nat)) :
    ∃ (ds : List ByteArray) (k : Nat) (fin : Bool),
      (∀ d ∈ ds, d.size = 64) ∧ ds.length ≤ (R.map (·.2)).sum ∧ k ≤ (tmpl s s.tables).length ∧
        (fin = true → k = (tmpl s s.tables).length) ∧
        ∀ L : List (ByteArray × List ByteArray), shape L = R →
          (liveOut s { salts := [] } L).1 =
              (fillAll (tmpl s s.tables) (List.zipWith Hm96.Default512.commitString ds (L.flatMap (·.2)))).take k ∧
            liveFinished s L = fin := by
  obtain ⟨L₀, hS₀⟩ : ∃ L₀, shape L₀ = R := ⟨_, shape_canon R⟩
  obtain ⟨k, hk, hfin, hsof⟩ := liveOut_blank s L₀
  obtain ⟨-, -, ds, hds, hle, hh, -⟩ := liveOut_rel s L₀ L₀ { salts := [] } [] rfl rfl
  refine ⟨ds, k, liveFinished s L₀, hds, ?_, hk, hfin, fun L hL => ?_⟩
  · have h1 : ds.length ≤ (L₀.flatMap (·.2)).length := hle
    rw [length_salts, hS₀] at h1
    exact h1
  · obtain ⟨hb, hf, ds₁, hds₁, hle₁, hh₁, hh₁'⟩ := liveOut_rel s L₀ L { salts := [] } [] (hS₀.trans hL.symm) rfl
    have hb' : (liveOut s { salts := [] } L₀).1.map blank = (liveOut s { salts := [] } L).1.map blank := hb
    have hf' : (liveOut s { salts := [] } L₀).2.map State.finished =
        (liveOut s { salts := [] } L).2.map State.finished := hf
    have hh₁'' : (liveOut s { salts := [] } L).1.flatMap holes =
        List.zipWith Hm96.Default512.commitString ds₁ (L.flatMap (·.2)) := hh₁'
    rw [digests_eq hds₁ hds hle₁ hle (hh₁.symm.trans hh)] at hh₁''
    refine ⟨?_, ?_⟩
    · have e3 := fillAll_blank_holes (liveOut s { salts := [] } L).1 []
      rw [List.append_nil, ← hb', hsof, hh₁'', fillAll_take] at e3
      exact e3.symm
    · rw [liveFinished_eq, liveFinished_eq, ← hf']

end Flock.SecurityProofs
