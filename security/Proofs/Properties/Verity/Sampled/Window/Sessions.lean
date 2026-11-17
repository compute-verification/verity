module

public import Proofs.Properties.Verity.Sampled.Stops
public import Proofs.Properties.Verity.Sampled.Layout

/-! SPOT's coupling over a window: the readings, a session at a time. -/

@[expose] public section

namespace Verity.Sampled

/-- **The tree an opening frame opens**; `none` at any other frame. -/
def _root_.Verity.Frame.tree? : Frame → Option ℕ
  | .opening i _ _ => some i
  | _ => none

def openStep (acc : List ℕ) (r : Release) : List ℕ :=
  match r.frame.tree? with
  | some i => if i ∈ acc then acc else acc ++ [i]
  | none => acc

def openedTrees (log : List Release) : List ℕ := log.foldl openStep []

theorem openStep_some {acc : List ℕ} {r : Release} {i : ℕ} (h : r.frame.tree? = some i) :
    openStep acc r = if i ∈ acc then acc else acc ++ [i] := by
  unfold openStep
  rw [h]

theorem openStep_none {acc : List ℕ} {r : Release} (h : r.frame.tree? = none) : openStep acc r = acc := by
  unfold openStep
  rw [h]

theorem openStep_prefix (acc : List ℕ) (r : Release) : ∃ d, openStep acc r = acc ++ d := by
  cases h : r.frame.tree? with
  | none => exact ⟨[], by rw [openStep_none h, List.append_nil]⟩
  | some i =>
    rw [openStep_some h]
    by_cases hi : i ∈ acc
    · exact ⟨[], by rw [ite_eq_left hi, List.append_nil]⟩
    · exact ⟨[i], by rw [ite_eq_right hi]⟩

theorem foldl_openStep_prefix : ∀ (l : List Release) (acc : List ℕ), ∃ d, l.foldl openStep acc = acc ++ d
  | [], acc => ⟨[], by rw [List.foldl_nil, List.append_nil]⟩
  | r :: l, acc => by
    obtain ⟨d, hd⟩ := openStep_prefix acc r
    obtain ⟨e, he⟩ := foldl_openStep_prefix l (openStep acc r)
    exact ⟨d ++ e, by rw [List.foldl_cons, he, hd, List.append_assoc]⟩

theorem openedTrees_append (L d : List Release) : ∃ e, openedTrees (L ++ d) = openedTrees L ++ e := by
  unfold openedTrees
  rw [List.foldl_append]
  exact foldl_openStep_prefix d _

theorem openStep_nodup {acc : List ℕ} (r : Release) (h : acc.Nodup) : (openStep acc r).Nodup := by
  cases hr : r.frame.tree? with
  | none =>
    rw [openStep_none hr]
    exact h
  | some i =>
    rw [openStep_some hr]
    by_cases hi : i ∈ acc
    · rw [ite_eq_left hi]
      exact h
    · rw [ite_eq_right hi]
      refine List.nodup_append.2 ⟨h, List.nodup_cons.2 ⟨List.not_mem_nil, List.nodup_nil⟩, fun a ha b hb e => ?_⟩
      rw [List.mem_singleton] at hb
      rw [e, hb] at ha
      exact hi ha

theorem foldl_openStep_nodup : ∀ (l : List Release) (acc : List ℕ), acc.Nodup → (l.foldl openStep acc).Nodup
  | [], _, h => h
  | r :: l, _, h => foldl_openStep_nodup l _ (openStep_nodup r h)

theorem openedTrees_nodup (L : List Release) : (openedTrees L).Nodup :=
  foldl_openStep_nodup L [] List.nodup_nil

theorem openedTrees_new {L : List Release} {r : Release} {i s n : ℕ} (hr : r.frame = .opening i s n)
    (hi : i ∉ openedTrees L) (d : List Release) : ∃ e, openedTrees (L ++ r :: d) = openedTrees L ++ i :: e := by
  have ht : r.frame.tree? = some i := by
    rw [hr]
    rfl
  obtain ⟨e, he⟩ := foldl_openStep_prefix d (openStep (openedTrees L) r)
  refine ⟨e, ?_⟩
  have h1 : openedTrees (L ++ r :: d) = d.foldl openStep (openStep (openedTrees L) r) := by
    unfold openedTrees
    rw [List.foldl_append, List.foldl_cons]
  rw [h1, he, openStep_some ht, ite_eq_right hi, List.append_assoc, List.singleton_append]

theorem openedTrees_getElem?_append {L : List Release} {u i : ℕ} (h : (openedTrees L)[u]? = some i)
    (d : List Release) : (openedTrees (L ++ d))[u]? = some i := by
  obtain ⟨e, he⟩ := openedTrees_append L d
  rw [he, List.getElem?_append_left (List.getElem?_eq_some_iff.1 h).1, h]

def slotStream (tables : Array String) (t : ℕ) (r : Fin 2) : Option String :=
  if t = tables.size ∧ r.val = 0 then some _root_.Spot.Challenger.TREE_SESSION
  else tables[t]?.map (repStreamName · r.val)

theorem slotStream_lt {tables : Array String} {t : ℕ} (ht : t < tables.size) (r : Fin 2) :
    slotStream tables t r = tables[t]?.map (repStreamName · r.val) := by
  have hc : ¬ (t = tables.size ∧ r.val = 0) := fun h => by omega
  unfold slotStream
  rw [if_neg hc]

theorem slotStream_table {tables : Array String} {t : ℕ} {a : String} (ha : tables[t]? = some a) (r : Fin 2) :
    slotStream tables t r = some (repStreamName a r.val) := by
  have ha' : tables.toList[t]? = some a := by
    rw [Array.getElem?_toList]
    exact ha
  have hlt : t < tables.size := (List.getElem?_eq_some_iff.1 ha').1
  rw [slotStream_lt hlt, ha, Option.map_some]

theorem slotStream_close (tables : Array String) :
    slotStream tables tables.size 0 = some _root_.Spot.Challenger.TREE_SESSION :=
  if_pos ⟨rfl, rfl⟩

theorem slotStream_session {tables : Array String} {t : ℕ} {r : Fin 2}
    (h : slotStream tables t r = some _root_.Spot.Challenger.TREE_SESSION) : t = tables.size ∧ r = 0 := by
  unfold slotStream at h
  by_cases hc : t = tables.size ∧ r.val = 0
  · exact ⟨hc.1, Fin.ext hc.2⟩
  · rw [if_neg hc] at h
    exact absurd h rep_ne_session

theorem slotStream_inj {tables : Array String} (ht : tables.toList.Nodup) {t t' : ℕ} {r r' : Fin 2} {x : String}
    (h : slotStream tables t r = some x) (h' : slotStream tables t' r' = some x) : t = t' ∧ r = r' := by
  by_cases hc : t = tables.size ∧ r.val = 0
  · have hx : x = _root_.Spot.Challenger.TREE_SESSION := by
      unfold slotStream at h
      rw [if_pos hc] at h
      exact (Option.some.inj h).symm
    rw [hx] at h'
    obtain ⟨h1, h2⟩ := slotStream_session h'
    exact ⟨hc.1.trans h1.symm, (Fin.ext hc.2 : r = 0).trans h2.symm⟩
  · by_cases hc' : t' = tables.size ∧ r'.val = 0
    · have hx : x = _root_.Spot.Challenger.TREE_SESSION := by
        unfold slotStream at h'
        rw [if_pos hc'] at h'
        exact (Option.some.inj h').symm
      rw [hx] at h
      obtain ⟨h1, rfl⟩ := slotStream_session h
      exact absurd ⟨h1, rfl⟩ hc
    · unfold slotStream at h h'
      rw [if_neg hc] at h
      rw [if_neg hc'] at h'
      obtain ⟨a, ha, hae⟩ := Option.map_eq_some_iff.1 h
      obtain ⟨a', ha', hae'⟩ := Option.map_eq_some_iff.1 h'
      obtain ⟨rfl, rfl⟩ := repStreamName_inj (hae.trans hae'.symm)
      rw [← Array.getElem?_toList] at ha ha'
      exact ⟨nodup_getElem?_inj ht ha ha', rfl⟩

def sessionRound (tables : Array String) (u : ℕ) : Slot → List Release → Frame → Prop
  | .link, pre, fr => ∃ i s n, fr = .opening i s n ∧ Served pre i s n ∧ i ∉ openedTrees pre ∧
      (openedTrees pre).length = u
  | .rep t r k, pre, fr => ∃ i s n, fr = .opening i s n ∧ Served pre i s n ∧ (openedTrees pre)[u]? = some i ∧
      ((helloSpecs pre)[i]?.bind fun sp => sp.streams[s]?) = slotStream tables t r ∧
      openedOf pre i s = k

theorem sessionRound_served {tables : Array String} {u : ℕ} {σ : Slot} {pre : List Release} {i s n : ℕ}
    (h : sessionRound tables u σ pre (.opening i s n)) : Served pre i s n := by
  cases σ with
  | link =>
    obtain ⟨_, _, _, hfr, hS, -⟩ := h
    obtain ⟨rfl, rfl, rfl⟩ := Frame.opening.inj hfr
    exact hS
  | rep t r k =>
    obtain ⟨_, _, _, hfr, hS, -⟩ := h
    obtain ⟨rfl, rfl, rfl⟩ := Frame.opening.inj hfr
    exact hS

theorem sessionRound_opening {tables : Array String} {u : ℕ} {σ : Slot} {pre : List Release} {fr : Frame}
    (h : sessionRound tables u σ pre fr) : ∃ i s n, fr = .opening i s n := by
  cases σ with
  | link =>
    obtain ⟨i, s, n, h, -⟩ := h
    exact ⟨i, s, n, h⟩
  | rep t r k =>
    obtain ⟨i, s, n, h, -⟩ := h
    exact ⟨i, s, n, h⟩

theorem sessionRound_unique {tables : Array String} (ht : tables.toList.Nodup) {u u' : ℕ} {σ σ' : Slot}
    {pre : List Release} {fr : Frame} (h : sessionRound tables u σ pre fr) (h' : sessionRound tables u' σ' pre fr) :
    u = u' ∧ σ = σ' := by
  cases σ with
  | link =>
    cases σ' with
    | link =>
      obtain ⟨_, _, _, _, _, _, hu⟩ := h
      obtain ⟨_, _, _, _, _, _, hu'⟩ := h'
      exact ⟨hu.symm.trans hu', rfl⟩
    | rep t' r' k' =>
      obtain ⟨i, s, n, rfl, _, hni, _⟩ := h
      obtain ⟨i', s', n', he, _, hu', _⟩ := h'
      obtain ⟨rfl, rfl, rfl⟩ := Frame.opening.inj he
      exact absurd (List.mem_of_getElem? hu') hni
  | rep t r k =>
    cases σ' with
    | link =>
      obtain ⟨i, s, n, rfl, _, hu, _⟩ := h
      obtain ⟨i', s', n', he, _, hni, _⟩ := h'
      obtain ⟨rfl, rfl, rfl⟩ := Frame.opening.inj he
      exact absurd (List.mem_of_getElem? hu) hni
    | rep t' r' k' =>
      obtain ⟨i, s, n, rfl, ⟨sp, hsp, hs, -⟩, hu, hname, hk⟩ := h
      obtain ⟨i', s', n', he, -, hu', hname', hk'⟩ := h'
      obtain ⟨rfl, rfl, rfl⟩ := Frame.opening.inj he
      have hX : ((helloSpecs pre)[i]?.bind fun sp => sp.streams[s]?) = some sp.streams[s] := by
        rw [hsp, Option.bind_some, Array.getElem?_eq_getElem hs]
      rw [hX] at hname hname'
      obtain ⟨rfl, rfl⟩ := slotStream_inj ht hname.symm hname'.symm
      obtain rfl := nodup_getElem?_inj (openedTrees_nodup pre) hu hu'
      subst hk hk'
      exact ⟨rfl, rfl⟩

theorem sessionRound_once {tables : Array String} {u : ℕ} {σ : Slot} {L₁ d : List Release} {r r' : Release}
    {Lg : List Release} (hn : ∀ sp ∈ helloSpecs Lg, sp.streams.toList.Nodup) {L₂' : List Release}
    (he' : Lg = (L₁ ++ r :: d) ++ r' :: L₂') (hq : sessionRound tables u σ L₁ r.frame)
    (hq' : sessionRound tables u σ (L₁ ++ r :: d) r'.frame) : False := by
  cases σ with
  | link =>
    obtain ⟨i, s, n, hfr, -, hni, hu⟩ := hq
    obtain ⟨-, -, -, -, -, -, hu'⟩ := hq'
    obtain ⟨e, he⟩ := openedTrees_new hfr hni d
    rw [he, List.length_append, List.length_cons] at hu'
    omega
  | rep t q k =>
    obtain ⟨i, s, n, hfr, ⟨sp, hsp, hs, -⟩, hu, hname, hk⟩ := hq
    obtain ⟨i', s', n', -, -, hu', hname', hk'⟩ := hq'
    rw [openedTrees_getElem?_append hu] at hu'
    obtain rfl := Option.some.inj hu'
    rw [helloSpecs_getElem?_append hsp, Option.bind_some] at hname'
    rw [hsp, Option.bind_some] at hname
    have hX : sp.streams.toList[s]? = some sp.streams[s] := by
      rw [Array.getElem?_toList, Array.getElem?_eq_getElem hs]
    have hX' : sp.streams.toList[s']? = some sp.streams[s] := by
      rw [Array.getElem?_toList, hname', ← hname, Array.getElem?_eq_getElem hs]
    have hmem : sp ∈ helloSpecs Lg := by
      rw [he', List.append_assoc]
      exact List.mem_of_getElem? (helloSpecs_getElem?_append hsp _)
    obtain rfl := nodup_getElem?_inj (hn sp hmem) hX hX'
    have := openedOf_lt L₁ hfr d
    omega

theorem picksOnce_of_once {R : ℕ} {P : Fin R → List Release → Frame → Prop} {Lg : List Release}
    (h : ∀ j L₁ r d r' L₂', Lg = (L₁ ++ r :: d) ++ r' :: L₂' → P j L₁ r.frame → P j (L₁ ++ r :: d) r'.frame →
      False) : PicksOnce P Lg := by
  have key : ∀ j L₁ L₁', PicksIn P Lg j L₁ → PicksIn P Lg j L₁' → L₁.length < L₁'.length → False := by
    intro j L₁ L₁' h₁ h₁' hl
    obtain ⟨r, L₂, he, hp⟩ := h₁
    obtain ⟨r', L₂', he', hp'⟩ := h₁'
    obtain ⟨d, rfl⟩ := prefix_le (he'.symm.trans he) (Nat.le_of_lt hl)
    have he₂ : d ++ r' :: L₂' = r :: L₂ := by
      rw [List.append_assoc] at he'
      exact List.append_cancel_left (he'.symm.trans he)
    cases d with
    | nil => simp at hl
    | cons x d =>
      rw [List.cons_append] at he₂
      obtain ⟨hx, -⟩ := List.cons.inj he₂
      subst hx
      exact h j L₁ x d r' L₂' he' hp hp'
  intro j L₁ L₁' h₁ h₁'
  rcases Nat.lt_trichotomy L₁.length L₁'.length with hl | hl | hl
  · exact (key j L₁ L₁' h₁ h₁' hl).elim
  · obtain ⟨r, L₂, he, -⟩ := h₁
    obtain ⟨r', L₂', he', -⟩ := h₁'
    exact (List.append_inj (he.symm.trans he') hl).1
  · exact (key j L₁' L₁ h₁' h₁ hl).elim

def closeSlot (T : ℕ) : Slot := .rep T 0 1

def slotOf (T per i : ℕ) : Slot := if i + 1 = per then closeSlot T else slotAt i

theorem slotOf_last (T : ℕ) {per i : ℕ} (h : i + 1 = per) : slotOf T per i = closeSlot T := if_pos h

theorem slotOf_of_ne (T : ℕ) {per i : ℕ} (h : i + 1 ≠ per) : slotOf T per i = slotAt i := if_neg h

theorem slotAt_ne_close {T : ℕ} (hT : 0 < T) (i : ℕ) : slotAt i ≠ closeSlot T := by
  intro h
  cases i with
  | zero => exact absurd h (by simp [slotAt, closeSlot])
  | succ i =>
    have h' : Slot.rep 0 ⟨i % 2, Nat.mod_lt i (by decide)⟩ (i / 2) = .rep T 0 1 := h
    rw [Slot.rep.injEq] at h'
    omega

theorem slotOf_link {T per i : ℕ} (h : slotOf T per i = .link) : i = 0 := by
  by_cases hc : i + 1 = per
  · rw [slotOf_last T hc] at h
    exact absurd h (by simp [closeSlot])
  · rw [slotOf_of_ne T hc] at h
    cases i with
    | zero => rfl
    | succ i => exact absurd h (by simp [slotAt])

theorem slotOf_rep {T per i t k : ℕ} {r : Fin 2} (h : slotOf T per i = .rep t r k) (ht : t < T) :
    i + 1 ≠ per ∧ slotAt i = .rep t r k := by
  by_cases hc : i + 1 = per
  · rw [slotOf_last T hc] at h
    have h' : Slot.rep T 0 1 = .rep t r k := h
    rw [Slot.rep.injEq] at h'
    exact absurd h'.1 (by omega)
  · exact ⟨hc, (slotOf_of_ne T hc).symm.trans h⟩

theorem slotOf_inj {T per i i' : ℕ} (hT : 0 < T) (hi : i < per) (hi' : i' < per)
    (h : slotOf T per i = slotOf T per i') : i = i' := by
  by_cases hc : i + 1 = per
  · by_cases hc' : i' + 1 = per
    · omega
    · rw [slotOf_last T hc, slotOf_of_ne T hc'] at h
      exact absurd h.symm (slotAt_ne_close hT i')
  · by_cases hc' : i' + 1 = per
    · rw [slotOf_of_ne T hc, slotOf_last T hc'] at h
      exact absurd h (slotAt_ne_close hT i)
    · rw [slotOf_of_ne T hc, slotOf_of_ne T hc'] at h
      exact slotAt_injective h

def executionPhaseReadingAt {R : ℕ} (tables : Array String) (per : ℕ) (reveal : ℕ → Slot → Exchange → Option Claims.Msg)
    (record : Exchange → Option Claims.Msg) (recordDigest : Claims.Msg) : Reading R where
  round j := sessionRound tables (j.val / per) (slotOf tables.size per (j.val % per))
  reveal j := reveal (j.val / per) (slotOf tables.size per (j.val % per))
  record := record
  recordDigest := recordDigest

theorem executionPhaseReadingAt_picks {R : ℕ} (tables : Array String) (per : ℕ)
    (reveal : ℕ → Slot → Exchange → Option Claims.Msg) (record : Exchange → Option Claims.Msg)
    (recordDigest : Claims.Msg) :
    ∀ (j : Fin R) pre fr, (executionPhaseReadingAt tables per reveal record recordDigest).round j pre fr →
      ∃ i s n, fr = .opening i s n :=
  fun _ _ _ h => sessionRound_opening h

theorem executionPhaseReadingAt_served {R : ℕ} (tables : Array String) (per : ℕ)
    (reveal : ℕ → Slot → Exchange → Option Claims.Msg) (record : Exchange → Option Claims.Msg)
    (recordDigest : Claims.Msg) :
    ∀ (j : Fin R) pre i s n, (executionPhaseReadingAt tables per reveal record recordDigest).round j pre (.opening i s n) →
      Served pre i s n :=
  fun _ _ _ _ _ h => sessionRound_served h

theorem executionPhaseReadingAt_unique {R : ℕ} {tables : Array String} (ht : tables.toList.Nodup) (hT : 0 < tables.size)
    {per : ℕ} (hper : 0 < per) (reveal : ℕ → Slot → Exchange → Option Claims.Msg)
    (record : Exchange → Option Claims.Msg) (recordDigest : Claims.Msg) {j j' : Fin R} {pre : List Release}
    {fr : Frame} (h : (executionPhaseReadingAt tables per reveal record recordDigest).round j pre fr)
    (h' : (executionPhaseReadingAt tables per reveal record recordDigest).round j' pre fr) : j = j' := by
  obtain ⟨hu, hs⟩ := sessionRound_unique ht h h'
  have hm := slotOf_inj hT (Nat.mod_lt _ hper) (Nat.mod_lt _ hper) hs
  exact Fin.ext (by rw [← Nat.div_add_mod j.val per, ← Nat.div_add_mod j'.val per, hu, hm])

theorem picksOnce_executionPhaseReadingAt {R : ℕ} (tables : Array String) (per : ℕ)
    (reveal : ℕ → Slot → Exchange → Option Claims.Msg) (record : Exchange → Option Claims.Msg)
    (recordDigest : Claims.Msg) {Lg : List Release} (hn : ∀ sp ∈ helloSpecs Lg, sp.streams.toList.Nodup) :
    PicksOnce (executionPhaseReadingAt (R := R) tables per reveal record recordDigest).round Lg :=
  picksOnce_of_once fun _ _ _ _ _ _ he hp hp' => sessionRound_once hn he hp hp'

end Verity.Sampled
