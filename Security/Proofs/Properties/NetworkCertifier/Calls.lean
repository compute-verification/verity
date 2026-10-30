import Properties.NetworkCertifier.Calls

/-!
# The certifier's calls: `RefusesPastCallBits`

`refusesPastCallBits`: a forward whose words root the certifier accepts stayed within its bits. The proof carries an
invariant (`Inv`) over the window's events and answers so far and the certifier's state after them: each forward's
calls' MACs are those of the calls of it the certifier took (`Inv.macs`), a forward of a call it took is in its state
(`Inv.took`), and the open forward is unrefused, took a call, and its count is at least the bits of the buckets counted
since any call of it (`Inv.isOpen`). The words root's step accepts only a forward whose count, with the release's bits,
doesn't pass the cap.

Its witnesses: `refusesPastCallBits_honest`, that the certifier accepts `honestEvents`' forward at its words root, and
`refusesPastCallBits_nontrivial`, that a certifier taking every event (`acceptsAll`) accepts `pastEvents`' forward,
which is past its bits.
-/

open NetworkCertifier.Calls

namespace Properties.NetworkCertifier

namespace CallsProofs

/-! ## The run, event by event -/

theorem run_cons (cap : Cap) (rb : Nat) (st : State) (ev : Event) (evs : List Event) :
    run cap rb st (ev :: evs) =
      ((step cap rb st ev).2 :: (run cap rb (step cap rb st ev).1 evs).1, (run cap rb (step cap rb st ev).1 evs).2) :=
  rfl

theorem run_nil (cap : Cap) (rb : Nat) (st : State) : run cap rb st [] = ([], st) := rfl

/-- The answers to the first `n` events are the run's on them. -/
theorem run_take (cap : Cap) (rb : Nat) (st : State) (evs : List Event) (n : Nat) :
    (run cap rb st evs).1.take n = (run cap rb st (evs.take n)).1 := by
  induction evs generalizing st n with
  | nil => simp [run_nil]
  | cons ev evs ih =>
    cases n with
    | zero => simp [run_nil]
    | succ n => simp [run_cons, ih]

/-- The answer to event `i` is the step's on it from the state after the events before it. -/
theorem run_getElem? (cap : Cap) (rb : Nat) (st : State) (evs : List Event) (i : Nat) :
    (run cap rb st evs).1[i]? = evs[i]?.map fun ev => (step cap rb (run cap rb st (evs.take i)).2 ev).2 := by
  induction evs generalizing st i with
  | nil => simp [run_nil]
  | cons ev evs ih =>
    cases i with
    | zero => simp [run_cons, run_nil]
    | succ i => simp [run_cons, ih]

theorem take_zip {α β : Type} :
    ∀ (l₁ : List α) (l₂ : List β) (n : Nat), (l₁.zip l₂).take n = (l₁.take n).zip (l₂.take n)
  | [], _, _ => by simp
  | _ :: _, [], _ => by simp
  | _ :: _, _ :: _, 0 => by simp
  | _ :: l₁, _ :: l₂, n + 1 => by simp [take_zip l₁ l₂ n]

/-! ## Finding a forward -/

/-- Forward `k` of the state, if it holds one. -/
def fwd (st : State) (k : Nat) : Option Forward := st.forwards.find? (·.id == k)

/-- The MACs of forward `k`'s calls in the state: none when it holds no forward `k`. -/
def macsOf (st : State) (k : Nat) : Nat :=
  match fwd st k with
  | some f => (f.calls.map (·.shape.macs)).sum
  | none => 0

theorem find?_update (st : State) (k k' : Nat) (g : Forward → Forward) (hg : ∀ f, (g f).id = f.id) :
    fwd (st.update k g) k' = (fwd st k').map fun f => if f.id == k then g f else f := by
  unfold fwd State.update
  rw [List.find?_map]
  have : ((fun x : Forward => x.id == k') ∘ fun f => if f.id == k then g f else f) = (fun x => x.id == k') := by
    funext f
    simp only [Function.comp_apply]
    split <;> simp [hg]
  rw [this]

theorem fwd_update_self (st : State) (k : Nat) (g : Forward → Forward) (hg : ∀ f, (g f).id = f.id) :
    fwd (st.update k g) k = (fwd st k).map g := by
  rw [find?_update st k k g hg]
  cases h : fwd st k with
  | none => rfl
  | some f =>
    have : f.id = k := by
      have := List.find?_some h
      simpa using this
    simp [this]

theorem fwd_update_ne (st : State) (k k' : Nat) (g : Forward → Forward) (hg : ∀ f, (g f).id = f.id) (hk : k' ≠ k) :
    fwd (st.update k g) k' = fwd st k' := by
  rw [find?_update st k k' g hg]
  cases h : fwd st k' with
  | none => rfl
  | some f =>
    have : f.id = k' := by
      have := List.find?_some h
      simpa using this
    simp [this, hk]

theorem find?_insertById_self (x : Forward) :
    ∀ l : List Forward, (∀ y ∈ l, y.id ≠ x.id) → (insertById Forward.id x l).find? (·.id == x.id) = some x
  | [], _ => by simp [insertById]
  | y :: ys, h => by
    unfold insertById
    have hy : y.id ≠ x.id := h y (by simp)
    split
    · simp
    · simp [hy, find?_insertById_self x ys fun z hz => h z (by simp [hz])]

theorem find?_insertById_ne (x : Forward) (k : Nat) (hk : x.id ≠ k) :
    ∀ l : List Forward, (insertById Forward.id x l).find? (·.id == k) = l.find? (·.id == k)
  | [] => by simp [insertById, hk]
  | y :: ys => by
    unfold insertById
    split
    · simp [hk]
    · by_cases hy : y.id = k
      · simp [hy]
      · simp [hy, find?_insertById_ne x k hk ys]

theorem find?_filter_ne (k k' : Nat) (hk : k ≠ k') :
    ∀ l : List Forward, (l.filter (·.id != k')).find? (·.id == k) = l.find? (·.id == k)
  | [] => rfl
  | y :: ys => by
    have ih := find?_filter_ne k k' hk ys
    by_cases hy' : y.id = k'
    · have hyk : (y.id == k) = false := by simp; omega
      have h1 : (y.id != k') = false := by simp [hy']
      simp only [List.filter_cons, h1, List.find?_cons, hyk]
      exact ih
    · have h1 : (y.id != k') = true := by simp [hy']
      simp only [List.filter_cons, h1, ite_true, List.find?_cons]
      cases (y.id == k) <;> simp [ih]

theorem sum_insertById (c : Call) :
    ∀ l : List Call, ((insertById Call.id c l).map (·.shape.macs)).sum = c.shape.macs + (l.map (·.shape.macs)).sum
  | [] => by simp [insertById]
  | y :: ys => by
    unfold insertById
    split
    · simp
    · simp [sum_insertById c ys]
      omega

theorem macsOf_of_fwd {st : State} {k : Nat} {f : Forward} (h : fwd st k = some f) :
    macsOf st k = (f.calls.map (·.shape.macs)).sum := by
  unfold macsOf; rw [h]

theorem macsOf_update (st : State) (k k' : Nat) (g : Forward → Forward) (hg : ∀ f, (g f).id = f.id)
    (hc : ∀ f, (g f).calls = f.calls) : macsOf (st.update k g) k' = macsOf st k' := by
  unfold macsOf
  rw [find?_update st k k' g hg]
  cases fwd st k' with
  | none => rfl
  | some f => simp only [Option.map_some]; split <;> simp [hc]

theorem isSome_update (st : State) (k k' : Nat) (g : Forward → Forward) (hg : ∀ f, (g f).id = f.id) :
    (fwd (st.update k g) k').isSome = (fwd st k').isSome := by
  rw [find?_update st k k' g hg]
  cases fwd st k' <;> rfl

/-! ## The window's events so far -/

theorem getElem?_snoc {α : Type} {ps : List α} {x y : α} {j : Nat} (h : (ps ++ [x])[j]? = some y) :
    ps[j]? = some y ∨ (j = ps.length ∧ y = x) := by
  by_cases hj : j < ps.length
  · rw [List.getElem?_append_left hj] at h
    exact Or.inl h
  · rw [List.getElem?_append_right (by omega)] at h
    right
    have : j - ps.length = 0 := by
      cases hk : j - ps.length with
      | zero => rfl
      | succ n => simp [hk] at h
    rw [this] at h
    simp at h
    exact ⟨by omega, h.symm⟩

theorem getElem?_snoc_old {α : Type} {ps : List α} (x : α) {y : α} {j : Nat} (h : ps[j]? = some y) :
    (ps ++ [x])[j]? = some y := by
  have hj : j < ps.length := by
    rcases List.getElem?_eq_some_iff.mp h with ⟨hj, -⟩
    exact hj
  rw [List.getElem?_append_left hj, h]

theorem getElem?_snoc_new {α : Type} (ps : List α) (x : α) : (ps ++ [x])[ps.length]? = some x := by
  simp

theorem sum_drop_snoc (ps : List (Event × CallAnswer)) (x : Event × CallAnswer) {y : Event × CallAnswer} {j : Nat}
    (h : ps[j]? = some y) :
    (((ps ++ [x]).drop (j + 1)).map countedBits).sum = ((ps.drop (j + 1)).map countedBits).sum + countedBits x := by
  have hj : j < ps.length := by
    rcases List.getElem?_eq_some_iff.mp h with ⟨hj, -⟩
    exact hj
  rw [List.drop_append_of_le_length (by omega)]
  simp

theorem sum_drop_snoc_new (ps : List (Event × CallAnswer)) (x : Event × CallAnswer) :
    (((ps ++ [x]).drop (ps.length + 1)).map countedBits).sum = 0 := by
  simp

theorem sum_snoc (ps : List (Event × CallAnswer)) (x : Event × CallAnswer) (g : Event × CallAnswer → Nat) :
    ((ps ++ [x]).map g).sum = (ps.map g).sum + g x := by
  simp

/-! ## The invariant -/

/-- What holds after the window's events `ps`, beside their answers, at the certifier's state `st`: each forward's
calls' MACs are those of the calls of it the certifier took, a forward of a call it took is in the state, and the open
forward is unrefused for its bits, took a call, and its count is at least the bits of the buckets counted since any call
of it. -/
structure Inv (ps : List (Event × CallAnswer)) (st : State) : Prop where
  macs : ∀ k : Nat, (ps.map (callMacs k)).sum = macsOf st k
  took : ∀ (k j : Nat) (x : Event × CallAnswer), ps[j]? = some x → TookCall k x → (fwd st k).isSome
  isOpen : ∀ k : Nat, st.isOpen = some k → ∃ f, fwd st k = some f ∧ f.verdict ≠ some .accepted ∧
    (∃ (j : Nat) (x : Event × CallAnswer), ps[j]? = some x ∧ TookCall k x) ∧
    ∀ (j : Nat) (x : Event × CallAnswer), ps[j]? = some x → TookCall k x →
      ((ps.drop (j + 1)).map countedBits).sum ≤ f.count

theorem inv_init (e : List Nat) : Inv [] (State.init e) where
  macs k := by simp [macsOf, fwd, State.init]
  took k j x h := by simp at h
  isOpen k h := by simp [State.init] at h

theorem callMacs_refusal (k : Nat) (ev : Event) : callMacs k (ev, .refusal) = 0 := by
  cases ev <;> simp [callMacs, refused]

theorem countedBits_refusal (ev : Event) : countedBits (ev, .refusal) = 0 := by
  cases ev <;> simp [countedBits, refused]

theorem not_tookCall_refusal (k : Nat) (ev : Event) : ¬TookCall k (ev, .refusal) := by
  rintro ⟨c, root, s, -, h⟩
  simp [refused] at h

/-- A refused event leaves the state as it was. -/
theorem step_refused (cap : Cap) (rb : Nat) (st : State) (ev : Event)
    (h : refused (step cap rb st ev).2 = true) : step cap rb st ev = (st, .refusal) := by
  cases ev with
  | aRoot forward call root shape =>
    simp only [step] at h ⊢
    split
    · rfl
    · rename_i hc
      rw [ite_eq_right hc] at h
      cases hf : NetworkCertifier.Calls.flip shape st.entropy with
      | none => simp [hf, refused] at h
      | some p => obtain ⟨c, e⟩ := p; simp [hf, refused] at h
  | bucket r frames =>
    simp only [step] at h ⊢
    split
    · rfl
    · rename_i hc
      rw [ite_eq_right hc] at h
      cases ho : st.isOpen <;> simp [ho, refused] at h
  | wordsRoot forward root =>
    simp only [step] at h ⊢
    split
    · rfl
    · rename_i hc
      rw [ite_eq_right hc] at h
      cases hf : st.forwards.find? (·.id == forward) with
      | none => rfl
      | some f => simp [hf, refused] at h
  | close =>
    simp only [step] at h ⊢
    split
    · rfl
    · rename_i hc
      rw [ite_eq_right hc] at h
      simp [refused] at h

theorem inv_refused {ps : List (Event × CallAnswer)} {st : State} (ev : Event) (hi : Inv ps st) :
    Inv (ps ++ [(ev, .refusal)]) st where
  macs k := by rw [sum_snoc, hi.macs k, callMacs_refusal]; rfl
  took k j x hx ht := by
    rcases getElem?_snoc hx with hx | ⟨-, rfl⟩
    · exact hi.took k j x hx ht
    · exact absurd ht (not_tookCall_refusal k ev)
  isOpen k hk := by
    obtain ⟨f, hf, hv, ⟨j, x, hx, ht⟩, hb⟩ := hi.isOpen k hk
    refine ⟨f, hf, hv, ⟨j, x, getElem?_snoc_old _ hx, ht⟩, fun j' y hy ht' => ?_⟩
    rcases getElem?_snoc hy with hy' | ⟨-, rfl⟩
    · rw [sum_drop_snoc _ _ hy', countedBits_refusal]
      exact hb j' y hy' ht'
    · exact absurd ht' (not_tookCall_refusal k ev)

/-! ## What a step the certifier takes does -/

theorem fwd_opened (st : State) (forward : Nat) (f : Forward) (hf : f.id = forward) (k : Nat) :
    fwd { st with forwards := insertById Forward.id f (st.forwards.filter (·.id != forward)),
                  isOpen := some forward } k = if k = forward then some f else fwd st k := by
  unfold fwd
  by_cases hk : k = forward
  · subst hk
    simp only [ite_true]
    rw [← hf]
    apply find?_insertById_self
    intro y hy
    rw [List.mem_filter] at hy
    simpa [hf] using hy.2
  · simp only [hk, ite_false]
    rw [find?_insertById_ne f k (by omega), find?_filter_ne k forward hk]

/-- An A root the certifier takes opens its forward, or finds it open; gives the forward the call; keeps its count and,
unless the coins fail, its verdict; and leaves the other forwards as they were. -/
theorem step_aRoot {cap : Cap} {rb : Nat} {st st' : State} {forward call : Nat} {root : List Nat} {shape : Shape}
    {a : CallAnswer} (hs : step cap rb st (.aRoot forward call root shape) = (st', a)) (ha : refused a = false) :
    (st.isOpen = some forward ∨ (st.isOpen = none ∧ fwd st forward = none)) ∧ st'.isOpen = some forward ∧
      (∀ k, k ≠ forward → fwd st' k = fwd st k) ∧
      ∃ f', fwd st' forward = some f' ∧ (f'.calls.map (·.shape.macs)).sum = shape.macs + macsOf st forward ∧
        f'.count = ((fwd st forward).map (·.count)).getD 0 ∧
        (∀ f, fwd st forward = some f → f.verdict ≠ some .accepted → f'.verdict ≠ some .accepted) ∧
        (fwd st forward = none → f'.verdict ≠ some .accepted) := by
  simp only [step] at hs
  split at hs
  · simp only [Prod.mk.injEq] at hs
    obtain ⟨-, rfl⟩ := hs
    simp [refused] at ha
  rename_i hc
  have hopen : st.isOpen = some forward ∨ (st.isOpen = none ∧ fwd st forward = none) := by
    simp only [Bool.or_eq_true, not_or, Bool.not_eq_true'] at hc
    obtain ⟨⟨⟨⟨-, -⟩, -⟩, ho⟩, -⟩ := hc
    have ho' : (st.isOpen == some forward ||
        (st.isOpen.isNone && !st.forwards.any fun x => x.id == forward)) = true := by
      cases hb : (st.isOpen == some forward || (st.isOpen.isNone && !st.forwards.any fun x => x.id == forward)) with
      | false => exact absurd hb ho
      | true => rfl
    simp only [Bool.or_eq_true, beq_iff_eq, Bool.and_eq_true, Option.isNone_iff_eq_none, Bool.not_eq_true',
      List.any_eq_false] at ho'
    rcases ho' with ho | ⟨ho, hn⟩
    · exact Or.inl ho
    · refine Or.inr ⟨ho, ?_⟩
      unfold fwd
      rw [List.find?_eq_none]
      intro x hx
      simpa using hn x hx
  generalize hf0 : (st.forwards.find? (·.id == forward)).getD ⟨forward, [], 0, none, none⟩ = f0 at hs
  have hf0' : fwd st forward = none → f0 = ⟨forward, [], 0, none, none⟩ := by
    intro h; unfold fwd at h; rw [h] at hf0; exact hf0.symm
  have hf0'' : ∀ f, fwd st forward = some f → f0 = f := by
    intro f h; unfold fwd at h; rw [h] at hf0; exact hf0.symm
  have hid : f0.id = forward := by
    cases h : fwd st forward with
    | none => rw [hf0' h]
    | some f =>
      rw [hf0'' f h]
      have := List.find?_some h
      simpa using this
  have hmacs0 : (f0.calls.map (·.shape.macs)).sum = macsOf st forward := by
    unfold macsOf
    cases h : fwd st forward with
    | none => rw [hf0' h]; rfl
    | some f => rw [hf0'' f h]
  have hcount0 : f0.count = ((fwd st forward).map (·.count)).getD 0 := by
    cases h : fwd st forward with
    | none => rw [hf0' h]; rfl
    | some f => rw [hf0'' f h]; rfl
  have hv0 : (∀ f, fwd st forward = some f → f.verdict ≠ some .accepted → f0.verdict ≠ some .accepted) ∧
      (fwd st forward = none → f0.verdict = none) := by
    refine ⟨fun f h hv => ?_, fun h => ?_⟩
    · rw [hf0'' f h]; exact hv
    · rw [hf0' h]
  generalize hf1 : ({ f0 with calls := insertById Call.id ⟨call, shape, root⟩ f0.calls } : Forward) = f1 at hs
  have hid1 : f1.id = forward := by rw [← hf1]; exact hid
  have hmacs1 : (f1.calls.map (·.shape.macs)).sum = shape.macs + macsOf st forward := by
    rw [← hf1, sum_insertById, hmacs0]
  have hcount1 : f1.count = ((fwd st forward).map (·.count)).getD 0 := by rw [← hf1]; exact hcount0
  have hv1 : f1.verdict = f0.verdict := by rw [← hf1]
  have hopened := fwd_opened st forward f1 hid1
  refine ⟨hopen, ?_⟩
  cases hfl : NetworkCertifier.Calls.flip shape st.entropy with
  | some p =>
    obtain ⟨c, e⟩ := p
    simp only [hfl, Prod.mk.injEq] at hs
    obtain ⟨rfl, rfl⟩ := hs
    refine ⟨rfl, fun k hk => ?_, f1, ?_, hmacs1, hcount1, fun f h hv => ?_, fun h => ?_⟩
    · have := hopened k
      simp only [hk, ite_false] at this
      exact this
    · have := hopened forward
      simp only [ite_true] at this
      exact this
    · rw [hv1]; exact hv0.1 f h hv
    · rw [hv1, hv0.2 h]; simp
  | none =>
    simp only [hfl, Prod.mk.injEq] at hs
    obtain ⟨rfl, rfl⟩ := hs
    have hr : ∀ f : Forward, (Forward.refuse .noCoins f).id = f.id := fun _ => rfl
    refine ⟨rfl, fun k hk => ?_, Forward.refuse .noCoins f1, ?_, hmacs1, hcount1, fun f h hv => ?_, fun h => ?_⟩
    · show fwd (State.update _ forward _) k = _
      rw [fwd_update_ne _ forward k _ hr hk]
      have := hopened k
      simp only [hk, ite_false] at this
      exact this
    · show fwd (State.update _ forward _) forward = _
      rw [fwd_update_self _ forward _ hr]
      have := hopened forward
      simp only [ite_true] at this
      rw [this]; rfl
    · simp only [Forward.refuse, hv1]
      cases hv' : f0.verdict with
      | none => simp
      | some v =>
        have := hv0.1 f h hv
        rw [hv'] at this
        simpa using this
    · simp only [Forward.refuse, hv1, hv0.2 h]
      simp

/-- A bucket the certifier takes is counted: its bits go to the open forward's count, and nothing else the invariant
reads changes. -/
theorem step_bucket {cap : Cap} {rb : Nat} {st st' : State} {r : Nat} {frames : List Nat} {a : CallAnswer}
    (hs : step cap rb st (.bucket r frames) = (st', a)) (ha : refused a = false) :
    a = .counted ∧ st'.isOpen = st.isOpen ∧ (∀ k, macsOf st' k = macsOf st k) ∧
      (∀ k, (fwd st' k).isSome = (fwd st k).isSome) ∧
      ∀ k f, st.isOpen = some k → fwd st k = some f →
        ∃ f', fwd st' k = some f' ∧ f'.verdict = f.verdict ∧ f'.count = f.count + (timingBits r + frames.sum) := by
  simp only [step] at hs
  split at hs
  · simp only [Prod.mk.injEq] at hs
    obtain ⟨-, rfl⟩ := hs
    simp [refused] at ha
  cases ho : st.isOpen with
  | none =>
    simp only [ho, Prod.mk.injEq] at hs
    obtain ⟨rfl, rfl⟩ := hs
    exact ⟨rfl, ho, fun _ => rfl, fun _ => rfl, fun k f h => by simp at h⟩
  | some k' =>
    simp only [ho, Prod.mk.injEq] at hs
    obtain ⟨rfl, rfl⟩ := hs
    have hg : ∀ f : Forward, ({ f with count := f.count + timingBits r + frames.sum } : Forward).id = f.id :=
      fun _ => rfl
    refine ⟨rfl, ho, fun k => macsOf_update _ _ _ _ hg fun _ => rfl, fun k => isSome_update _ _ _ _ hg,
      fun k f hk hf => ?_⟩
    obtain rfl : k' = k := by simpa using hk
    rw [fwd_update_self _ _ _ hg, hf]
    exact ⟨_, rfl, rfl, by simp only [Nat.add_assoc]⟩

/-- A words root the certifier takes closes its forward and changes no forward's calls. -/
theorem step_wordsRoot {cap : Cap} {rb : Nat} {st st' : State} {forward : Nat} {root : List Nat} {a : CallAnswer}
    (hs : step cap rb st (.wordsRoot forward root) = (st', a)) (ha : refused a = false) :
    st'.isOpen = none ∧ (∀ k, macsOf st' k = macsOf st k) ∧ (∀ k, (fwd st' k).isSome = (fwd st k).isSome) := by
  simp only [step] at hs
  split at hs
  · simp only [Prod.mk.injEq] at hs
    obtain ⟨-, rfl⟩ := hs
    simp [refused] at ha
  cases hf : st.forwards.find? (·.id == forward) with
  | none =>
    simp only [hf, Prod.mk.injEq] at hs
    obtain ⟨-, rfl⟩ := hs
    simp [refused] at ha
  | some f =>
    simp only [hf, Prod.mk.injEq] at hs
    obtain ⟨rfl, rfl⟩ := hs
    refine ⟨rfl, fun k => ?_, fun k => ?_⟩
    · exact macsOf_update st forward k _ (fun _ => rfl) (fun _ => rfl)
    · exact isSome_update st forward k _ (fun _ => rfl)

/-- The window's close, when the certifier takes it, closes every forward and changes no forward's calls. -/
theorem step_close {cap : Cap} {rb : Nat} {st st' : State} {a : CallAnswer}
    (hs : step cap rb st .close = (st', a)) (ha : refused a = false) :
    st'.isOpen = none ∧ (∀ k, macsOf st' k = macsOf st k) ∧ (∀ k, (fwd st' k).isSome = (fwd st k).isSome) := by
  simp only [step] at hs
  split at hs
  · simp only [Prod.mk.injEq] at hs
    obtain ⟨-, rfl⟩ := hs
    simp [refused] at ha
  cases ho : st.isOpen with
  | none =>
    simp only [ho, Prod.mk.injEq] at hs
    obtain ⟨rfl, rfl⟩ := hs
    exact ⟨rfl, fun _ => rfl, fun _ => rfl⟩
  | some k' =>
    simp only [ho, Prod.mk.injEq] at hs
    obtain ⟨rfl, rfl⟩ := hs
    refine ⟨rfl, fun k => ?_, fun k => ?_⟩
    · exact macsOf_update st k' k _ (fun _ => rfl) (fun _ => rfl)
    · exact isSome_update st k' k _ (fun _ => rfl)

/-! ## The invariant through the window -/

/-- An event that brings no call and leaves no forward open keeps the invariant. -/
theorem inv_keep {ps : List (Event × CallAnswer)} {st st' : State} {x : Event × CallAnswer} (hi : Inv ps st)
    (ho : st'.isOpen = none) (hm : ∀ k, macsOf st' k = macsOf st k)
    (hs : ∀ k, (fwd st' k).isSome = (fwd st k).isSome) (hc : ∀ k, callMacs k x = 0) (ht : ∀ k, ¬TookCall k x) :
    Inv (ps ++ [x]) st' where
  macs k := by rw [sum_snoc, hi.macs k, hc k, hm k]; rfl
  took k j y hy ht' := by
    rcases getElem?_snoc hy with hy | ⟨-, rfl⟩
    · rw [hs k]; exact hi.took k j y hy ht'
    · exact absurd ht' (ht k)
  isOpen k hk := by rw [ho] at hk; cases hk

theorem tookCall_aRoot {k forward call : Nat} {root : List Nat} {shape : Shape} {a : CallAnswer}
    (h : TookCall k (.aRoot forward call root shape, a)) : k = forward := by
  obtain ⟨c, r, s, he, -⟩ := h
  cases he
  rfl

theorem not_tookCall_of {k : Nat} {x : Event × CallAnswer} (h : ∀ f c r s, x.1 ≠ .aRoot f c r s) : ¬TookCall k x := by
  rintro ⟨c, r, s, he, -⟩
  exact h _ _ _ _ he

/-- **One step keeps the invariant.** -/
theorem inv_step (cap : Cap) (rb : Nat) {ps : List (Event × CallAnswer)} {st : State} (ev : Event) (hi : Inv ps st) :
    Inv (ps ++ [(ev, (step cap rb st ev).2)]) (step cap rb st ev).1 := by
  cases hr : refused (step cap rb st ev).2 with
  | true =>
    rw [step_refused cap rb st ev hr]
    exact inv_refused ev hi
  | false =>
    generalize hs : step cap rb st ev = res at hr ⊢
    obtain ⟨st', a⟩ := res
    simp only at hr ⊢
    cases ev with
    | aRoot forward call root shape =>
      obtain ⟨hpre, hopen', hother, f', hf', hmacs', hcount', hv1, hv2⟩ := step_aRoot hs hr
      have hcm : ∀ k, callMacs k (.aRoot forward call root shape, a) = if forward = k then shape.macs else 0 := by
        intro k; simp [callMacs, hr]
      have hstill : ∀ f, fwd st forward = some f → st.isOpen = some forward := by
        intro f h
        rcases hpre with h' | ⟨-, h'⟩
        · exact h'
        · rw [h'] at h; cases h
      refine ⟨fun k => ?_, fun k j y hy ht => ?_, fun k hk => ?_⟩
      · rw [sum_snoc, hi.macs k, hcm k]
        by_cases hk : k = forward
        · subst hk
          rw [macsOf_of_fwd hf', hmacs']
          simp only [ite_true]
          omega
        · have : forward ≠ k := fun h => hk h.symm
          simp only [this, ite_false, Nat.add_zero]
          unfold macsOf
          rw [hother k hk]
      · rcases getElem?_snoc hy with hy | ⟨-, rfl⟩
        · by_cases hk : k = forward
          · subst hk; rw [hf']; rfl
          · rw [hother k hk]; exact hi.took k j y hy ht
        · rw [tookCall_aRoot ht, hf']; rfl
      · rw [hopen'] at hk
        obtain rfl : forward = k := by simpa using hk
        refine ⟨f', hf', ?_, ⟨ps.length, _, getElem?_snoc_new _ _, ⟨call, root, shape, rfl, hr⟩⟩, fun j y hy ht => ?_⟩
        · cases h0 : fwd st forward with
          | none => exact hv2 h0
          | some f =>
            obtain ⟨f₂, hf₂, hv₂, -⟩ := hi.isOpen forward (hstill f h0)
            rw [h0] at hf₂
            cases hf₂
            exact hv1 f h0 hv₂
        · rcases getElem?_snoc hy with hy' | ⟨rfl, rfl⟩
          · have hsome := hi.took forward j y hy' ht
            cases h0 : fwd st forward with
            | none => rw [h0] at hsome; cases hsome
            | some f =>
              obtain ⟨f₂, hf₂, -, -, hb⟩ := hi.isOpen forward (hstill f h0)
              rw [h0] at hf₂
              cases hf₂
              rw [sum_drop_snoc _ _ hy', hcount', h0]
              have : countedBits (.aRoot forward call root shape, a) = 0 := rfl
              rw [this, Nat.add_zero]
              exact hb j y hy' ht
          · rw [sum_drop_snoc_new]; exact Nat.zero_le _
    | bucket r frames =>
      obtain ⟨rfl, hopen', hm, hsome, hcount⟩ := step_bucket hs hr
      have hcb : countedBits (.bucket r frames, .counted) = timingBits r + frames.sum := rfl
      have hnt : ∀ k, ¬TookCall k (.bucket r frames, .counted) :=
        fun k => not_tookCall_of fun _ _ _ _ h => by cases h
      refine ⟨fun k => ?_, fun k j y hy ht => ?_, fun k hk => ?_⟩
      · rw [sum_snoc, hi.macs k, hm k]; rfl
      · rcases getElem?_snoc hy with hy | ⟨-, rfl⟩
        · rw [hsome k]; exact hi.took k j y hy ht
        · exact absurd ht (hnt k)
      · rw [hopen'] at hk
        obtain ⟨f, hf, hv, ⟨j, x, hx, ht⟩, hb⟩ := hi.isOpen k hk
        obtain ⟨f', hf', hv', hc'⟩ := hcount k f hk hf
        refine ⟨f', hf', by rw [hv']; exact hv, ⟨j, x, getElem?_snoc_old _ hx, ht⟩, fun j' y hy ht' => ?_⟩
        rcases getElem?_snoc hy with hy' | ⟨-, rfl⟩
        · rw [sum_drop_snoc _ _ hy', hcb, hc']
          have := hb j' y hy' ht'
          omega
        · exact absurd ht' (hnt k)
    | wordsRoot forward root =>
      obtain ⟨ho, hm, hsome⟩ := step_wordsRoot hs hr
      exact inv_keep hi ho hm hsome (fun k => rfl) fun k => not_tookCall_of fun _ _ _ _ h => by cases h
    | close =>
      obtain ⟨ho, hm, hsome⟩ := step_close hs hr
      exact inv_keep hi ho hm hsome (fun k => rfl) fun k => not_tookCall_of fun _ _ _ _ h => by cases h

/-- **The invariant holds through the window.** -/
theorem inv_run (cap : Cap) (rb : Nat) (evs : List Event) :
    ∀ (ps : List (Event × CallAnswer)) (st : State), Inv ps st →
      Inv (ps ++ evs.zip (run cap rb st evs).1) (run cap rb st evs).2 := by
  induction evs with
  | nil => intro ps st h; simpa [run_nil] using h
  | cons ev evs ih =>
    intro ps st h
    have := ih _ _ (inv_step cap rb ev h)
    rw [run_cons]
    simpa using this

/-- A words root the certifier accepts is the open forward's, and the forward's verdict, from its count with the
release's bits against the cap, is accepted. -/
theorem step_wordsRoot_accepted {cap : Cap} {rb : Nat} {st : State} {k : Nat} {root : List Nat}
    (h : (step cap rb st (.wordsRoot k root)).2 = .verdict .accepted) :
    st.isOpen = some k ∧ ∃ f, fwd st k = some f ∧
      f.verdict.getD (if cap.passed (f.calls.map (·.shape.macs)).sum (f.count + rb) then .pastBits else .accepted) =
        .accepted := by
  generalize hs : step cap rb st (.wordsRoot k root) = res at h
  obtain ⟨st', a⟩ := res
  simp only at h
  subst h
  simp only [step] at hs
  split at hs
  · simp only [Prod.mk.injEq] at hs
    exact absurd hs.2 (by intro h; cases h)
  rename_i hc
  have ho : st.isOpen = some k := by
    cases hb : (st.isOpen != some k) with
    | true => simp [hb] at hc
    | false => simpa using hb
  refine ⟨ho, ?_⟩
  cases hf : st.forwards.find? (·.id == k) with
  | none =>
    simp only [hf, Prod.mk.injEq] at hs
    exact absurd hs.2 (by intro h; cases h)
  | some f =>
    simp only [hf, Prod.mk.injEq] at hs
    obtain ⟨-, hv⟩ := hs
    injection hv with hv
    exact ⟨f, hf, hv⟩

end CallsProofs

open CallsProofs

/-- **A forward the certifier accepts at its words root stayed within its bits.** -/
theorem refusesPastCallBits : RefusesPastCallBits := by
  intro cap rb e evs i k root h
  have hx := h
  rw [List.getElem?_zip_eq_some] at hx
  obtain ⟨hev, hans⟩ := hx
  simp only at hev hans
  rw [run_getElem?, hev] at hans
  simp only [Option.map_some, Option.some.injEq] at hans
  have hi : i < evs.length := by
    rcases List.getElem?_eq_some_iff.mp hev with ⟨hi, -⟩
    exact hi
  have htake : (evs.zip (run cap rb (State.init e) evs).1).take i =
      (evs.take i).zip (run cap rb (State.init e) (evs.take i)).1 := by
    rw [← run_take]
    exact take_zip _ _ _
  have hinv := inv_run cap rb (evs.take i) [] (State.init e) (inv_init e)
  rw [List.nil_append, ← htake] at hinv
  generalize (evs.zip (run cap rb (State.init e) evs).1) = xs at hinv ⊢
  generalize (run cap rb (State.init e) (evs.take i)).2 = st at hinv hans
  obtain ⟨ho, f, hf, hv⟩ := step_wordsRoot_accepted hans
  obtain ⟨f₂, hf₂, hv₂, ⟨j, x, hx, ht⟩, hb⟩ := hinv.isOpen k ho
  rw [hf] at hf₂
  cases hf₂
  have hm := hinv.macs k
  rw [macsOf_of_fwd hf] at hm
  have hle : (f.count + rb) * cap.fwdMacs ≤ cap.cap * (f.calls.map (·.shape.macs)).sum := by
    cases hfv : f.verdict with
    | some v =>
      rw [hfv] at hv hv₂
      simp only [Option.getD_some] at hv
      exact absurd (by rw [hv]) hv₂
    | none =>
      rw [hfv] at hv
      simp only [Option.getD_none] at hv
      cases hp : cap.passed (f.calls.map (·.shape.macs)).sum (f.count + rb) with
      | true => rw [hp] at hv; cases hv
      | false =>
        unfold Cap.passed at hp
        have := of_decide_eq_false hp
        omega
  have hj : j < i := by
    rcases List.getElem?_eq_some_iff.mp hx with ⟨hj, -⟩
    have := List.length_take_le i xs
    omega
  refine ⟨⟨j, hj, x, ?_, ht⟩, fun j' hj' y hy ht' => ?_⟩
  · rw [List.getElem?_take_of_lt hj] at hx
    exact hx
  · rw [← List.getElem?_take_of_lt hj'] at hy
    have := hb j' y hy ht'
    rw [hm]
    calc ((((xs.take i).drop (j' + 1)).map countedBits).sum + rb) * cap.fwdMacs
        ≤ (f.count + rb) * cap.fwdMacs := Nat.mul_le_mul_right _ (by omega)
      _ ≤ _ := hle

/-! ## Its witnesses -/

/-- `RefusesPastCallBits`' honest witness: the certifier accepts `honestEvents`' forward at its words root, at the cap,
with the honest release's bits. -/
theorem refusesPastCallBits_honest :
    (honestEvents.zip (run witnessCap honestReleaseBits (State.init honestEntropy) honestEvents).1)[2]? =
      some (.wordsRoot 0 witnessRoot, .verdict .accepted) := by
  rfl

/-- `RefusesPastCallBits`' nontrivial witness: a certifier taking every event (`acceptsAll`) accepts `pastEvents`'
forward, whose 1,001 bits pass its cap of 32, so its acceptance is not within its bits. -/
theorem refusesPastCallBits_nontrivial : ¬WithinBits witnessCap 0 (acceptsAll pastEvents) 2 0 := by
  rintro ⟨-, h⟩
  have := h 0 (by decide) _ rfl ⟨0, witnessRoot, witnessShape, rfl, rfl⟩
  revert this
  decide

end Properties.NetworkCertifier
