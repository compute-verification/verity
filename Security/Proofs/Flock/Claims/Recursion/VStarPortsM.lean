import Proofs.Flock.Claims.Recursion.VStarPorts

/-!
# Where the stager places the message rounds' rows (an algebra part's `m<i>` ports)

`VStar.In.algebraIn` gives each of the part's ports, in order, its input: `acc0` and `acc1` the running sums' rows, and
a port `m<i>` the stager's message round `i` (`Src.msgs`), both reps' rows, rep `r`'s at instance `r`; then `v` and
`out`. So port `m<i>`'s instance `r < 2` reads round `i`'s rep-`r` row (`portRow_algebra`), whichever of the part's
ports named `m<i>` `portRow` finds first.
-/

namespace RecDeploy.VStarBuild

open _root_.Flock
open Claims (Msg)

/-- **A loop that pushes one named entry a step**: its result's names are the start's, then each element's, and each
entry it pushed is some element's, named by it, under `Q`. -/
theorem forIn_push {α β : Type} (g : α → String) (Q : α → β → Prop)
    {f : α → Array (String × β) → Except String (ForInStep (Array (String × β)))}
    (hf : ∀ a b s, f a b = .ok s → ∃ y, s = .yield (b.push (g a, y)) ∧ Q a y) :
    ∀ (l : List α) (b v : Array (String × β)), forIn l b f = .ok v →
      v.toList.map Prod.fst = b.toList.map Prod.fst ++ l.map g ∧
        ∀ e ∈ v.toList, e ∈ b.toList ∨ ∃ a ∈ l, e.1 = g a ∧ Q a e.2
  | [], b, v, h => by
    simp only [List.forIn_nil, pure, Except.pure, Except.ok.injEq] at h
    subst h
    exact ⟨by simp, fun e he => Or.inl he⟩
  | a :: l, b, v, h => by
    rw [List.forIn_cons] at h
    simp only [bind, Except.bind] at h
    split at h
    · cases h
    · rename_i s hs
      obtain ⟨y, rfl, hy⟩ := hf a b s hs
      obtain ⟨h1, h2⟩ := forIn_push g Q hf l (b.push (g a, y)) v h
      refine ⟨by rw [h1]; simp, fun e he => ?_⟩
      rcases h2 e he with he' | ⟨a', ha', he1, he2⟩
      · simp only [Array.toList_push, List.mem_append, List.mem_cons, List.not_mem_nil, or_false] at he'
        rcases he' with he' | rfl
        · exact Or.inl he'
        · exact Or.inr ⟨a, List.mem_cons_self, rfl, hy⟩
      · exact Or.inr ⟨a', List.mem_cons_of_mem _ ha', he1, he2⟩

theorem mIndex_acc0 : RecVStar.mIndex "acc0" = none := by
  unfold RecVStar.mIndex
  exact ite_eq_right_iff.2 fun h => absurd h (by decide +kernel)

theorem mIndex_acc1 : RecVStar.mIndex "acc1" = none := by
  unfold RecVStar.mIndex
  exact ite_eq_right_iff.2 fun h => absurd h (by decide +kernel)

theorem mIndex_v : RecVStar.mIndex "v" = none := by
  unfold RecVStar.mIndex
  exact ite_eq_right_iff.2 fun h => absurd h (by decide +kernel)

theorem mIndex_out' : RecVStar.mIndex "out" = none := by
  unfold RecVStar.mIndex
  exact ite_eq_right_iff.2 fun h => absurd h (by decide +kernel)

/-- An entry's input, where its name is `m<i>`, is message round `i`'s two rows, rep `r`'s at instance `r`. -/
def MRow (src : _root_.VStar.In.Src) (nm : String) (y : _root_.VStar.In.PortIn) : Prop :=
  ∀ i, RecVStar.mIndex nm = some i → ∃ rs, src.msgs[i]? = some rs ∧ rs.size = 2 ∧ y = { rows := rs, refs := #[0, 1] }

/-- **An algebra part's port inputs**: every port has an entry, and each `m<i>` entry is message round `i`'s rows. -/
theorem algebraIn_m {src : _root_.VStar.In.Src} {prog : String} {ports : List (String × ℕ)} {v : List ℕ}
    {pin : _root_.VStar.In.PartIn} (h : _root_.VStar.In.algebraIn src prog ports v = .ok pin) :
    (∀ nm ∈ ports.map Prod.fst, nm ∈ pin.toList.map Prod.fst) ∧ ∀ e ∈ pin.toList, MRow src e.1 e.2 := by
  unfold _root_.VStar.In.algebraIn at h
  simp only [bind, Except.bind] at h
  split at h
  · cases h
  · split at h
    · cases h
    · split at h
      · cases h
      · rename_i out hout
        obtain ⟨hn, hm⟩ := forIn_push (fun a : String × ℕ => a.1) (fun a y => MRow src a.1 y) (fun a b s hs => by
          rcases a with ⟨nm, k⟩
          dsimp only at hs ⊢
          by_cases h0 : (nm == "acc0") = true
          · have hn : nm = "acc0" := beq_iff_eq.1 h0
            simp only [h0, ↓reduceIte, bind, Except.bind, pure, Except.pure, Except.ok.injEq] at hs
            subst hs
            refine ⟨_, rfl, fun i hi => ?_⟩
            rw [hn, mIndex_acc0] at hi
            cases hi
          · by_cases h1 : (nm == "acc1") = true
            · have hn : nm = "acc1" := beq_iff_eq.1 h1
              simp only [h0, h1, Bool.false_eq_true, ↓reduceIte, bind, Except.bind, pure, Except.pure,
                Except.ok.injEq] at hs
              subst hs
              refine ⟨_, rfl, fun i hi => ?_⟩
              rw [hn, mIndex_acc1] at hi
              cases hi
            · cases hi : RecVStar.mIndex nm with
              | none =>
                simp only [h0, h1, Bool.false_eq_true, ↓reduceIte, hi] at hs
                all_goals cases hs
              | some i =>
                cases hr : src.msgs[i]? with
                | none =>
                  simp only [h0, h1, Bool.false_eq_true, ↓reduceIte, hi, hr] at hs
                  all_goals cases hs
                | some rs =>
                  by_cases h2 : (rs.size == 2) = true
                  · simp only [h0, h1, h2, Bool.false_eq_true, Bool.not_true, ↓reduceIte, hi, hr, bind, Except.bind,
                      pure, Except.pure, Except.ok.injEq] at hs
                    subst hs
                    refine ⟨_, rfl, fun i' hi' => ?_⟩
                    rw [hi] at hi'
                    cases hi'
                    exact ⟨rs, hr, beq_iff_eq.1 h2, rfl⟩
                  · simp only [h0, h1, h2, Bool.false_eq_true, ↓reduceIte, hi, hr, bind, Except.bind] at hs
                    all_goals cases hs) ports #[] out hout
        simp only [pure, Except.pure, Except.ok.injEq] at h
        subst h
        refine ⟨fun nm hnm => ?_, fun e he => ?_⟩
        · have h1 : nm ∈ out.toList.map Prod.fst := by
            rw [hn]
            simpa using hnm
          rw [Array.toList_push, Array.toList_push, List.map_append, List.map_append]
          exact List.mem_append_left _ (List.mem_append_left _ h1)
        · simp only [Array.toList_push, List.mem_append, List.mem_cons, List.not_mem_nil, or_false] at he
          rcases he with (he | rfl) | rfl
          · rcases hm e he with he' | ⟨a, -, he1, he2⟩
            · simp at he'
            · rw [he1]
              exact he2
          · intro i hi
            rw [mIndex_v] at hi
            cases hi
          · intro i hi
            rw [mIndex_out'] at hi
            cases hi

/-- **An algebra part's port `m<i>` at instance `r < 2` reads message round `i`'s rep-`r` row.** -/
theorem portRow_algebra {src : _root_.VStar.In.Src} {prog : String} {ports : List (String × ℕ)} {v : List ℕ}
    {pin : _root_.VStar.In.PartIn} (h : _root_.VStar.In.algebraIn src prog ports v = .ok pin) {nm : String} {i : ℕ}
    (hnm : nm ∈ ports.map Prod.fst) (hi : RecVStar.mIndex nm = some i) {rep : ℕ} (hrep : rep < 2) :
    ∃ rs, src.msgs[i]? = some rs ∧ rs.size = 2 ∧ portRow pin nm rep = rs[rep]? := by
  obtain ⟨hn, hm⟩ := algebraIn_m h
  obtain ⟨e, he, hfind⟩ : ∃ e, e ∈ pin.toList ∧ pin.find? (fun e => e.1 == nm) = some e := by
    cases hf : pin.find? (fun e => e.1 == nm) with
    | none =>
      exfalso
      obtain ⟨e, he, rfl⟩ := List.mem_map.1 (hn nm hnm)
      have := Array.find?_eq_none.1 hf e (Array.mem_toList_iff.1 he)
      simp at this
    | some e => exact ⟨e, Array.mem_toList_iff.2 (Array.mem_of_find?_eq_some hf), rfl⟩
  have he1 : e.1 = nm := by simpa using Array.find?_some hfind
  obtain ⟨rs, hrs, hsz, he2⟩ := hm e he i (he1 ▸ hi)
  refine ⟨rs, hrs, hsz, ?_⟩
  obtain rfl | rfl : rep = 0 ∨ rep = 1 := by omega
  · simp [portRow, hfind, he2]
  · simp [portRow, hfind, he2]

end RecDeploy.VStarBuild
