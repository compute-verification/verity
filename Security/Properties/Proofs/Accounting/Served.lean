import Properties.Accounting.Served

/-!
# Proofs of the served tie's statements (`Accounting.Served`, `Properties.Served`)

`TieCommits` goes class by class: a registered value's commit coin is `0`; an A row's is its call's round, which is its
first reader's, since only its own call reads it (`firstRead_cases`), or below the draw when nothing does
(`round_lt`); an admit word's is its call's round, the floor of its cut; a receipt's or `z`'s is the round after its
own, which its coin read reaches (`Tie.noised`); an inner value's is its cut. `TieLive`: a unit with a word has a
first gate, its round is a round of the window, and every one is live. `SeededScheduleNotLive`: `UnitsOfLayout` puts a
read run input at the draw, `LayerCommitted` at its first reader's round, which `Wf` puts below it.
`Llama31Shape` is a kernel computation.
-/

namespace Properties.Served

open Accounting.Compose Accounting.Served

theorem llama31Shape : Llama31Shape := by
  unfold Llama31Shape
  refine ⟨by decide +kernel, by decide +kernel, by decide +kernel, ?_⟩
  unfold Layout.Wf
  decide +kernel

theorem stepOf_lt {L : Layout} {c : ℕ} (hc : c < L.steps * L.calls) : L.stepOf c < L.steps := by
  unfold Layout.stepOf
  exact Nat.div_lt_of_lt_mul (by rwa [Nat.mul_comm] at hc)

theorem linearOf_lt {L : Layout} (hL : L.Wf) (c : ℕ) : L.linearOf c < L.calls := by
  obtain ⟨_, hcalls, _⟩ := hL
  exact Nat.mod_lt _ hcalls

theorem roundInWindow : RoundInWindow := by
  intro L F s w _ hs hw
  cases F with
  | beside => exact hs
  | served =>
    show s * L.calls + w < L.steps * L.calls
    calc s * L.calls + w < s * L.calls + L.calls := Nat.add_lt_add_left hw _
      _ = (s + 1) * L.calls := by ring
      _ ≤ L.steps * L.calls := Nat.mul_le_mul_right _ (Nat.succ_le_of_lt hs)

/-- A tied circuit's gate is in a round of the window. -/
theorem round_lt {ι : Type} {L : Layout} {F : Form} {Cd : Declared ι} {cls : ℕ → Cls} (hL : L.Wf)
    (T : Tie L F Cd cls) (i : ℕ) (hi : i < Cd.dag.length) : Cd.round (Cd.call i) < Cd.rounds := by
  have hc := T.inRange i hi
  have hc' : Cd.call i < L.steps * L.calls := T.calls ▸ hc
  rw [T.round _ hc, T.rounds]
  exact roundInWindow L F _ _ hL (stepOf_lt hc') (linearOf_lt hL _)

/-- A value's first reader: none (the draw), or a gate that reads it. -/
theorem firstRead_cases {ι : Type} (Cd : Declared ι) (i : ℕ) :
    Cd.firstRead i = Cd.rounds ∨
      ∃ j, j < Cd.dag.length ∧ Cd.Reads j i ∧ Cd.firstRead i = Cd.round (Cd.call j) := by
  unfold Declared.firstRead
  cases h : List.find? (fun j => (Cd.dag[j]?).any fun g => g.args.contains i) (List.range Cd.dag.length) with
  | none => exact Or.inl rfl
  | some j => exact Or.inr ⟨j, List.mem_range.1 (List.mem_of_find?_eq_some h), List.find?_some h, rfl⟩

/-- A value some gate reads has a first reader, a gate that reads it. -/
theorem firstRead_of_reads {ι : Type} (Cd : Declared ι) {i j : ℕ} (hj : j < Cd.dag.length) (hR : Cd.Reads j i) :
    ∃ j', j' < Cd.dag.length ∧ Cd.Reads j' i ∧ Cd.firstRead i = Cd.round (Cd.call j') := by
  unfold Declared.firstRead
  cases h : List.find? (fun j => (Cd.dag[j]?).any fun g => g.args.contains i) (List.range Cd.dag.length) with
  | none => exact absurd hR (List.find?_eq_none.1 h j (List.mem_range.2 hj))
  | some j' => exact ⟨j', List.mem_range.1 (List.mem_of_find?_eq_some h), List.find?_some h, rfl⟩

theorem tieCommits : TieCommits := by
  intro ι L F Cd cls hL T ℓ hW
  obtain ⟨hlt, hnc⟩ := hW
  have hc : Cd.call ℓ < Cd.calls := T.inRange ℓ hlt
  have hround := T.round _ hc
  have hin := round_lt hL T ℓ hlt
  cases hk : cls ℓ with
  | coin => exact absurd ((T.coin ℓ hlt).1 hk) hnc
  | inner => exact le_of_eq (T.inner ℓ hlt hk)
  | registered =>
    rw [T.commit ℓ hlt (by simp [hk]) (by simp [hk]), hk]
    simp [commitCoin]
  | act =>
    obtain ⟨hb, hr⟩ := T.act ℓ hlt hk
    rw [T.commit ℓ hlt (by simp [hk]) (by simp [hk]), hk]
    simp only [commitCoin, Declared.cutOf, hb]
    rw [← hround]
    rcases firstRead_cases Cd ℓ with h | ⟨j, hj, hR, h⟩
    · rw [h]
      exact hin.le
    · rw [h]
      exact le_of_eq (congrArg Cd.round (hr j hj hR)).symm
  | admit =>
    have hb := T.computed ℓ hlt (by simp [hk])
    rw [T.commit ℓ hlt (by simp [hk]) (by simp [hk]), hk]
    simp only [commitCoin, Declared.cutOf, hb]
    rw [← hround]
    exact le_max_left _ _
  | receipt =>
    have hb := T.computed ℓ hlt (by simp [hk])
    have hn := T.noised ℓ hlt (by simp [hk])
    rw [T.commit ℓ hlt (by simp [hk]) (by simp [hk]), hk]
    simp only [commitCoin, Declared.cutOf, hb]
    rw [← hround]
    exact le_max_of_le_right hn
  | out =>
    have hb := T.computed ℓ hlt (by simp [hk])
    have hn := T.noised ℓ hlt (by simp [hk])
    rw [T.commit ℓ hlt (by simp [hk]) (by simp [hk]), hk]
    simp only [commitCoin, Declared.cutOf, hb]
    rw [← hround]
    exact le_max_of_le_right hn

theorem tieLive : TieLive := by
  intro ι K Sc L F Cd cls hL T u _ hw
  obtain ⟨i, hi, _⟩ := List.any_eq_true.1 hw
  obtain ⟨hiw, hiu⟩ := List.mem_filter.1 hi
  obtain ⟨hir, _⟩ := List.mem_filter.1 hiw
  have hmem : i ∈ unitGates Cd u := List.mem_filter.2 ⟨hir, hiu⟩
  unfold Declared.roundOf
  cases hl : unitGates Cd u with
  | nil =>
    rw [hl] at hmem
    simp at hmem
  | cons i0 rest =>
    have h0 : i0 ∈ unitGates Cd u := by rw [hl]; simp
    obtain ⟨h0r, _⟩ := List.mem_filter.1 h0
    show Cd.coinMode (Cd.round (Cd.call i0)) = .live
    exact T.live _ (round_lt hL T i0 (List.mem_range.1 h0r))

theorem seededScheduleNotLive : SeededScheduleNotLive := by
  intro ι Cd Np Kd p query l hl hU hWf ⟨ℓ, j, hb, hj, hR⟩ hLC
  obtain ⟨_, _, _, hd, _, _⟩ := hU l hl
  obtain ⟨hW1, hW2, _⟩ := hWf
  have hlt : ℓ < Cd.dag.length := by
    by_contra hge
    have hnone : Cd.dag[ℓ]? = none := List.getElem?_eq_none (by omega)
    simp [Declared.binding, hnone] at hb
  have hW : Cd.Writes ℓ := ⟨hlt, by rw [hb]; simp⟩
  have h1 : Cd.commitRound ℓ = Cd.rounds := by simpa [hb] using hd ℓ hW
  have h2 := hLC ℓ hW
  have h3 : Cd.cutOf ℓ = Cd.firstRead ℓ := by simp [Declared.cutOf, hb]
  obtain ⟨j', hj', _, h⟩ := firstRead_of_reads Cd hj hR
  have hr := hW2 _ (hW1 j' hj').1
  omega

end Properties.Served
