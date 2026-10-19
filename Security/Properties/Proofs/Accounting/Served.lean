import Properties.Accounting.Served

/-!
# Proofs of the served tie's statements (`Accounting.Served`, `Properties.Served`)

`TieCommits` goes class by class: a registered value's commit coin is `0`; an A row's or a `y` row's is its round,
which is its first reader's, since only its own call reads it (`firstRead_cases`), or below the draw when nothing does
(`round_lt`); an admit word's or `zTree`'s `z` word's is its round, the floor of its cut; a receipt's or `perCall`'s
`y` word's is the round after its own, which its coin read reaches (`Tie.noised`); an inner value's is its cut.
`TieLive`: a unit with a word has a first gate, its round is a round of the window, and every one is live.
`SeededScheduleNotLive`: `UnitsOfLayout` puts a read run input at the draw, `LayerCommitted` at its first reader's
round, which `Wf` puts below it. `TieScope` is `Scope` from `TieLive`, `TieCommits` and its three hypotheses, and
`epochScope` each window's of an `EpochTie`. `CoinsFresh` is not vacuous: one salt an epoch fails it over two windows
(`oneSalt_not_fresh`) and a live epoch a window meets it (`perWindow_fresh`). `Llama31Shape` and `Llama31Rounds` are
kernel computations.
-/

namespace Properties.Served

open Accounting.Compose Accounting.Served

theorem llama31Shape : Llama31Shape := by
  unfold Llama31Shape
  refine ⟨by decide +kernel, by decide +kernel, by decide +kernel, by decide +kernel, ?_⟩
  unfold Layout.Wf
  decide +kernel

theorem stepOf_lt {L : Layout} {c : ℕ} (hc : c < L.steps * L.calls) : L.stepOf c < L.steps := by
  unfold Layout.stepOf
  exact Nat.div_lt_of_lt_mul (by rwa [Nat.mul_comm] at hc)

/-- A call site's round is a round of its form. -/
theorem roundOf_lt {L : Layout} (F : Form) {c : ℕ} (hc : c < L.steps * L.calls) : F.roundOf L c < F.rounds L := by
  cases F <;> first | exact stepOf_lt hc | exact hc

/-- An `out` word's commit coin is its round's, except in `perCall`. -/
theorem commitCoin_out {F : Form} (hF : F ≠ .perCall) (r : ℕ) : commitCoin F .out r = r := by
  cases F <;> first | rfl | exact (hF rfl).elim

/-- A tied circuit's gate is in a round of the window. -/
theorem round_lt {ι : Type} {L : Layout} {F : Form} {Cd : Declared ι} {cls : ℕ → Cls} (T : Tie L F Cd cls) (i : ℕ)
    (hi : i < Cd.dag.length) : Cd.round (Cd.call i) < Cd.rounds := by
  have hc := T.inRange i hi
  have hc' : Cd.call i < L.steps * L.calls := by rw [← T.calls]; exact hc
  rw [T.round _ hc, T.rounds]
  exact roundOf_lt F hc'

/-- A value's first reader: none (the draw), or a gate that reads it. -/
theorem firstRead_cases {ι : Type} (Cd : Declared ι) (i : ℕ) :
    Cd.firstRead i = Cd.rounds ∨
      ∃ j, j < Cd.dag.length ∧ Cd.Reads j i ∧ Cd.firstRead i = Cd.round (Cd.call j) := by
  unfold Declared.firstRead
  cases h : List.find? (fun j => (Cd.dag[j]?).any fun g => g.args.contains i) (List.range Cd.dag.length) with
  | none => exact Or.inl rfl
  | some j =>
    have hp := List.find?_some h
    exact Or.inr ⟨j, List.mem_range.1 (List.mem_of_find?_eq_some h), hp, rfl⟩

/-- A value some gate reads has a first reader, a gate that reads it. -/
theorem firstRead_of_reads {ι : Type} (Cd : Declared ι) {i j : ℕ} (hj : j < Cd.dag.length) (hR : Cd.Reads j i) :
    ∃ j', j' < Cd.dag.length ∧ Cd.Reads j' i ∧ Cd.firstRead i = Cd.round (Cd.call j') := by
  unfold Declared.firstRead
  cases h : List.find? (fun j => (Cd.dag[j]?).any fun g => g.args.contains i) (List.range Cd.dag.length) with
  | none => exact absurd hR (List.find?_eq_none.1 h j (List.mem_range.2 hj))
  | some j' =>
    have hp := List.find?_some h
    exact ⟨j', List.mem_range.1 (List.mem_of_find?_eq_some h), hp, rfl⟩

/-- A run input only its own call reads is cut no earlier than its call's round. -/
theorem round_le_cut_of_run {ι : Type} {L : Layout} {F : Form} {Cd : Declared ι} {cls : ℕ → Cls}
    (T : Tie L F Cd cls) {ℓ : ℕ} (hlt : ℓ < Cd.dag.length) (hb : Cd.binding ℓ = some .run)
    (hr : ∀ j, j < Cd.dag.length → Cd.Reads j ℓ → Cd.call j = Cd.call ℓ) :
    Cd.round (Cd.call ℓ) ≤ Cd.cutOf ℓ := by
  simp only [Declared.cutOf, hb]
  rcases firstRead_cases Cd ℓ with h | ⟨j, hj, hR, h⟩
  · rw [h]
    exact (round_lt T ℓ hlt).le
  · rw [h]
    exact le_of_eq (congrArg Cd.round (hr j hj hR)).symm

/-- A gate that is no input is cut no earlier than its call's round. -/
theorem round_le_cut_of_none {ι : Type} {Cd : Declared ι} {ℓ : ℕ} (hb : Cd.binding ℓ = none) :
    Cd.round (Cd.call ℓ) ≤ Cd.cutOf ℓ := by
  simp only [Declared.cutOf, hb]
  exact le_max_left _ _

theorem tieCommits : TieCommits := by
  intro ι L F Cd cls T ℓ hW
  obtain ⟨hlt, hnc⟩ := hW
  have hround := T.round _ (T.inRange ℓ hlt)
  cases hk : cls ℓ with
  | coin => exact absurd ((T.coin ℓ hlt).1 hk) hnc
  | inner => exact le_of_eq (T.inner ℓ hlt hk)
  | registered =>
    rw [T.commit ℓ hlt (by simp [hk]) (by simp [hk]), hk]
    simp [commitCoin]
  | act =>
    obtain ⟨hb, hr⟩ := T.act ℓ hlt (Or.inl hk)
    rw [T.commit ℓ hlt (by simp [hk]) (by simp [hk]), hk, ← hround]
    exact round_le_cut_of_run T hlt hb hr
  | served =>
    obtain ⟨hb, hr⟩ := T.act ℓ hlt (Or.inr hk)
    rw [T.commit ℓ hlt (by simp [hk]) (by simp [hk]), hk, ← hround]
    exact round_le_cut_of_run T hlt hb hr
  | admit =>
    have hb := T.computed ℓ hlt (by simp [hk])
    rw [T.commit ℓ hlt (by simp [hk]) (by simp [hk]), hk, ← hround]
    exact round_le_cut_of_none hb
  | out =>
    have hb := T.computed ℓ hlt (by simp [hk])
    rw [T.commit ℓ hlt (by simp [hk]) (by simp [hk]), hk, ← hround]
    by_cases hF : F = .perCall
    · subst hF
      have hn := T.noised ℓ hlt (Or.inr ⟨hk, rfl⟩)
      show Cd.round (Cd.call ℓ) + 1 ≤ Cd.cutOf ℓ
      simp only [Declared.cutOf, hb]
      exact le_max_of_le_right hn
    · rw [commitCoin_out hF]
      exact round_le_cut_of_none hb
  | receipt =>
    have hb := T.computed ℓ hlt (by simp [hk])
    have hn := T.noised ℓ hlt (Or.inl hk)
    rw [T.commit ℓ hlt (by simp [hk]) (by simp [hk]), hk, ← hround]
    show Cd.round (Cd.call ℓ) + 1 ≤ Cd.cutOf ℓ
    simp only [Declared.cutOf, hb]
    exact le_max_of_le_right hn

theorem tieLive : TieLive := by
  intro ι K Sc L F Cd cls T u _ hw
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
    exact T.live _ (round_lt T i0 (List.mem_range.1 h0r))

theorem tieScope : TieScope := by
  intro ι K Sc L F Cd cls δs T hWf hδ hL
  unfold Scope
  exact ⟨hWf, tieLive ι K Sc L F Cd cls T, hδ, hL, tieCommits ι L F Cd cls T⟩

/-- Each window of a tied epoch is in Compose's scope. -/
theorem epochScope {ι C K : Type} {L : Layout} {F : Form} {W : ℕ} {Cd : ℕ → Declared ι} {cls : ℕ → ℕ → Cls}
    {coin : ℕ → ℕ → C} (E : EpochTie L F W Cd cls coin) (Sc : Scheme K) (δs : ℝ) {w : ℕ} (hw : w < W)
    (hWf : (Cd w).Wf) (hδ : 0 < δs) (hL : Sc.LuckProved) : Scope Sc (Cd w) δs :=
  tieScope ι K Sc L F (Cd w) (cls w) δs (E.tie w hw) hWf hδ hL

/-- A coin that doesn't read the window, as main's one salt an epoch (`c` constant), isn't fresh over two windows of a
round. -/
theorem oneSalt_not_fresh {C : Type} {W R : ℕ} (c : ℕ → C) (hW : 2 ≤ W) (hR : 0 < R) :
    ¬ CoinsFresh W R (fun _ r => c r) := by
  intro h
  have h01 := (h 0 0 1 0 (by omega) (by omega) hR hR rfl).1
  omega

/-- A live epoch a window, window `w`'s round-`r` coin `(w, r)`, is fresh. -/
theorem perWindow_fresh (W R : ℕ) : CoinsFresh W R (fun w r => (w, r)) := by
  intro w r w' r' _ _ _ _ h
  exact ⟨congrArg Prod.fst h, congrArg Prod.snd h⟩

theorem llama31Rounds : Llama31Rounds := by
  unfold Llama31Rounds
  refine ⟨fun F hF => ?_, by decide +kernel, by decide +kernel, by decide +kernel⟩
  cases F <;> first | exact (hF rfl).elim | decide +kernel

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
