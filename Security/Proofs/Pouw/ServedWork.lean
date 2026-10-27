import Properties.Pouw.ServedWork
import Proofs.Pouw.PeriodJ10R

/-!
# Proof of `ServedWork`

`Properties.Pouw.Served.ServedWork` from `H1J10RCheckedCreditPeriod` and `SAKServed`:

* `prob_salt_le`: a window's salt coin is uniform given anything fixed before it, so the salt falls in a degenerate set
  chosen from what is fixed before it with that set's probability (by windows: slices of the earlier windows' coins, then
  the window's round-0 coin among its coins).
* `servedWork`: on an accepted path outside the words guarantee's event, the credit is at most `n · fixed`, which
  `FixedAreRight` bounds by the windows' right words; outside every window's degenerate salts `SAKServed` prices each
  window's right words at `p` each, less `lam`. So the event is inside the words guarantee's event or some window's
  degenerate salts: `δ_s + n·(δ_bind + δ_ks)` plus `n·κ_deg(B)`.
-/

namespace Properties.Pouw.Served.SecurityProofs

open Properties.Pouw.Served Properties.Pouw.PeriodJ10R Properties.Pouw.PeriodJ10R.SecurityProofs
open Accounting.TS Accounting.Compose Accounting.Assumptions.TS Accounting.Assumptions.Compose Pouw.RelativeSecurity

/-! ## A window's salt is uniform given what is fixed before it -/

theorem prob_of_isEmpty {Ω : Type*} [Fintype Ω] [IsEmpty Ω] (p : Ω → Prop) : prob p = 0 := by
  unfold prob
  simp

theorem prob_false_eq {Ω : Type*} [Fintype Ω] : prob (fun _ : Ω => False) = 0 := by
  unfold prob
  simp

/-- An event whose every slice at the first coordinate has probability at most `κ ≥ 0` has probability at most `κ`. -/
theorem prob_le_of_slices_fst {α β : Type*} [Fintype α] [Fintype β] (E : α × β → Prop) {κ : ℝ} (hκ : 0 ≤ κ)
    (h : ∀ a, prob (fun b => E (a, b)) ≤ κ) : prob E ≤ κ := by
  rw [prob_equiv (Equiv.prodComm α β) E]
  exact prob_le_of_slices _ hκ fun a => h a

theorem paths_nonempty {Ω : Type} [h : Nonempty Ω] : ∀ n, Nonempty (Paths Ω n)
  | 0 => ⟨()⟩
  | n + 1 => by
    obtain ⟨x⟩ := paths_nonempty (Ω := Ω) n
    obtain ⟨ω⟩ := h
    exact ⟨(ω, x)⟩

/-- **The first window's salt coin**: an event of window 0's round-0 coin has that coin's probability. -/
theorem prob_coin_le {C B : Type} [Fintype C] [Fintype B] {m : ℕ} (hm : 0 < m + 1) {n : ℕ} (P : C → Prop) {κ : ℝ}
    (hκ : 0 ≤ κ) (hP : prob P ≤ κ) :
    prob (fun x : Paths (Paths C (m + 1) × B) (n + 1) => P (coinAt x.1.1 ⟨0, hm⟩)) ≤ κ := by
  refine prob_le_of_slices (fun x : (Paths C (m + 1) × B) × Paths (Paths C (m + 1) × B) n => P x.1.1.1) hκ
    fun _ => ?_
  refine prob_le_of_slices (fun ω : Paths C (m + 1) × B => P ω.1.1) hκ fun _ => ?_
  exact prob_le_of_slices (fun y : C × Paths C m => P y.1) hκ fun _ => hP

/-- **A window's salt is uniform given what is fixed before it**: for `g` fixed before window `w`'s salt and sets
`D a` of salts of probability at most `κ`, the salt falls in `D (g x)` with probability at most `κ`. -/
theorem prob_salt_le {C B X : Type} [Fintype C] [Fintype B] {m : ℕ} (hm : 0 < m) (D : X → C → Prop) {κ : ℝ}
    (hκ : 0 ≤ κ) (hD : ∀ a, prob (D a) ≤ κ) {n : ℕ} (g : Paths (Paths C m × B) n → X) (w : Fin n)
    (hg : ∀ x z, g (spliceAt w.val 0 x z) = g x) :
    prob (fun x => D (g x) (saltCoin hm x w)) ≤ κ := by
  obtain ⟨m, rfl⟩ : ∃ m', m = m' + 1 := ⟨m - 1, by omega⟩
  induction n with
  | zero => exact w.elim0
  | succ n ih =>
    obtain ⟨w, hw⟩ := w
    cases w with
    | zero =>
      by_cases he : IsEmpty (Paths (Paths C (m + 1) × B) (n + 1))
      · rw [@prob_of_isEmpty _ _ he]
        exact hκ
      · obtain ⟨x0⟩ := not_isEmpty_iff.mp he
        have hc : ∀ x, g x = g x0 := fun x => hg x0 x
        refine (prob_mono fun x hx => ?_).trans (prob_coin_le hm (n := n) (D (g x0)) hκ (hD _))
        rw [← hc x]
        exact hx
    | succ w =>
      have key : ∀ x0 : Paths C (m + 1) × B, prob (fun x' : Paths (Paths C (m + 1) × B) n =>
          D (g (x0, x')) (saltCoin (n := n + 1) hm (x0, x') ⟨w + 1, hw⟩)) ≤ κ := fun x0 =>
        ih ⟨w, Nat.lt_of_succ_lt_succ hw⟩ (fun x' => g (x0, x')) fun x' z' => hg (x0, x') (x0, z')
      exact prob_le_of_slices_fst (α := Paths C (m + 1) × B) (β := Paths (Paths C (m + 1) × B) n)
        (fun x => D (g x) (saltCoin (n := n + 1) hm x ⟨w + 1, hw⟩)) hκ key

/-! ## The guarantee -/

theorem servedWork : ServedWork := by
  intro ι K T L U C Ξ V I W St _ _ _ _ _ _ _ _ _ nB mB σ k hk sT unitOf Sc Cd n hn hN hr hM hKd ws opened enc place
    tile Wp Pl folded ρ δs δbind δks G μ Reg hon p lam κdeg B inst state work hS hρ hρ1 hρN hwmax hlk hG hDD hPre
    hMoved hRBD hDraw hServ hSalt hLucks hUF g1 g2 p1 p2 hSAK hSt hext hReg hp hFix hRight
  have hW := h1J10RCheckedCreditPeriod σ k hk sT unitOf Sc Cd hn hN hr hM hKd ws opened enc place tile Wp Pl folded
    ρ δs δbind δks hS hρ hρ1 hρN hwmax hlk hG hDD hPre hMoved hRBD hDraw hServ hSalt hLucks hUF
  have hne : Nonempty (Coin C Ξ Cd.rounds σ sT) := by
    obtain ⟨y⟩ := paths_nonempty (Ω := C) Cd.rounds
    obtain ⟨ξ⟩ := ‹Nonempty Ξ›
    exact ⟨(y, (Equiv.refl _, ξ))⟩
  obtain ⟨x0⟩ := @paths_nonempty (Coin C Ξ Cd.rounds σ sT) hne n
  have hκ : 0 ≤ κdeg B := by
    obtain ⟨Deg, hDeg, -⟩ := hSAK g1 g2 p1 p2 (inst x0 ⟨0, hn⟩) (hReg x0 _) B St hSt (state x0 ⟨0, hn⟩)
      (work x0 ⟨0, hn⟩) (hext x0 _)
    exact (prob_nonneg _).trans hDeg
  classical
  let Dg : I × St × G.Mc.Alg (St × C) (Fin (tileCount σ sT) → W) → C → Prop := fun a =>
    if h : a.1 ∈ Reg ∧ ∀ i, G.ext a.2.2 i = 0 then
      Classical.choose (hSAK g1 g2 p1 p2 a.1 h.1 B St hSt a.2.1 a.2.2 h.2) else fun _ => False
  have hDg : ∀ a, prob (Dg a) ≤ κdeg B := fun a => by
    by_cases h : a.1 ∈ Reg ∧ ∀ i, G.ext a.2.2 i = 0
    · simp only [Dg, dite_eq_left h]
      exact (Classical.choose_spec (hSAK g1 g2 p1 p2 a.1 h.1 B St hSt a.2.1 a.2.2 h.2)).1
    · simp only [Dg, dite_eq_right h]
      rw [prob_false_eq]
      exact hκ
  have hspec : ∀ x w, ¬ Dg (inst x w, state x w, work x w) (saltCoin hr x w) →
      p * Pouw.Assumptions.rightWords hon (fun t => (tileWords Sc Cd (tileCount σ sT) t : ℝ)) (inst x w) (saltCoin hr x w)
        (G.Mc.eval (work x w) (state x w, saltCoin hr x w)) - lam (inst x w) ≤
          G.cost μ (work x w) (state x w, saltCoin hr x w) := fun x w hd => by
    have h : (inst x w, state x w, work x w).1 ∈ Reg ∧ ∀ i, G.ext (inst x w, state x w, work x w).2.2 i = 0 :=
      ⟨hReg x w, hext x w⟩
    simp only [Dg, dite_eq_left h] at hd
    exact (Classical.choose_spec (hSAK g1 g2 p1 p2 _ h.1 B St hSt _ _ h.2)).2 _ hd
  have hsub : ∀ x, (Pl.accepts x ∧ ∑ w, G.cost μ (work x w) (state x w, saltCoin hr x w) <
      p * creditPrefix Sc Cd (tileCount σ sT) (ws.prefixLen x) (creditLive Sc Cd Wp ws x) 1 1 (j10rRate (k sT) ρ) δs -
        ∑ w, lam (inst x w)) →
      (Pl.accepts x ∧ n * (Pl.run x).fixed <
        creditPrefix Sc Cd (tileCount σ sT) (ws.prefixLen x) (creditLive Sc Cd Wp ws x) 1 1 (j10rRate (k sT) ρ) δs) ∨
      ∃ w ∈ Finset.univ, Dg (inst x w, state x w, work x w) (saltCoin hr x w) := by
    rintro x ⟨hacc, hlt⟩
    by_cases hd : ∃ w ∈ Finset.univ, Dg (inst x w, state x w, work x w) (saltCoin hr x w)
    · exact Or.inr hd
    refine Or.inl ⟨hacc, not_le.mp fun hle => ?_⟩
    have hfr : n * (Pl.run x).fixed ≤ ∑ w, Pouw.Assumptions.rightWords hon
        (fun t => (tileWords Sc Cd (tileCount σ sT) t : ℝ)) (inst x w) (saltCoin hr x w)
        (G.Mc.eval (work x w) (state x w, saltCoin hr x w)) := hRight x hacc
    have hsum := Finset.sum_le_sum fun w (_ : w ∈ Finset.univ) =>
      hspec x w fun h => hd ⟨w, Finset.mem_univ w, h⟩
    rw [Finset.sum_sub_distrib, ← Finset.mul_sum] at hsum
    have key := mul_le_mul_of_nonneg_left (hle.trans hfr) hp
    linarith
  calc _ ≤ prob (fun x => (Pl.accepts x ∧ n * (Pl.run x).fixed <
          creditPrefix Sc Cd (tileCount σ sT) (ws.prefixLen x) (creditLive Sc Cd Wp ws x) 1 1 (j10rRate (k sT) ρ) δs) ∨
        ∃ w ∈ Finset.univ, Dg (inst x w, state x w, work x w) (saltCoin hr x w)) := prob_mono hsub
    _ ≤ (δs + n * (δbind + δks)) + ∑ w ∈ Finset.univ,
          prob (fun x => Dg (inst x w, state x w, work x w) (saltCoin hr x w)) :=
        (prob_or_le _ _).trans (add_le_add hW (prob_exists_le Finset.univ
          (fun w x => Dg (inst x w, state x w, work x w) (saltCoin hr x w))))
    _ ≤ (δs + n * (δbind + δks)) + ∑ _w ∈ (Finset.univ : Finset (Fin n)), κdeg B :=
        add_le_add le_rfl (Finset.sum_le_sum fun w _ =>
          prob_salt_le hr Dg hκ hDg (fun x => (inst x w, state x w, work x w)) w fun x z => hFix x z w)
    _ = δs + n * (δbind + δks) + n * κdeg B := by
        rw [Finset.sum_const, Finset.card_univ, Fintype.card_fin, nsmul_eq_mul]

end Properties.Pouw.Served.SecurityProofs
