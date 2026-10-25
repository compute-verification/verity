import Definitions.Accounting.Guarantees.RemoteBinding
import Proofs.Accounting.TS.Drawn
import Proofs.Hash.RowBinding

/-!
# The receipts' binding bound: the proof

Fix window `t` and a receipt `p`, and read a cut of coins and its continuations `cont t` (the other coins and the
earlier windows' openings coins kept, window `t`'s and the later ones fresh), which have `spliceAt`'s two laws
(`Binding.cont_left`, `Binding.cont_swap`).

* **Per receipt** (`Binding.Fork.moved_le_forked`). `Plurality.moved_le_forked`, for any value `c` fixed by the cut that
  no value outnumbers among its continuations, not only the plurality `Classical.epsilon` picks: the continuations
  showing a row other than `c` (`Fork.Mov`), squared, are at most the pairs of continuations showing two different rows
  (`Plurality.Frk`). The held row is such a `c` (`HeldLikeliest`), and a move shows the presented row against it, so
  `Pr[p moved]² ≤ Pr[p forked]`.
* **Over the window's receipts** (`Binding.window_le`). The union bound and Cauchy–Schwarz over `L_t` give
  `(Σ_p Pr[p moved])² ≤ |L_t|·Σ_p Pr[p forked]`. A run opens at most `s_t` receipts (`Binding.card_opened_le`), so
  `Σ_p Pr[p forked] ≤ s_t·Pr[some p forked]` (`Binding.Fork.sum_prob_le`). A forked receipt is accepted in both runs
  with two different rows: at one verifier, root and rank that is a SHA-512 collision among what the two checks hashed
  (`Hash.SecurityProofs.RowBinding`), the fork's break (`Receipts.Broken`), and otherwise the roots split
  (`Receipts.Split`).
* **Under `cr/sha-512`** (`Binding.fork_le_finder`). A break is a collision the window's fork finder outputs, so
  `fork_t ≤ Pr[finder succeeds] ≤ q_t²/2^513`.
* **Fixed roots** (`Binding.binding_le_of_fixed`). Where the window's cut fixes its receipts' roots, ranks and
  verifiers, `split_t = 0`, and `√(|L_t|·s_t·q_t²/2^513) = √(|L_t|·s_t)·q_t/2^256.5`.
-/

namespace Accounting.SecurityProofs.Remote

open Finset Accounting.TS Accounting.Remote.Draw Accounting.Remote.Draw.Assumptions Pouw.RelativeSecurity
open Accounting.SecurityProofs.TS.Plurality (Frk sum_sp)

namespace Binding

namespace Fork

variable {Y V : Type} [Fintype Y] {sp : Y → Y → Y} {opens : Y → V → Prop}

/-- `y` shows a value other than `c y`. -/
def Mov (opens : Y → V → Prop) (c : Y → V) (y : Y) : Prop := ∃ v, opens y v ∧ v ≠ c y

open Classical in
/-- **At one node**: the continuations of `x` showing a value other than `c`, squared, are at most its forked pairs, for
any `c` fixed along continuations that no value outnumbers. -/
theorem moved_sq_le (hl : ∀ x z z', sp (sp x z) z' = sp x z') {c : Y → V} (hc : ∀ x z, c (sp x z) = c x)
    (hmax : ∀ x u, tally sp opens x u ≤ tally sp opens x (c x)) (x : Y) :
    (∑ z, if Mov opens c (sp x z) then 1 else 0 : ℕ) ^ 2 ≤
      ∑ z1, ∑ z2, if Frk sp opens (sp x z1) z2 then 1 else 0 := by
  have hmov : ∀ z, Mov opens c (sp x z) ↔ ∃ v, opens (sp x z) v ∧ v ≠ c x := fun z => by
    simp only [Mov, hc]
  have hfrk : ∀ z1 z2, Frk sp opens (sp x z1) z2 ↔ ∃ v v', opens (sp x z1) v ∧ opens (sp x z2) v' ∧ v ≠ v' :=
    fun z1 z2 => by simp only [Frk, hl]
  set Nn : Finset Y := Finset.univ.filter fun z => ∃ v, opens (sp x z) v with hNn
  set Mv : Finset Y := Finset.univ.filter fun z => Mov opens c (sp x z) with hMv
  set Ec : Finset Y := Finset.univ.filter fun z => Alone opens (sp x z) (c x) with hEc
  have hdisj : Disjoint Mv Ec := by
    rw [Finset.disjoint_left]
    intro z hm he
    simp only [hMv, hEc, Finset.mem_filter, Finset.mem_univ, true_and] at hm he
    rw [hmov] at hm
    unfold Alone at he
    obtain ⟨v, hv, hne⟩ := hm
    exact hne (he.2 v hv)
  have hunion : Mv ∪ Ec = Nn := by
    ext z
    simp only [hMv, hEc, hNn, Finset.mem_union, Finset.mem_filter, Finset.mem_univ, true_and]
    rw [hmov]
    unfold Alone
    constructor
    · rintro (⟨v, hv, -⟩ | ⟨hv, -⟩)
      · exact ⟨v, hv⟩
      · exact ⟨c x, hv⟩
    · rintro ⟨v, hv⟩
      by_cases hall : ∀ v', opens (sp x z) v' → v' = c x
      · exact Or.inr ⟨hall v hv ▸ hv, hall⟩
      · exact Or.inl (Classical.byContradiction fun hno =>
          hall fun v' hv' => Classical.byContradiction fun hne => hno ⟨v', hv', hne⟩)
  have hpart : Mv.card + Ec.card = Nn.card := by
    rw [← hunion, Finset.card_union_of_disjoint hdisj]
  have hrow : ∀ z1, (∃ v, opens (sp x z1) v) →
      Mv.card ≤ (Finset.univ.filter fun z2 => Frk sp opens (sp x z1) z2).card := by
    rintro z1 ⟨v1, h1⟩
    have hsub : Nn ⊆ (Finset.univ.filter fun z2 => Frk sp opens (sp x z1) z2) ∪
        (Finset.univ.filter fun z2 => Alone opens (sp x z2) v1) := by
      intro z2 hz2
      simp only [hNn, Finset.mem_union, Finset.mem_filter, Finset.mem_univ, true_and] at hz2 ⊢
      by_cases hf : Frk sp opens (sp x z1) z2
      · exact Or.inl hf
      · right
        obtain ⟨v2, h2⟩ := hz2
        have key : ∀ v', opens (sp x z2) v' → v' = v1 := fun v' h' => by
          by_contra hne
          exact hf ((hfrk z1 z2).2 ⟨v1, v', h1, h', Ne.symm hne⟩)
        exact And.intro (key v2 h2 ▸ h2) key
    have h3 := (Finset.card_le_card hsub).trans (Finset.card_union_le _ _)
    have h4 : (Finset.univ.filter fun z2 => Alone opens (sp x z2) v1).card ≤ Ec.card := hmax x v1
    omega
  have hM : (∑ z, if Mov opens c (sp x z) then 1 else 0 : ℕ) = Mv.card := by
    rw [hMv, Finset.card_filter]
  have hR : ∀ z1, (∑ z2, if Frk sp opens (sp x z1) z2 then 1 else 0 : ℕ) =
      (Finset.univ.filter fun z2 => Frk sp opens (sp x z1) z2).card := fun z1 => by
    rw [Finset.card_filter]
  rw [hM, sq]
  calc Mv.card * Mv.card = ∑ _z1 ∈ Mv, Mv.card := by rw [Finset.sum_const, smul_eq_mul]
    _ ≤ ∑ z1 ∈ Mv, ∑ z2, (if Frk sp opens (sp x z1) z2 then 1 else 0 : ℕ) := by
        refine Finset.sum_le_sum fun z1 hz1 => ?_
        rw [hR]
        refine hrow z1 ?_
        simp only [hMv, Finset.mem_filter, Finset.mem_univ, true_and, hmov] at hz1
        obtain ⟨v, hv, -⟩ := hz1
        exact ⟨v, hv⟩
    _ ≤ ∑ z1, ∑ z2, (if Frk sp opens (sp x z1) z2 then 1 else 0 : ℕ) :=
        Finset.sum_le_sum_of_subset_of_nonneg (Finset.subset_univ _) fun _ _ _ => Nat.zero_le _

open Classical in
/-- **Showing a value other than `c`, squared, is at most the fork form**, for any continuation map with `spliceAt`'s
laws and any `c` fixed along continuations that no value outnumbers. -/
theorem moved_le_forked (hl : ∀ x z z', sp (sp x z) z' = sp x z') (hs : ∀ x z, sp (sp x z) (sp z x) = x)
    {c : Y → V} (hc : ∀ x z, c (sp x z) = c x) (hmax : ∀ x u, tally sp opens x u ≤ tally sp opens x (c x)) :
    prob (fun x => Mov opens c x) ^ 2 ≤ prob (fun p : Y × Y => Frk sp opens p.1 p.2) := by
  set C := Fintype.card Y with hC
  set M : ℕ := ∑ y, if Mov opens c y then 1 else 0 with hM
  set Fk : ℕ := ∑ y, ∑ z, if Frk sp opens y z then 1 else 0 with hFk
  have hA : ∑ x, (∑ z, if Mov opens c (sp x z) then 1 else 0 : ℕ) = C * M :=
    sum_sp hs (fun y => if Mov opens c y then 1 else 0)
  have hF : ∑ x, (∑ z1, ∑ z2, if Frk sp opens (sp x z1) z2 then 1 else 0 : ℕ) = C * Fk := by
    calc ∑ x, (∑ z1, ∑ z2, if Frk sp opens (sp x z1) z2 then 1 else 0 : ℕ)
        = ∑ x, ∑ z2, ∑ z1, (if Frk sp opens (sp x z1) z2 then 1 else 0 : ℕ) :=
          Finset.sum_congr rfl fun x _ => Finset.sum_comm
      _ = ∑ z2, ∑ x, ∑ z1, (if Frk sp opens (sp x z1) z2 then 1 else 0 : ℕ) := Finset.sum_comm
      _ = ∑ z2, C * ∑ y, (if Frk sp opens y z2 then 1 else 0 : ℕ) :=
          Finset.sum_congr rfl fun z2 _ => sum_sp hs (fun y => if Frk sp opens y z2 then 1 else 0)
      _ = C * Fk := by rw [← Finset.mul_sum, hFk, Finset.sum_comm]
  have hCS : (∑ x, (∑ z, if Mov opens c (sp x z) then 1 else 0 : ℕ)) ^ 2 ≤
      C * ∑ x, (∑ z, if Mov opens c (sp x z) then 1 else 0 : ℕ) ^ 2 := by
    simpa using sq_sum_le_card_mul_sum_sq (s := Finset.univ)
      (f := fun x => (∑ z, if Mov opens c (sp x z) then 1 else 0 : ℕ))
  have h2 : (C * M) ^ 2 ≤ C * (C * Fk) := by
    rw [← hA, ← hF]
    exact hCS.trans (Nat.mul_le_mul_left _ (Finset.sum_le_sum fun x _ => moved_sq_le hl hc hmax x))
  have hcM : (Finset.univ.filter fun x => Mov opens c x).card = M := by rw [Finset.card_filter]
  have hcF : (Finset.univ.filter fun p : Y × Y => Frk sp opens p.1 p.2).card = Fk := by
    rw [Finset.card_filter, Fintype.sum_prod_type]
  unfold prob
  rw [hcM, hcF, Fintype.card_prod, ← hC]
  rcases Nat.eq_zero_or_pos C with h0 | hpos
  · simp [h0]
  have hMF : M ^ 2 ≤ Fk := by
    have : C ^ 2 * M ^ 2 ≤ C ^ 2 * Fk := by nlinarith [h2]
    exact Nat.le_of_mul_le_mul_left this (by positivity)
  rw [div_pow, Nat.cast_mul, ← sq]
  exact div_le_div_of_nonneg_right (by exact_mod_cast hMF) (by positivity)

/-- Events each of which names a member of a set of at most `k`: their probabilities sum to at most `k` times that of
some one of them. -/
theorem sum_prob_le {P : Type} (L : Finset P) (F : P → Y → Prop) (A : Y → Finset P) (k : ℕ)
    (hA : ∀ p y, F p y → p ∈ A y) (hk : ∀ y, (A y).card ≤ k) :
    ∑ p ∈ L, prob (F p) ≤ k * prob (fun y => ∃ p ∈ L, F p y) := by
  classical
  have key : ∀ y, (∑ p ∈ L, if F p y then 1 else 0 : ℕ) ≤ k * (if ∃ p ∈ L, F p y then 1 else 0) := by
    intro y
    split_ifs with h
    · rw [mul_one, ← Finset.card_filter]
      exact (Finset.card_le_card fun p hp => hA p y (Finset.mem_filter.1 hp).2).trans (hk y)
    · rw [mul_zero, Nat.le_zero]
      exact Finset.sum_eq_zero fun p hp => by simp [show ¬ F p y from fun hF => h ⟨p, hp, hF⟩]
  have hn : (∑ p ∈ L, (univ.filter (F p)).card : ℕ) ≤ k * (univ.filter fun y => ∃ p ∈ L, F p y).card := by
    calc (∑ p ∈ L, (univ.filter (F p)).card : ℕ) = ∑ p ∈ L, ∑ y, (if F p y then 1 else 0 : ℕ) := by
          simp only [Finset.card_filter]
      _ = ∑ y, ∑ p ∈ L, (if F p y then 1 else 0 : ℕ) := Finset.sum_comm
      _ ≤ ∑ y, k * (if ∃ p ∈ L, F p y then 1 else 0) := Finset.sum_le_sum fun y _ => key y
      _ = k * (univ.filter fun y => ∃ p ∈ L, F p y).card := by rw [← Finset.mul_sum, Finset.card_filter]
  unfold prob
  rw [← Finset.sum_div, mul_div_assoc']
  refine div_le_div_of_nonneg_right ?_ (Nat.cast_nonneg _)
  have := (Nat.cast_le (α := ℝ)).2 hn
  push_cast at this
  convert this using 2
  congr 2
  ext y
  simp

end Fork

variable {n m : ℕ} {s : Fin n → ℕ} {κ : ℕ} {X : Type}

theorem cont_left (t : Fin n) (ω ζ ζ' : Coins n κ X) : cont t (cont t ω ζ) ζ' = cont t ω ζ' := by
  unfold cont
  refine Prod.ext rfl (funext fun j => ?_)
  by_cases h : j < t <;> simp [h]

theorem cont_swap (t : Fin n) (ω ζ : Coins n κ X) : cont t (cont t ω ζ) (cont t ζ ω) = ω := by
  unfold cont
  refine Prod.ext rfl (funext fun j => ?_)
  by_cases h : j < t <;> simp [h]

theorem toReal_pr {α : Type} [Fintype α] (P : α → Prop) : (Pous.pr P).toReal = prob P := by
  unfold Pous.pr prob
  rw [ENNReal.toReal_div, ENNReal.toReal_natCast, ENNReal.toReal_natCast]

theorem claims_pr {α : Type} [Fintype α] (P : α → Prop) : Claims.Pr P = prob P := by
  unfold Claims.Pr prob
  congr

/-- A window opens at most `s_t` receipts: its draws, `min(s_t, U_t)`. -/
theorem card_opened_le (D : Session n m s κ X) (prf : Fin n → Pous.Bits κ → Blocks m)
    (rest : Coins n κ X → Fin n → ℕ → ℕ) (t : Fin n) (ω : Coins n κ X) : (D.opened prf rest t ω).card ≤ s t := by
  unfold Session.opened openings
  exact Finset.card_image_le.trans ((Finset.card_range _).le.trans (by unfold Session.draws; exact min_le_left _ _))

/-- **Window `t`'s share of the binding term** is at most `√(|L_t|·s_t·(fork_t + split_t))`. -/
theorem window_le [Fintype X] (D : Session n m s κ X) (R : Receipts n κ X) (prf : Fin n → Pous.Bits κ → Blocks m)
    (rest : Coins n κ X → Fin n → ℕ → ℕ) (L : Fin n → Finset ℕ) (hL : ∀ t ω, D.opened prf rest t ω ⊆ L t)
    (hH : HeldLikeliest R D prf rest) (t : Fin n) :
    prob (fun ω => ∃ p ∈ D.opened prf rest t ω, R.Moved t ω p ∧ R.Collides t ω p) ≤
      √((L t).card * s t * ((R.fork D prf rest t).toReal + (R.split D prf rest t).toReal)) := by
  set a : ℕ → ℝ := fun p => prob (Fork.Mov (R.Shows D prf rest t p) fun ω => (R.held t ω p).row) with ha
  set Fk : ℕ → Coins n κ X × Coins n κ X → Prop := fun p y => Frk (cont t) (R.Shows D prf rest t p) y.1 y.2
    with hFk
  have h1 : prob (fun ω => ∃ p ∈ D.opened prf rest t ω, R.Moved t ω p ∧ R.Collides t ω p) ≤ ∑ p ∈ L t, a p := by
    refine (prob_mono fun ω hω => ?_).trans
      (prob_exists_le (L t) fun p ω => Fork.Mov (R.Shows D prf rest t p) (fun ω => (R.held t ω p).row) ω)
    obtain ⟨p, hp, ⟨-, hacc, hne⟩, -⟩ := hω
    exact ⟨p, hL t ω hp, (R.opened t ω p).row, ⟨hp, hacc, rfl⟩, Ne.symm hne⟩
  have hsq : ∀ p, a p ^ 2 ≤ prob (Fk p) := fun p =>
    Fork.moved_le_forked (cont_left t) (cont_swap t) (fun ω ζ => (hH t ω p).1 ζ) (fun ω v => (hH t ω p).2 v)
  have hsum : ∑ p ∈ L t, prob (Fk p) ≤ s t * prob (fun y => ∃ p ∈ L t, Fk p y) :=
    Fork.sum_prob_le (L t) Fk (fun y => D.opened prf rest t y.1) (s t) (fun p y h => by
      obtain ⟨v, v', ⟨hp, -⟩, -⟩ := h
      exact hp) (fun y => card_opened_le D prf rest t y.1)
  have hbreak : prob (fun y => ∃ p ∈ L t, Fk p y) ≤
      (R.fork D prf rest t).toReal + (R.split D prf rest t).toReal := by
    rw [Receipts.fork, Receipts.split, toReal_pr, toReal_pr]
    refine (prob_mono fun y hy => ?_).trans (prob_or_le _ _)
    obtain ⟨p, -, v, v', ⟨hp, hac, rfl⟩, ⟨hp', hac', rfl⟩, hne⟩ := hy
    by_cases hk : R.verifier t (cont t y.1 y.2) p = R.verifier t y.1 p ∧
        R.root t (cont t y.1 y.2) p = R.root t y.1 p ∧ R.rank t (cont t y.1 y.2) p = R.rank t y.1 p
    · left
      obtain ⟨hv, hr, hk'⟩ := hk
      rw [hv, hr, hk'] at hac'
      rcases Hash.SecurityProofs.RowBinding R.hash R.hash _ _ _ _ _ hac hac' with h | h | h
      · exact absurd h hne
      · exact ⟨p, hp, hp', Or.inl (by rw [hv, hk']; exact h)⟩
      · exact ⟨p, hp, hp', Or.inr (by rw [hv]; exact h)⟩
    · exact Or.inr ⟨p, hp, hp', hk⟩
  have hA : 0 ≤ ∑ p ∈ L t, a p := Finset.sum_nonneg fun p _ => prob_nonneg _
  have hfin : (∑ p ∈ L t, a p) ^ 2 ≤
      (L t).card * s t * ((R.fork D prf rest t).toReal + (R.split D prf rest t).toReal) := by
    calc (∑ p ∈ L t, a p) ^ 2 ≤ (L t).card * ∑ p ∈ L t, a p ^ 2 := sq_sum_le_card_mul_sum_sq
      _ ≤ (L t).card * ∑ p ∈ L t, prob (Fk p) :=
          mul_le_mul_of_nonneg_left (Finset.sum_le_sum fun p _ => hsq p) (Nat.cast_nonneg _)
      _ ≤ (L t).card * (s t * ((R.fork D prf rest t).toReal + (R.split D prf rest t).toReal)) :=
          mul_le_mul_of_nonneg_left (hsum.trans (mul_le_mul_of_nonneg_left hbreak (Nat.cast_nonneg _)))
            (Nat.cast_nonneg _)
      _ = _ := by ring
  exact h1.trans ((Real.sqrt_sq hA).symm.le.trans (Real.sqrt_le_sqrt hfin))

/-- A fork break is a collision the window's fork finder outputs. -/
theorem fork_le_finder [Fintype X] [Nonempty X] (D : Session n m s κ X) (R : Receipts n κ X)
    (prf : Fin n → Pous.Bits κ → Blocks m) (rest : Coins n κ X → Fin n → ℕ → ℕ) (t : Fin n) (q : ℕ) :
    (R.fork D prf rest t).toReal ≤ (R.finder D prf rest t q).Pr (fun u => (R.hash u).bytes) := by
  rw [Receipts.fork, toReal_pr, Claims.Finder.Pr, claims_pr]
  refine prob_mono (Ω := Coins n κ X × Coins n κ X) fun y hy => ?_
  have hex : ∃ uv, R.Finds D prf rest t y.1 y.2 uv := by
    obtain ⟨p, hp, hp', h | h⟩ := hy
    · obtain ⟨u, hu, v, hv, hc⟩ := h
      exact ⟨(u, v), hc, p, hp, hp', Or.inl ⟨hu, hv⟩⟩
    · obtain ⟨u, hu, v, hv, hc⟩ := h
      exact ⟨(u, v), hc, p, hp, hp', Or.inr ⟨hu, hv⟩⟩
  refine ⟨hex.choose.1, hex.choose.2, ?_, hex.choose_spec.1⟩
  simp [Receipts.finder, hex]

end Binding

/-- **The receipts' binding term, in its fork form** (`Accounting.Guarantees.Remote.ReceiptsBindingOfFork`). -/
theorem ReceiptsBindingOfFork : Accounting.Guarantees.Remote.ReceiptsBindingOfFork := by
  intro n m s κ X _ D R prf rest L hH hL
  rw [Receipts.binding, Binding.toReal_pr]
  refine (prob_mono fun ω hω => ?_).trans ((prob_exists_le Finset.univ fun t ω =>
    ∃ p ∈ D.opened prf rest t ω, R.Moved t ω p ∧ R.Collides t ω p).trans
      (Finset.sum_le_sum fun t _ => Binding.window_le D R prf rest L hL hH t))
  obtain ⟨t, h⟩ := hω
  exact ⟨t, Finset.mem_univ _, h⟩

/-- **The receipts' binding term under `cr/sha-512`** (`Accounting.Guarantees.Remote.ReceiptsBindingOfCR`). -/
theorem ReceiptsBindingOfCR : Accounting.Guarantees.Remote.ReceiptsBindingOfCR := by
  intro n m s κ X _ _ D R prf rest L q hH hL hq
  refine (ReceiptsBindingOfFork D R prf rest L hH hL).trans (Finset.sum_le_sum fun t _ => Real.sqrt_le_sqrt ?_)
  refine mul_le_mul_of_nonneg_left (add_le_add ?_ le_rfl) (by positivity)
  exact (Binding.fork_le_finder D R prf rest t (q t)).trans (hq t (q t) fun _ => le_rfl)

namespace Binding

variable {n m : ℕ} {s : Fin n → ℕ} {κ : ℕ} {X : Type}

/-- **Fixed roots**: where every window's cut fixes its receipts' verifiers, roots and ranks, the roots never split,
and under `cr/sha-512` the binding term is at most `Σ_t √(|L_t|·s_t)·q_t/2^256.5`, `ledger.binding`'s charge. -/
theorem binding_le_of_fixed [Fintype X] [Nonempty X] (D : Session n m s κ X) (R : Receipts n κ X)
    (prf : Fin n → Pous.Bits κ → Blocks m) (rest : Coins n κ X → Fin n → ℕ → ℕ) (L : Fin n → Finset ℕ)
    (q : Fin n → ℕ) (hH : HeldLikeliest R D prf rest) (hL : ∀ t ω, D.opened prf rest t ω ⊆ L t)
    (hq : ∀ t, Claims.StrictBirthday (fun u => (R.hash u).bytes) (R.finder D prf rest t (q t)))
    (hF : ∀ t ω ζ p, R.verifier t (cont t ω ζ) p = R.verifier t ω p ∧ R.root t (cont t ω ζ) p = R.root t ω p ∧
      R.rank t (cont t ω ζ) p = R.rank t ω p) :
    (R.binding D prf rest).toReal ≤ ∑ t, √((L t).card * s t) * q t / 2 ^ (256.5 : ℝ) := by
  refine (ReceiptsBindingOfCR D R prf rest L q hH hL hq).trans (le_of_eq (Finset.sum_congr rfl fun t _ => ?_))
  have hs : (R.split D prf rest t).toReal = 0 := by
    rw [Receipts.split, toReal_pr]
    have : (fun y : Coins n κ X × Coins n κ X => R.Split D prf rest t y.1 y.2) = fun _ => False := by
      funext y
      exact propext ⟨fun ⟨p, _, _, h⟩ => h (hF t y.1 y.2 p), False.elim⟩
    rw [this]
    unfold prob
    simp
  have h2 : √((2 : ℝ) ^ (513 : ℝ)) = 2 ^ (256.5 : ℝ) := by
    rw [Real.sqrt_eq_rpow, ← Real.rpow_mul (by norm_num)]
    norm_num
  rw [hs, add_zero, Real.sqrt_mul (by positivity), Real.sqrt_div (by positivity), Real.sqrt_sq (by positivity), h2,
    mul_div_assoc]

end Binding

end Accounting.SecurityProofs.Remote
