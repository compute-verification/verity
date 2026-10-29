import Proofs.Accounting.TS.Fork

/-!
# (I8) for a sampled window

A unit's `Good` reads the run's modes, and E1(c) is per unit (`UnitFixed`). `DrawnRightOff` and `DrawnBindingOfFork`
don't read `UnitFixed`; `DrawnLayer.covers_of_cone` is the share's side of `Covers`: a share that charges each unit the
run didn't pay for its late cone's weight covers every unit that isn't good.

`DrawnRightOff`: the event is "in some round, binding fails (`MovedIn`) or some drawn unit's proof is unsound", which
the union bound puts at `S·(δ_bind + δ_ks)`. Off it, an accepted proof opens its unit's committed values (not
`MovedIn`), which satisfy its relation (not `Unsound`), so they are right (`Pins`).

`ProofsSoundOfDrawn`: `DrawnRightOff`'s event; off it, on an accepted path every drawn unit's proof is accepted (`Of`),
so the ideal game accepts (`Of`), and the good units' words over `S` (`UnitFixed`) are at least
`(S·N_w − d·N_w)/S` by `Covers` and `Σ wt = S·N_w`.

`DrawnBindingOfFork`: a round's binding failure moves one of its positions (`Moved`). Per position, the moved mass
squared is at most the forked mass at its cut (`Plurality.moved_le_forked`, for any continuation map with
`spliceAt`'s two laws): fix the coins before the cut; of its continuations, `A` open the position, `E_v` open it to `v`
alone, and the plurality `c` has the most, so the moved are `M = A − E_c`. A continuation that opens some `v₁` forks
with every continuation that opens the position, except those opening `v₁` alone, so with at least `A − E_c = M`
(`moved_sq_le`), and the moved alone give `M²` forked pairs. Cauchy–Schwarz over the coins before the cut, then over
`L`, gives `|L|²`.
-/

namespace Accounting.SecurityProofs.TS

open Accounting.TS Accounting.Assumptions.TS Pouw.RelativeSecurity

/-- A continuation's continuation from the same coin is a continuation of the first path. -/
theorem spliceAt_left {C B : Type} {m : ℕ} : ∀ {n : ℕ} (w k : ℕ) (x z z' : Paths (Paths C m × B) n),
    spliceAt w k (spliceAt w k x z) z' = spliceAt w k x z'
  | 0, _, _, _, _, _ => rfl
  | _ + 1, 0, k, x, z, z' => by simp only [spliceAt, splice_left]
  | _ + 1, w + 1, k, x, z, z' => by simp only [spliceAt, spliceAt_left w k x.2 z.2 z'.2]

/-- Swapping two paths' coins from round `w`'s coin `k` on, twice, gives the first back. -/
theorem spliceAt_swap {C B : Type} {m : ℕ} : ∀ {n : ℕ} (w k : ℕ) (x z : Paths (Paths C m × B) n),
    spliceAt w k (spliceAt w k x z) (spliceAt w k z x) = x
  | 0, _, _, _, _ => rfl
  | _ + 1, 0, k, x, z => by simp only [spliceAt, splice_swap]
  | _ + 1, w + 1, k, x, z => by simp only [spliceAt, spliceAt_swap w k x.2 z.2]

namespace Plurality

variable {X V : Type} [Fintype X] {sp : X → X → X} {opens : X → V → Prop}

/-- Summing over every path and every continuation of it is summing over paths, each `|X|` times. -/
theorem sum_sp (hs : ∀ x z, sp (sp x z) (sp z x) = x) (q : X → ℕ) :
    ∑ x, ∑ z, q (sp x z) = Fintype.card X * ∑ y, q y := by
  let Φ : X × X → X × X := fun p => (sp p.1 p.2, sp p.2 p.1)
  have hΦ : Function.Involutive Φ := fun p => by simp only [Φ, hs]
  have h := Equiv.sum_comp (hΦ.toPerm Φ) (fun p => q p.1)
  simp only [Function.Involutive.coe_toPerm, Φ] at h
  rw [← Fintype.sum_prod_type' (fun x z => q (sp x z)), h, Fintype.sum_prod_type (fun p => q p.1),
    Finset.mul_sum]
  simp [Finset.sum_const, Finset.card_univ, mul_comm]

/-- Binding's failure at `x`: `x` opens a value other than the plurality. -/
def Mov [Nonempty V] (sp : X → X → X) (opens : X → V → Prop) (x : X) : Prop :=
  ∃ v, opens x v ∧ v ≠ plural sp opens x

/-- `x` and its continuation on `z` open two different values. -/
def Frk (sp : X → X → X) (opens : X → V → Prop) (x z : X) : Prop :=
  ∃ v v', opens x v ∧ opens (sp x z) v' ∧ v ≠ v'

theorem tally_sp (hl : ∀ x z z', sp (sp x z) z' = sp x z') (x z : X) :
    tally sp opens (sp x z) = tally sp opens x := by
  funext v
  unfold tally
  congr 1
  ext z'
  simp only [Finset.mem_filter, Finset.mem_univ, true_and, hl]

/-- Every continuation of `x` has `x`'s plurality. -/
theorem plural_sp [Nonempty V] (hl : ∀ x z z', sp (sp x z) z' = sp x z') (x z : X) :
    plural sp opens (sp x z) = plural sp opens x := by
  unfold plural; rw [tally_sp hl]

open Classical in
/-- The plurality has the most continuations opening it alone. -/
theorem tally_le_plural [Nonempty V] (x : X) (u : V) :
    tally sp opens x u ≤ tally sp opens x (plural sp opens x) := by
  have hb : BddAbove (Set.range (tally sp opens x)) := by
    refine ⟨Fintype.card X, ?_⟩
    rintro _ ⟨u, rfl⟩
    exact (Finset.card_filter_le _ _).trans_eq Finset.card_univ
  obtain ⟨v, hv⟩ := Nat.sSup_mem (Set.range_nonempty (tally sp opens x)) hb
  have hex : ∃ v, ∀ u, tally sp opens x u ≤ tally sp opens x v :=
    ⟨v, fun u => hv ▸ le_csSup hb ⟨u, rfl⟩⟩
  exact Classical.epsilon_spec hex u

open Classical in
/-- **At one node**: the moved continuations of `x`, squared, are at most its forked pairs. -/
theorem moved_sq_le [Nonempty V] (hl : ∀ x z z', sp (sp x z) z' = sp x z') (x : X) :
    (∑ z, if Mov sp opens (sp x z) then 1 else 0 : ℕ) ^ 2 ≤
      ∑ z1, ∑ z2, if Frk sp opens (sp x z1) z2 then 1 else 0 := by
  set c := plural sp opens x with hc
  have hmov : ∀ z, Mov sp opens (sp x z) ↔ ∃ v, opens (sp x z) v ∧ v ≠ c := fun z => by
    simp only [Mov, plural_sp hl, hc]
  have hfrk : ∀ z1 z2, Frk sp opens (sp x z1) z2 ↔ ∃ v v', opens (sp x z1) v ∧ opens (sp x z2) v' ∧ v ≠ v' :=
    fun z1 z2 => by simp only [Frk, hl]
  set Nn : Finset X := Finset.univ.filter fun z => ∃ v, opens (sp x z) v with hNn
  set Mv : Finset X := Finset.univ.filter fun z => Mov sp opens (sp x z) with hMv
  set Ec : Finset X := Finset.univ.filter fun z => Alone opens (sp x z) c with hEc
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
      · exact ⟨c, hv⟩
    · rintro ⟨v, hv⟩
      by_cases hall : ∀ v', opens (sp x z) v' → v' = c
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
    have h4 : (Finset.univ.filter fun z2 => Alone opens (sp x z2) v1).card ≤ Ec.card :=
      tally_le_plural (sp := sp) (opens := opens) x v1
    omega
  have hM : (∑ z, if Mov sp opens (sp x z) then 1 else 0 : ℕ) = Mv.card := by
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
/-- **Binding's failure, squared, is at most its fork form's**, for any continuation map with `spliceAt`'s laws. -/
theorem moved_le_forked [Nonempty V] (hl : ∀ x z z', sp (sp x z) z' = sp x z')
    (hs : ∀ x z, sp (sp x z) (sp z x) = x) :
    prob (fun x => Mov sp opens x) ^ 2 ≤ prob (fun p : X × X => Frk sp opens p.1 p.2) := by
  set C := Fintype.card X with hC
  set M : ℕ := ∑ y, if Mov sp opens y then 1 else 0 with hM
  set Fk : ℕ := ∑ y, ∑ z, if Frk sp opens y z then 1 else 0 with hFk
  have hA : ∑ x, (∑ z, if Mov sp opens (sp x z) then 1 else 0 : ℕ) = C * M :=
    sum_sp hs (fun y => if Mov sp opens y then 1 else 0)
  have hF : ∑ x, (∑ z1, ∑ z2, if Frk sp opens (sp x z1) z2 then 1 else 0 : ℕ) = C * Fk := by
    calc ∑ x, (∑ z1, ∑ z2, if Frk sp opens (sp x z1) z2 then 1 else 0 : ℕ)
        = ∑ x, ∑ z2, ∑ z1, (if Frk sp opens (sp x z1) z2 then 1 else 0 : ℕ) :=
          Finset.sum_congr rfl fun x _ => Finset.sum_comm
      _ = ∑ z2, ∑ x, ∑ z1, (if Frk sp opens (sp x z1) z2 then 1 else 0 : ℕ) := Finset.sum_comm
      _ = ∑ z2, C * ∑ y, (if Frk sp opens y z2 then 1 else 0 : ℕ) :=
          Finset.sum_congr rfl fun z2 _ => sum_sp hs (fun y => if Frk sp opens y z2 then 1 else 0)
      _ = C * Fk := by rw [← Finset.mul_sum, hFk, Finset.sum_comm]
  have hCS : (∑ x, (∑ z, if Mov sp opens (sp x z) then 1 else 0 : ℕ)) ^ 2 ≤
      C * ∑ x, (∑ z, if Mov sp opens (sp x z) then 1 else 0 : ℕ) ^ 2 := by
    simpa using sq_sum_le_card_mul_sum_sq (s := Finset.univ)
      (f := fun x => (∑ z, if Mov sp opens (sp x z) then 1 else 0 : ℕ))
  have h2 : (C * M) ^ 2 ≤ C * (C * Fk) := by
    rw [← hA, ← hF]
    exact hCS.trans (Nat.mul_le_mul_left _ (Finset.sum_le_sum fun x _ => moved_sq_le hl x))
  have hcM : (Finset.univ.filter fun x => Mov sp opens x).card = M := by rw [Finset.card_filter]
  have hcF : (Finset.univ.filter fun p : X × X => Frk sp opens p.1 p.2).card = Fk := by
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

end Plurality

namespace DrawnLayer

variable {C B : Type} [Fintype C] [Fintype B] {S m s : ℕ} {J Q Pos V : Type} [Nonempty V]
  (pf : Accounting.TS.DrawnLayer C B S m s J Q Pos V)

/-- Binding's failure at one position, squared, is at most its fork form's at the position's cut. -/
theorem moved_le_forked (ℓ : Pos) :
    prob (fun x => pf.Moved x ℓ) ^ 2 ≤
      prob (fun p : Paths (Paths C m × B) S × Paths (Paths C m × B) S => pf.Forked ℓ p.1 p.2) :=
  Plurality.moved_le_forked (sp := pf.cont ℓ) (opens := fun y v => pf.Opens y ℓ v)
    (spliceAt_left (pf.cut ℓ).1 (pf.cut ℓ).2) (spliceAt_swap (pf.cut ℓ).1 (pf.cut ℓ).2)

/-- A round's binding failure moves one of the positions its units read. -/
theorem moved_of_movedIn {x : Paths (Paths C m × B) S} {w : Fin S} (h : pf.MovedIn x w) :
    ∃ i k, pf.Moved x (pf.pos w (pf.drawnAt x w i) k) := by
  obtain ⟨i, hacc, hne⟩ := h
  obtain ⟨k, hk⟩ := Function.ne_iff.mp hne
  exact ⟨i, k, pf.opened x w i k, ⟨w, i, k, rfl, hacc, rfl⟩, hk⟩

omit [Fintype C] [Fintype B] [Nonempty V] in
open Classical in
/-- **A share that charges each unit the run didn't pay for its late cone's weight covers**: a unit that isn't good has
a unit of its late cone the run didn't pay for, whose cone weight counts it, at the run's modes. -/
theorem covers_of_cone [Fintype J] {K T L U : Type} (P : Play (Paths C m × B) S K T L U)
    (mode : Paths (Paths C m × B) S → Fin S × J → Mode) (wt : Fin S → J → ℝ) (Nw : ℝ) (hw : ∀ w j, 0 ≤ wt w j)
    (h : ∀ x, P.accepts x →
      (∑ u : Fin S × J, if mode x u = .paid then 0 else pf.coneWt wt u) ≤ P.game.deficiency x * Nw) :
    pf.Covers P mode wt Nw := by
  intro x hx
  refine le_trans ?_ (h x hx)
  rw [← Fintype.sum_prod_type' (fun w j => if pf.Good mode x w j then 0 else wt w j)]
  have key : ∀ v : Fin S × J, (if pf.Good mode x v.1 v.2 then 0 else wt v.1 v.2) ≤
      ∑ u : Fin S × J, if mode x u = .paid then 0 else
        (if Relation.ReflTransGen pf.Late u v then wt v.1 v.2 else 0) := by
    intro v
    by_cases hg : pf.Good mode x v.1 v.2
    · rw [ite_eq_left hg]
      exact Finset.sum_nonneg fun u _ => by split_ifs <;> simp [hw v.1 v.2]
    · rw [ite_eq_right hg]
      obtain ⟨u, hu, hb⟩ : ∃ u, Relation.ReflTransGen pf.Late u v ∧ ¬ mode x u = .paid :=
        Classical.byContradiction fun hno =>
          hg fun u hr => Classical.byContradiction fun hb => hno ⟨u, hr, hb⟩
      calc wt v.1 v.2 = (if mode x u = .paid then 0 else
            (if Relation.ReflTransGen pf.Late u v then wt v.1 v.2 else 0)) := by
            rw [ite_eq_right hb, ite_eq_left hu]
        _ ≤ _ := Finset.single_le_sum (f := fun u => if mode x u = .paid then 0 else
              (if Relation.ReflTransGen pf.Late u v then wt v.1 v.2 else 0))
            (fun u _ => by split_ifs <;> simp [hw v.1 v.2]) (Finset.mem_univ u)
  calc (∑ v : Fin S × J, if pf.Good mode x v.1 v.2 then 0 else wt v.1 v.2)
      ≤ ∑ v : Fin S × J, ∑ u : Fin S × J, (if mode x u = .paid then 0 else
          (if Relation.ReflTransGen pf.Late u v then wt v.1 v.2 else 0)) :=
        Finset.sum_le_sum fun v _ => key v
    _ = ∑ u : Fin S × J, ∑ v : Fin S × J, (if mode x u = .paid then 0 else
          (if Relation.ReflTransGen pf.Late u v then wt v.1 v.2 else 0)) := Finset.sum_comm
    _ = ∑ u : Fin S × J, if mode x u = .paid then 0 else pf.coneWt wt u := by
        refine Finset.sum_congr rfl fun u _ => ?_
        by_cases hb : mode x u = .paid
        · simp [hb]
        · simp [hb, Accounting.TS.DrawnLayer.coneWt]

end DrawnLayer

/-- A sampled window's drawn units are right at their committed values, outside binding and joint soundness failures. -/
theorem DrawnRightOff : Accounting.Guarantees.TS.DrawnRightOff := by
  intro J Q Pos V _ C B _ _ S m s pf δbind δks hB hZ hPin
  refine ⟨fun x => ∃ w ∈ (Finset.univ : Finset (Fin S)), pf.MovedIn x w ∨ ∃ i, pf.Unsound x w i, ?_, ?_⟩
  · calc prob _ ≤ ∑ w : Fin S, prob (fun x => pf.MovedIn x w ∨ ∃ i, pf.Unsound x w i) := prob_exists_le _ _
      _ ≤ ∑ _w : Fin S, (δbind + δks) :=
          Finset.sum_le_sum fun w _ => (prob_or_le _ _).trans (add_le_add (hB w) (hZ w))
      _ = S * (δbind + δks) := by simp only [Finset.sum_const, Finset.card_univ, Fintype.card_fin, nsmul_eq_mul]
  · intro x hE w i hacc
    have heq : pf.opened x w i = pf.view x w (pf.drawnAt x w i) := by
      by_contra hne
      exact hE ⟨w, Finset.mem_univ _, Or.inl ⟨i, hacc, hne⟩⟩
    apply hPin
    rw [← heq]
    by_contra hr
    exact hE ⟨w, Finset.mem_univ _, Or.inr ⟨i, hacc, hr⟩⟩

open Classical in
/-- (I8) for a sampled window, from `DrawnRightOff`'s pieces, the share's cover and E1(c) per unit. -/
theorem ProofsSoundOfDrawn : Accounting.Guarantees.TS.ProofsSoundOfDrawn := by
  intro K T L U J Q Pos V _ _ _ _ _ _ C B _ _ S m s P pf mode wt folded Nw δbind δks hS hsum hWR hOf hB hZ hPin hCov
    hUF
  obtain ⟨E, hE, hoff⟩ := DrawnRightOff C B S m s pf δbind δks hB hZ hPin
  refine ⟨E, by simpa only [Nat.cast_one, one_mul] using hE, fun x hx hEx => ⟨?_, ?_⟩⟩
  · exact hOf.2 x fun w i => hoff x hEx w i (hOf.1 x hx w i)
  · have hSpos : (0 : ℝ) < S := by exact_mod_cast hS
    have hsplit : (∑ w, ∑ j, if pf.Good mode x w j then wt w j else 0) +
        (∑ w, ∑ j, if pf.Good mode x w j then 0 else wt w j) = S * Nw := by
      rw [← hsum, ← Finset.sum_add_distrib]
      refine Finset.sum_congr rfl fun w _ => ?_
      rw [← Finset.sum_add_distrib]
      exact Finset.sum_congr rfl fun j _ => by split_ifs <;> ring
    calc (1 - P.game.deficiency x / S) * Nw = (S * Nw - P.game.deficiency x * Nw) / S := by
          field_simp
      _ ≤ (∑ w, ∑ j, if pf.Good mode x w j then wt w j else 0) / S :=
          div_le_div_of_nonneg_right (by linarith [hCov x hx]) hSpos.le
      _ ≤ (P.run x).fixed := (hUF hWR x hx).1

/-- Binding at every position's cut from the fork form: per position (`DrawnLayer.moved_le_forked`), then the union
over `L` and Cauchy–Schwarz. -/
theorem DrawnBindingOfFork : Accounting.Guarantees.TS.DrawnBindingOfFork := by
  intro J Q Pos V _ C B _ _ S m s pf w L hL
  set F := prob (fun p : Paths (Paths C m × B) S × Paths (Paths C m × B) S => ∃ ℓ ∈ L, pf.Forked ℓ p.1 p.2) with hF
  have hm : ∀ ℓ ∈ L, prob (fun x => pf.Moved x ℓ) ^ 2 ≤ F := fun ℓ hℓ =>
    (DrawnLayer.moved_le_forked pf ℓ).trans (prob_mono fun p h => ⟨ℓ, hℓ, h⟩)
  have h1 : prob (fun x => pf.MovedIn x w) ≤ ∑ ℓ ∈ L, prob (fun x => pf.Moved x ℓ) := by
    refine (prob_mono fun x hx => ?_).trans (prob_exists_le L fun ℓ x => pf.Moved x ℓ)
    obtain ⟨i, k, h⟩ := DrawnLayer.moved_of_movedIn pf hx
    exact ⟨_, hL _ k, h⟩
  have h2 : (∑ ℓ ∈ L, prob (fun x => pf.Moved x ℓ)) ^ 2 ≤ L.card * ∑ ℓ ∈ L, prob (fun x => pf.Moved x ℓ) ^ 2 :=
    sq_sum_le_card_mul_sum_sq
  have h3 : ∑ ℓ ∈ L, prob (fun x => pf.Moved x ℓ) ^ 2 ≤ L.card * F := by
    calc ∑ ℓ ∈ L, prob (fun x => pf.Moved x ℓ) ^ 2 ≤ ∑ _ℓ ∈ L, F := Finset.sum_le_sum hm
      _ = L.card * F := by rw [Finset.sum_const, nsmul_eq_mul]
  calc prob (fun x => pf.MovedIn x w) ^ 2
      ≤ (∑ ℓ ∈ L, prob (fun x => pf.Moved x ℓ)) ^ 2 := pow_le_pow_left₀ (prob_nonneg _) h1 2
    _ ≤ L.card * ∑ ℓ ∈ L, prob (fun x => pf.Moved x ℓ) ^ 2 := h2
    _ ≤ L.card * (L.card * F) := mul_le_mul_of_nonneg_left h3 (by positivity)
    _ = (L.card : ℝ) ^ 2 * F := by ring

end Accounting.SecurityProofs.TS
