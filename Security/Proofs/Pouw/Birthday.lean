import Proofs.Pouw.PerCoin
import Proofs.Pouw.QTree

/-!
# The birthday bound for an adaptive program against the uniform oracle

PoUW's game draws `H : Q → R` uniformly. An adaptive program that makes at most `q` queries on every oracle finds a
collision on its own log (two distinct points it queried with the same answer) with probability at most
`q(q − 1)/(2|R|)`.

* `collides_append`: a collision on what was seen and then queried is one among what was seen, or a fresh hit
  (`FreshHit`): a query at a point not seen before whose answer is one already seen.
* `card_freshHit_le`: given the answers at the points seen, each fresh point's answer is uniform, so the `j`-th query
  is a fresh hit with probability at most `(|seen| + j)/|R|`. By induction on the program, conditioned on the answers
  at the points already seen.
* `pr_collides_le`: the bound for one program, and `pr_finder_collides_le` for the finder `B^A` of an adversary, at
  its query count `A.queries U`.
-/

namespace Pouw

open Finset

variable {Q R : Type}

namespace QTree

theorem queries_bind {α β : Type} (H : Q → R) (f : α → QTree Q R β) :
    ∀ t : QTree Q R α, (t.bind f).queries H = t.queries H ++ (f (t.eval H)).queries H
  | done a => by simp [bind, queries, eval]
  | query x k => by simp [bind, queries, eval, queries_bind H f (k (H x))]

/-- **A fresh hit**: some query of `t` against `H`, at a point outside `seen` and the points queried before it,
answers as one of those points does. -/
def FreshHit {α : Type} (H : Q → R) : List Q → QTree Q R α → Prop
  | _, done _ => False
  | seen, query x k => (x ∉ seen ∧ ∃ y ∈ seen, H y = H x) ∨ FreshHit H (seen ++ [x]) (k (H x))

end QTree

namespace Proofs

open QTree

/-- A collision on the points seen and then queried is one among the points seen, or a fresh hit. -/
theorem collides_append {α : Type} (H : Q → R) :
    ∀ (t : QTree Q R α) (seen : List Q), Collides H (seen ++ t.queries H) → Collides H seen ∨ FreshHit H seen t
  | done _, seen, h => Or.inl (by simpa [queries] using h)
  | query x k, seen, h => by
    have h' : Collides H ((seen ++ [x]) ++ (k (H x)).queries H) := by simpa [queries] using h
    rcases collides_append H (k (H x)) (seen ++ [x]) h' with hc | hc
    · obtain ⟨a, ha, b, hb, hab, hH⟩ := hc
      simp only [List.mem_append, List.mem_singleton] at ha hb
      by_cases hx : x ∈ seen
      · left
        rcases ha with ha | rfl <;> rcases hb with hb | rfl
        · exact ⟨a, ha, b, hb, hab, hH⟩
        · exact ⟨a, ha, b, hx, hab, hH⟩
        · exact ⟨a, hx, b, hb, hab, hH⟩
        · exact absurd rfl hab
      · rcases ha with ha | rfl <;> rcases hb with hb | rfl
        · exact Or.inl ⟨a, ha, b, hb, hab, hH⟩
        · exact Or.inr (Or.inl ⟨hx, a, ha, hH⟩)
        · exact Or.inr (Or.inl ⟨hx, b, hb, hH.symm⟩)
        · exact absurd rfl hab
    · exact Or.inr (Or.inr hc)

theorem bound_succ (q m : ℕ) :
    2 * (q + 1) * m + (q + 1) * (q + 1 - 1) = 2 * m + (2 * q * (m + 1) + q * (q - 1)) := by
  rcases q with _ | n
  · simp
  · simp only [Nat.add_sub_cancel]; ring

variable [Fintype Q] [DecidableEq Q] [Fintype R]

open Classical in
/-- Given the answers `ρ` at the points `seen`, every answer at a fresh point `x` is equally likely. -/
theorem card_fiber_eq (seen : List Q) (ρ : Q → R) {x : Q} (hx : x ∉ seen) (r r' : R) :
    ((univ.filter fun H : Q → R => ∀ y ∈ seen, H y = ρ y).filter fun H => H x = r).card =
      ((univ.filter fun H : Q → R => ∀ y ∈ seen, H y = ρ y).filter fun H => H x = r').card := by
  refine Finset.card_nbij' (fun H => Function.update H x r') (fun H => Function.update H x r) ?_ ?_ ?_ ?_
  · intro H hH
    simp only [Finset.mem_filter, Finset.coe_filter, Finset.mem_univ, true_and, Set.mem_ofPred_eq] at hH ⊢
    refine ⟨fun y hy => ?_, by simp⟩
    rw [Function.update_of_ne (show y ≠ x from fun h => hx (h ▸ hy))]
    exact hH.1 y hy
  · intro H hH
    simp only [Finset.mem_filter, Finset.coe_filter, Finset.mem_univ, true_and, Set.mem_ofPred_eq] at hH ⊢
    refine ⟨fun y hy => ?_, by simp⟩
    rw [Function.update_of_ne (show y ≠ x from fun h => hx (h ▸ hy))]
    exact hH.1 y hy
  · intro H hH
    simp only [Finset.mem_filter, Finset.coe_filter, Finset.mem_univ, true_and, Set.mem_ofPred_eq] at hH
    simp only [Function.update_idem]
    exact Function.update_eq_self_iff.2 hH.2.symm
  · intro H hH
    simp only [Finset.mem_filter, Finset.coe_filter, Finset.mem_univ, true_and, Set.mem_ofPred_eq] at hH
    simp only [Function.update_idem]
    exact Function.update_eq_self_iff.2 hH.2.symm

open Classical in
/-- **The fresh-hit count**, conditioned on the answers `ρ` at the points `seen`: for a program that makes at most `q`
queries wherever `H` agrees with `ρ` on `seen`, `2|R|` times the number of such `H` with a fresh hit is at most the
number of such `H` times `2·q·|seen| + q(q − 1)`. -/
theorem card_freshHit_le {α : Type} :
    ∀ (q : ℕ) (t : QTree Q R α) (seen : List Q) (ρ : Q → R),
    (∀ H : Q → R, (∀ y ∈ seen, H y = ρ y) → t.calls H ≤ q) →
    2 * Fintype.card R * (univ.filter fun H : Q → R => (∀ y ∈ seen, H y = ρ y) ∧ FreshHit H seen t).card ≤
      (univ.filter fun H : Q → R => ∀ y ∈ seen, H y = ρ y).card * (2 * q * seen.length + q * (q - 1)) := by
  classical
  intro q
  induction q with
  | zero =>
    intro t seen ρ hq
    have h0 : (univ.filter fun H : Q → R => (∀ y ∈ seen, H y = ρ y) ∧ FreshHit H seen t) = ∅ := by
      refine Finset.filter_eq_empty_iff.2 (by
        intro H _ hH
        cases t with
        | done a => simp [FreshHit] at hH
        | query x k => have := hq H hH.1; simp [calls] at this)
    simp [h0]
  | succ q ih =>
    intro t seen ρ hq
    cases t with
    | done a => simp [FreshHit]
    | query x k =>
      set A := univ.filter fun H : Q → R => ∀ y ∈ seen, H y = ρ y with hA
      have hcalls : ∀ H : Q → R, (∀ y ∈ seen, H y = ρ y) → (k (H x)).calls H ≤ q := fun H hH => by
        have := hq H hH; simp only [calls] at this; omega
      by_cases hx : x ∈ seen
      · set A' := univ.filter fun H : Q → R => ∀ y ∈ seen ++ [x], H y = ρ y
        have hagree : ∀ H : Q → R, (∀ y ∈ seen ++ [x], H y = ρ y) ↔ (∀ y ∈ seen, H y = ρ y) := fun H =>
          ⟨fun h y hy => h y (List.mem_append_left _ hy), fun h y hy => by
            rcases List.mem_append.1 hy with hy | hy
            · exact h y hy
            · rw [List.mem_singleton.1 hy]; exact h x hx⟩
        have hAA : A' = A := by
          ext H; simp only [A', hA, Finset.mem_filter, Finset.mem_univ, true_and]; exact hagree H
        have ih' := ih (k (ρ x)) (seen ++ [x]) ρ fun H hH => by
          have h1 := (hagree H).1 hH
          have := hcalls H h1; rwa [h1 x hx] at this
        have hsub : (univ.filter fun H : Q → R => (∀ y ∈ seen, H y = ρ y) ∧ FreshHit H seen (query x k)) ⊆
            univ.filter fun H : Q → R => (∀ y ∈ seen ++ [x], H y = ρ y) ∧ FreshHit H (seen ++ [x]) (k (ρ x)) := by
          rw [Finset.subset_iff]; intro H
          simp only [Finset.mem_filter, Finset.mem_univ, true_and, FreshHit]
          rintro ⟨hH, (⟨hn, -⟩ | hf)⟩
          · exact absurd hx hn
          · exact ⟨(hagree H).2 hH, by rwa [hH x hx] at hf⟩
        rw [← hAA]
        calc 2 * Fintype.card R *
              (univ.filter fun H : Q → R => (∀ y ∈ seen, H y = ρ y) ∧ FreshHit H seen (query x k)).card
            ≤ 2 * Fintype.card R * (univ.filter fun H : Q → R =>
                (∀ y ∈ seen ++ [x], H y = ρ y) ∧ FreshHit H (seen ++ [x]) (k (ρ x))).card :=
              Nat.mul_le_mul_left _ (Finset.card_le_card hsub)
          _ ≤ A'.card * (2 * q * (seen ++ [x]).length + q * (q - 1)) := ih'
          _ ≤ A'.card * (2 * (q + 1) * seen.length + (q + 1) * (q + 1 - 1)) := by
              refine Nat.mul_le_mul_left _ ?_
              rw [bound_succ, List.length_append, List.length_singleton]
              exact Nat.le_add_left _ _
      · -- the fibers of `A` by the answer at `x`
        let F : R → Finset (Q → R) := fun r => A.filter fun H => H x = r
        have hfib : A.card = ∑ r, (F r).card :=
          Finset.card_eq_sum_card_fiberwise (f := fun H : Q → R => H x) (by intro _ _; exact Finset.mem_univ _)
        have hunif : ∀ r, Fintype.card R * (F r).card = A.card := fun r => by
          rw [hfib]
          symm
          calc ∑ r', (F r').card = ∑ _r' : R, (F r).card :=
                Finset.sum_congr rfl fun r' _ => card_fiber_eq seen ρ hx r' r
            _ = Fintype.card R * (F r).card := by rw [Finset.sum_const, Finset.card_univ, smul_eq_mul]
        let E1 := univ.filter fun H : Q → R => (∀ y ∈ seen, H y = ρ y) ∧ ∃ y ∈ seen, H y = H x
        let E2 := univ.filter fun H : Q → R =>
          (∀ y ∈ seen, H y = ρ y) ∧ FreshHit H (seen ++ [x]) (k (H x))
        have hsub : (univ.filter fun H : Q → R => (∀ y ∈ seen, H y = ρ y) ∧ FreshHit H seen (query x k)) ⊆
            E1 ∪ E2 := by
          rw [Finset.subset_iff]; intro H
          simp only [Finset.mem_filter, Finset.mem_univ, true_and, FreshHit, Finset.mem_union, E1, E2]
          rintro ⟨hH, (⟨-, h⟩ | hf)⟩
          · exact Or.inl ⟨hH, h⟩
          · exact Or.inr ⟨hH, hf⟩
        -- the first query is a fresh hit on at most `m` of the `|R|` fibers
        have h1 : Fintype.card R * E1.card ≤ seen.length * A.card := by
          let img := (seen.map ρ).toFinset
          have hE1 : E1.card = ∑ r ∈ img, (F r).card := by
            rw [Finset.card_eq_sum_card_fiberwise (f := fun H : Q → R => H x) (t := img)]
            · refine Finset.sum_congr rfl fun r hr => congrArg Finset.card ?_
              ext H
              simp only [Finset.mem_filter, Finset.mem_univ, true_and, E1, F, hA]
              constructor
              · rintro ⟨⟨hH, -⟩, rfl⟩; exact ⟨hH, rfl⟩
              · rintro ⟨hH, rfl⟩
                refine ⟨⟨hH, ?_⟩, rfl⟩
                obtain ⟨y, hy, hy'⟩ := List.mem_map.1 (List.mem_toFinset.1 hr)
                exact ⟨y, hy, (hH y hy).trans hy'⟩
            · intro H hH
              simp only [Finset.mem_filter, Finset.coe_filter, Finset.mem_univ, true_and, Set.mem_ofPred_eq,
                E1] at hH
              obtain ⟨hH, y, hy, hy'⟩ := hH
              exact List.mem_toFinset.2 (List.mem_map.2 ⟨y, hy, (hH y hy).symm.trans hy'⟩)
          rw [hE1, Finset.mul_sum]
          calc ∑ r ∈ img, Fintype.card R * (F r).card = ∑ _r ∈ img, A.card :=
                Finset.sum_congr rfl fun r _ => hunif r
            _ = img.card * A.card := by rw [Finset.sum_const, smul_eq_mul]
            _ ≤ seen.length * A.card := Nat.mul_le_mul_right _ ((List.toFinset_card_le _).trans (by simp))
        -- later queries: the fiber at `r` runs `k r` with `x` seen and answering `r`
        have h2 : 2 * Fintype.card R * E2.card ≤ A.card * (2 * q * (seen.length + 1) + q * (q - 1)) := by
          have hE2 : E2.card = ∑ r, (E2.filter fun H => H x = r).card :=
            Finset.card_eq_sum_card_fiberwise (f := fun H : Q → R => H x) (by intro _ _; exact Finset.mem_univ _)
          have hr : ∀ r, 2 * Fintype.card R * (E2.filter fun H => H x = r).card ≤
              (F r).card * (2 * q * (seen.length + 1) + q * (q - 1)) := fun r => by
            have hagree : ∀ H : Q → R, (∀ y ∈ seen ++ [x], H y = Function.update ρ x r y) ↔
                (∀ y ∈ seen, H y = ρ y) ∧ H x = r := fun H => by
              constructor
              · intro h
                refine ⟨fun y hy => ?_, by simpa using h x (List.mem_append_right _ (List.mem_singleton_self x))⟩
                have := h y (List.mem_append_left _ hy)
                rwa [Function.update_of_ne (show y ≠ x from fun e => hx (e ▸ hy))] at this
              · rintro ⟨h, rfl⟩ y hy
                rcases List.mem_append.1 hy with hy | hy
                · rw [Function.update_of_ne (show y ≠ x from fun e => hx (e ▸ hy))]; exact h y hy
                · rw [List.mem_singleton.1 hy]; simp
            have ih' := ih (k r) (seen ++ [x]) (Function.update ρ x r) fun H hH => by
              obtain ⟨h1, h2⟩ := (hagree H).1 hH
              have := hcalls H h1; rwa [h2] at this
            have hl : (seen ++ [x]).length = seen.length + 1 := by simp
            rw [hl] at ih'
            have hE : (E2.filter fun H => H x = r) = univ.filter fun H : Q → R =>
                (∀ y ∈ seen ++ [x], H y = Function.update ρ x r y) ∧ FreshHit H (seen ++ [x]) (k r) := by
              ext H
              simp only [Finset.mem_filter, Finset.mem_univ, true_and, E2, hagree]
              constructor
              · rintro ⟨⟨h, hf⟩, rfl⟩; exact ⟨⟨h, rfl⟩, hf⟩
              · rintro ⟨⟨h, rfl⟩, hf⟩; exact ⟨⟨h, hf⟩, rfl⟩
            have hF : F r = univ.filter fun H : Q → R => ∀ y ∈ seen ++ [x], H y = Function.update ρ x r y := by
              ext H
              simp only [Finset.mem_filter, Finset.mem_univ, true_and, F, hA, hagree]
            rw [hE, hF]
            exact ih'
          rw [hE2, Finset.mul_sum]
          calc ∑ r, 2 * Fintype.card R * (E2.filter fun H => H x = r).card
              ≤ ∑ r, (F r).card * (2 * q * (seen.length + 1) + q * (q - 1)) := Finset.sum_le_sum fun r _ => hr r
            _ = A.card * (2 * q * (seen.length + 1) + q * (q - 1)) := by rw [← Finset.sum_mul, ← hfib]
        calc 2 * Fintype.card R *
              (univ.filter fun H : Q → R => (∀ y ∈ seen, H y = ρ y) ∧ FreshHit H seen (query x k)).card
            ≤ 2 * Fintype.card R * (E1.card + E2.card) :=
              Nat.mul_le_mul_left _ ((Finset.card_le_card hsub).trans (Finset.card_union_le _ _))
          _ = 2 * (Fintype.card R * E1.card) + 2 * Fintype.card R * E2.card := by ring
          _ ≤ 2 * (seen.length * A.card) + A.card * (2 * q * (seen.length + 1) + q * (q - 1)) := by
              linarith [h1, h2]
          _ = A.card * (2 * (q + 1) * seen.length + (q + 1) * (q + 1 - 1)) := by rw [bound_succ]; ring

/-- **The birthday bound for one program**: a program that makes at most `q` queries on every oracle collides on its
log against the uniform oracle with probability at most `q(q − 1)/(2|R|)`. -/
theorem pr_collides_le {α : Type} (t : QTree Q R α) (q : ℕ) (ht : ∀ H, t.calls H ≤ q) :
    pr (fun H : Q → R => Collides H (t.queries H)) ≤ (q : ℝ) * (q - 1) / (2 * Fintype.card R) := by
  classical
  have hq0 : 0 ≤ (q : ℝ) * (q - 1) := by
    rcases Nat.eq_zero_or_pos q with h | h
    · simp [h]
    · exact mul_nonneg (Nat.cast_nonneg _) (sub_nonneg.2 (by exact_mod_cast h))
  rcases Nat.eq_zero_or_pos (Fintype.card R) with hR | hR
  · rw [pr_eq_zero fun H ⟨x, _, _⟩ => absurd hR (Fintype.card_pos_iff.2 ⟨H x⟩).ne']
    simp [hR]
  haveI : Nonempty R := Fintype.card_pos_iff.1 hR
  have hc := card_freshHit_le q t [] (fun _ => Classical.arbitrary R) fun H _ => ht H
  have e1 : (univ.filter fun H : Q → R => (∀ y ∈ ([] : List Q), H y = Classical.arbitrary R) ∧ FreshHit H [] t) =
      univ.filter fun H => FreshHit H [] t := by ext H; simp
  have e2 : (univ.filter fun H : Q → R => ∀ y ∈ ([] : List Q), H y = Classical.arbitrary R) = univ := by
    ext H; simp
  rw [e1, e2, Finset.card_univ] at hc
  simp only [List.length_nil, mul_zero, zero_add] at hc
  have hsub : (univ.filter fun H : Q → R => Collides H (t.queries H)) ⊆ univ.filter fun H => FreshHit H [] t := by
    rw [Finset.subset_iff]; intro H
    simp only [Finset.mem_filter, Finset.mem_univ, true_and]
    intro h
    rcases collides_append H t [] (by simpa using h) with ⟨_, h, _⟩ | h
    · simp at h
    · exact h
  have hpos : (0 : ℝ) < Fintype.card (Q → R) := by exact_mod_cast Fintype.card_pos
  unfold pr
  rw [div_le_div_iff₀ hpos (by positivity)]
  have hc' : (2 * Fintype.card R * (univ.filter fun H : Q → R => FreshHit H [] t).card : ℝ) ≤
      Fintype.card (Q → R) * ((q * (q - 1) : ℕ) : ℝ) := by exact_mod_cast hc
  have hcast : ((q * (q - 1) : ℕ) : ℝ) = (q : ℝ) * (q - 1) := by
    rcases Nat.eq_zero_or_pos q with h | h
    · simp [h]
    · rw [Nat.cast_mul, Nat.cast_sub h, Nat.cast_one]
  have hle : ((univ.filter fun H : Q → R => Collides H (t.queries H)).card : ℝ) ≤
      (univ.filter fun H : Q → R => FreshHit H [] t).card := by exact_mod_cast Finset.card_le_card hsub
  rw [hcast] at hc'
  calc ((univ.filter fun H : Q → R => Collides H (t.queries H)).card : ℝ) * (2 * Fintype.card R)
      ≤ (univ.filter fun H : Q → R => FreshHit H [] t).card * (2 * Fintype.card R) :=
        mul_le_mul_of_nonneg_right hle (by positivity)
    _ = 2 * Fintype.card R * (univ.filter fun H : Q → R => FreshHit H [] t).card := by ring
    _ ≤ Fintype.card (Q → R) * ((q : ℝ) * (q - 1)) := hc'
    _ = (q : ℝ) * (q - 1) * Fintype.card (Q → R) := by ring

/-- **The birthday bound for the finder `B^A`**: over the uniform oracle and salt, the finder built from `A` collides
on its log with probability at most `q_A(q_A − 1)/(2|R|)`, at `q_A = A.queries U`. -/
theorem pr_finder_collides_le [Fintype S] {CM : CostModel Q R S} (hQ : CM.Queries) (A : Adversary CM) (U : Workload) :
    pr (fun x : (Q → R) × S => Collides x.1 (finderQueries hQ A U x.1 x.2)) ≤
      (A.queries U : ℝ) * (A.queries U - 1) / (2 * Fintype.card R) := by
  classical
  have hq0 : 0 ≤ (A.queries U : ℝ) * (A.queries U - 1) / (2 * Fintype.card R) := by
    refine div_nonneg ?_ (by positivity)
    rcases Nat.eq_zero_or_pos (A.queries U) with h | h
    · simp [h]
    · exact mul_nonneg (Nat.cast_nonneg _) (sub_nonneg.2 (by exact_mod_cast h))
  rw [← pr_equiv (Equiv.prodComm S (Q → R)) (fun x : (Q → R) × S => Collides x.1 (finderQueries hQ A U x.1 x.2))]
  refine pr_prod_le _ _ hq0 fun s => ?_
  let t := A.pre.bind fun σ => CM.tree hQ A.online U A.act σ s
  have hfq : ∀ H, finderQueries hQ A U H s = t.queries H := fun H => by
    simp only [t, QTree.queries_bind, finderQueries]
  have hcalls : ∀ H, t.calls H ≤ A.queries U := fun H => by
    have h := (Classical.choose_spec (hQ A.online U A.act (A.pre.eval H) s) H).2
    simp only [t, QTree.calls_bind]
    rw [show CM.tree hQ A.online U A.act (A.pre.eval H) s =
      Classical.choose (hQ A.online U A.act (A.pre.eval H) s) from rfl, ← h]
    exact Finset.le_sup (f := fun x : (Q → R) × S => A.pre.calls x.1 + CM.calls A.online x.1 U A.act
      (A.pre.eval x.1) x.2) (Finset.mem_univ (H, s))
  simp only [Equiv.prodComm_apply, Prod.swap_prod_mk, hfq]
  exact pr_collides_le t _ hcalls

end Proofs

end Pouw
