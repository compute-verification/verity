import Models.Accounting.Guarantees.FanIn
import Mathlib.Data.Fintype.Card

/-!
# The fan-in floor's proofs

`deps P v` is the set of inputs node `v` reads through every instruction it depends on. A node's value is a function of
its deps (`eval_local`), so every essential input of `P.run out` is in the deps of an output. The deps of a set `S` of
nodes number at most `|S| + (f − 1)·m` (`card_deps_le`, by induction on the program): replacing the last instruction
in `S` by its `f` sources adds at most `f − 1` nodes and keeps the deps. A fold of binding steps changes when any one
word does (`foldRun_ne`), so all `N` of its words are essential.
-/

namespace Accounting.FanInProofs

open Accounting.Program

/-- The inputs node `v` reads through every instruction it depends on. -/
def deps {α : Type} {n f : ℕ} : {m : ℕ} → Prog α n f m → Fin (n + m) → Finset (Fin n)
  | _, .nil => fun i => {i}
  | _, .snoc P src _ => Fin.lastCases (motive := fun _ => Finset (Fin n))
      (Finset.univ.biUnion fun k => deps P (src k)) (deps P)

theorem eval_local {α : Type} {n f m : ℕ} (P : Prog α n f m) (x y : Fin n → α) :
    ∀ v, (∀ j ∈ deps P v, x j = y j) → P.eval x v = P.eval y v := by
  induction P with
  | nil => exact fun v h => h v (Finset.mem_singleton_self v)
  | snoc P src op ih =>
    intro v
    refine Fin.lastCases ?_ (fun w => ?_) v
    · intro h
      simp only [deps, Fin.lastCases_last] at h
      simp only [Prog.eval, Fin.snoc_last]
      congr 1
      funext k
      exact ih (src k) fun j hj => h j (Finset.mem_biUnion.2 ⟨k, Finset.mem_univ _, hj⟩)
    · intro h
      simp only [deps, Fin.lastCases_castSucc] at h
      simp only [Prog.eval, Fin.snoc_castSucc]
      exact ih w h

theorem card_deps_le {α : Type} {n f m : ℕ} (P : Prog α n f m) :
    ∀ S : Finset (Fin (n + m)), (S.biUnion (deps P)).card ≤ S.card + (f - 1) * m := by
  classical
  induction P with
  | nil =>
    intro S
    simp only [deps, Finset.biUnion_singleton_eq_self]
    omega
  | @snoc m P src op ih =>
    intro S
    let pre : Finset (Fin (n + m)) := Finset.univ.filter fun w => w.castSucc ∈ S
    let top : Finset (Fin (n + m)) := if Fin.last (n + m) ∈ S then Finset.univ.image src else ∅
    have hsub : S.biUnion (deps (P.snoc src op)) ⊆ (pre ∪ top).biUnion (deps P) := by
      intro j hj
      obtain ⟨v, hv, hjv⟩ := Finset.mem_biUnion.1 hj
      revert hv hjv
      refine Fin.lastCases ?_ (fun w => ?_) v
      · intro hv hjv
        simp only [deps, Fin.lastCases_last] at hjv
        obtain ⟨k, -, hk⟩ := Finset.mem_biUnion.1 hjv
        refine Finset.mem_biUnion.2 ⟨src k, Finset.mem_union_right _ ?_, hk⟩
        have hv' : Fin.last (n + m) ∈ S := hv
        simp only [top, hv', ↓reduceIte]
        exact Finset.mem_image_of_mem src (Finset.mem_univ k)
      · intro hv hjv
        simp only [deps, Fin.lastCases_castSucc] at hjv
        exact Finset.mem_biUnion.2 ⟨w, Finset.mem_union_left _ (by simp [pre, hv]), hjv⟩
    have htop : top.card ≤ if Fin.last (n + m) ∈ S then f else 0 := by
      simp only [top]
      split_ifs
      · exact (Finset.card_image_le).trans (by simp)
      · simp
    have hpre : pre.card + (if Fin.last (n + m) ∈ S then 1 else 0) ≤ S.card := by
      have hmap : pre.map Fin.castSuccEmb ⊆ S.erase (Fin.last (n + m)) := by
        intro v hv
        obtain ⟨w, hw, rfl⟩ := Finset.mem_map.1 hv
        simp only [pre, Finset.mem_filter, Finset.mem_univ, true_and] at hw
        exact Finset.mem_erase.2 ⟨Fin.castSucc_ne_last w, hw⟩
      have h1 := Finset.card_le_card hmap
      rw [Finset.card_map] at h1
      split_ifs with hl
      · have := Finset.card_erase_add_one hl
        omega
      · rw [Finset.erase_eq_self.2 hl] at h1
        omega
    have h1 := Finset.card_le_card hsub
    have h2 := ih (pre ∪ top)
    have h3 := Finset.card_union_le pre top
    have hm : (f - 1) * (m + 1) = (f - 1) * m + (f - 1) := Nat.mul_succ _ _
    rw [hm]
    generalize (f - 1) * m = t at h2 ⊢
    split_ifs at htop hpre <;> omega

theorem essential_subset {α : Type} {n f m c : ℕ} (P : Prog α n f m) (out : Fin c → Fin (n + m)) :
    {i | Essential (P.run out) i} ⊆ ↑((Finset.univ.image out).biUnion (deps P)) := by
  classical
  intro i hi
  by_contra hn
  obtain ⟨x, y, hxy, hne⟩ := hi
  apply hne
  funext o
  refine eval_local P x y (out o) fun j hj => hxy j ?_
  rintro rfl
  exact hn (Finset.mem_coe.2 (Finset.mem_biUnion.2 ⟨out o, Finset.mem_image_of_mem _ (Finset.mem_univ _), hj⟩))

theorem fanInFloor {α : Type} {n f m c : ℕ} (P : Prog α n f m) (out : Fin c → Fin (n + m)) :
    {i | Essential (P.run out) i}.ncard ≤ c + (f - 1) * m := by
  classical
  calc {i | Essential (P.run out) i}.ncard
      ≤ (↑((Finset.univ.image out).biUnion (deps P)) : Set (Fin n)).ncard :=
        Set.ncard_le_ncard (essential_subset P out) (Finset.finite_toSet _)
    _ = ((Finset.univ.image out).biUnion (deps P)).card := Set.ncard_coe_finset _
    _ ≤ (Finset.univ.image out).card + (f - 1) * m := card_deps_le P _
    _ ≤ c + (f - 1) * m := by
        have := Finset.card_image_le (s := Finset.univ) (f := out)
        simp only [Finset.card_univ, Fintype.card_fin] at this
        omega

/-- Two runs of a fold of binding steps that differ at one word, and only there, end in different states. -/
theorem foldRun_ne {α : Type} {c : ℕ} :
    ∀ {N : ℕ} (g : Fin N → (Fin c → α) → α → (Fin c → α)), (∀ i s, Function.Injective (g i s)) →
      (∀ i d, Function.Injective fun s => g i s d) → ∀ (s : Fin c → α) (i : Fin N) (d e : Fin N → α),
      (∀ j, j ≠ i → d j = e j) → d i ≠ e i → foldRun g s d ≠ foldRun g s e := by
  intro N
  induction N with
  | zero => exact fun _ _ _ _ i => i.elim0
  | succ N ih =>
    intro g hw hs s i d e hde hi
    simp only [foldRun, Fin.foldl_succ_last]
    revert hde hi
    refine Fin.lastCases ?_ (fun i' => ?_) i
    · intro hde hi
      have hpre : (Fin.foldl N (fun t j => g j.castSucc t (d j.castSucc)) s) =
          Fin.foldl N (fun t j => g j.castSucc t (e j.castSucc)) s := by
        congr 1
        funext t j
        rw [hde j.castSucc (Fin.castSucc_ne_last j)]
      rw [hpre]
      exact fun h => hi (hw _ _ h)
    · intro hde hi
      have hlast : d (Fin.last N) = e (Fin.last N) := hde _ (Fin.castSucc_ne_last i').symm
      rw [hlast]
      intro h
      refine ih (fun j => g j.castSucc) (fun j => hw j.castSucc) (fun j => hs j.castSucc) s i'
        (fun j => d j.castSucc) (fun j => e j.castSucc) (fun j hj => hde _ ?_) hi (hs _ _ h)
      exact fun h' => hj (Fin.castSucc_injective _ h')

theorem foldFanIn {w N f m c : ℕ} (g : Fin N → BindingStep c (BitVec (w + 1))) (s : Fin c → BitVec (w + 1))
    (P : Prog (BitVec (w + 1)) N f m) (out : Fin c → Fin (N + m)) :
    (∃ d, P.run out d ≠ foldRun (fun i => (g i).1) s d) ∨ N ≤ c + (f - 1) * m := by
  classical
  by_cases hP : ∃ d, P.run out d ≠ foldRun (fun i => (g i).1) s d
  · exact Or.inl hP
  right
  simp only [not_exists, not_not] at hP
  have hall : {i | Essential (P.run out) i} = Set.univ := by
    refine Set.eq_univ_of_forall fun i => ?_
    refine ⟨fun _ => 0, Function.update (fun _ => 0) i 1, fun j hj => by simp [hj], ?_⟩
    rw [hP, hP]
    exact foldRun_ne _ (fun j => (g j).2.1) (fun j => (g j).2.2) s i _ _
      (fun j hj => by simp [hj]) (by simp)
  have := fanInFloor P out
  rwa [hall, Set.ncard_univ, Nat.card_eq_fintype_card, Fintype.card_fin] at this

end Accounting.FanInProofs

namespace Accounting.SecurityProofs

theorem FanInFloor : Accounting.Guarantees.FanInFloor :=
  fun _ _ _ _ _ P out => Accounting.FanInProofs.fanInFloor P out

theorem FoldFanIn : Accounting.Guarantees.FoldFanIn :=
  fun _ _ _ _ _ g s P out => Accounting.FanInProofs.foldFanIn g s P out

end Accounting.SecurityProofs
