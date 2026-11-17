module

public import Mathlib.Data.Fintype.Card
public import Mathlib.Algebra.BigOperators.Group.Finset.Defs
import Mathlib.Algebra.BigOperators.Ring.Finset
import Mathlib.Tactic.Ring.RingNF

/-! Equal distributions of functions of uniform randomness (`SameDist`), written as fiber counts without division. -/

@[expose] public section

namespace FlockSoundness.ZK

open Finset

open Classical in
def SameDist {Ω Ω' V : Type*} [Fintype Ω] [Fintype Ω'] (f : Ω → V) (g : Ω' → V) : Prop :=
  ∀ v, #{ω | f ω = v} * Fintype.card Ω' = #{ω' | g ω' = v} * Fintype.card Ω

theorem card_comp_eq_sum {Ω V W : Type*} [Fintype Ω] [DecidableEq V] [DecidableEq W] (k : Ω → V) (φ : V → W)
    (w : W) (s : Finset V) (hk : ∀ ω, k ω ∈ s) :
    #{ω | φ (k ω) = w} = ∑ v ∈ s.filter (fun v => φ v = w), #{ω | k ω = v} := by
  rw [Finset.card_eq_sum_card_fiberwise (f := k) (t := s.filter fun v => φ v = w)
    (fun ω hω => mem_filter.2 ⟨hk ω, (mem_filter.1 hω).2⟩)]
  refine Finset.sum_congr rfl fun v hv => ?_
  rw [Finset.filter_filter]
  congr 1
  ext ω
  simp only [mem_filter, mem_univ, true_and] at hv ⊢
  constructor
  · exact fun h => h.2
  · intro h
    exact ⟨by rw [h]; exact hv.2, h⟩

theorem SameDist.comp {Ω Ω' V W : Type*} [Fintype Ω] [Fintype Ω'] {f : Ω → V} {g : Ω' → V}
    (h : SameDist f g) (φ : V → W) : SameDist (fun ω => φ (f ω)) (fun ω' => φ (g ω')) := by
  classical
  intro w
  let s : Finset V := univ.image f ∪ univ.image g
  have hf : ∀ ω, f ω ∈ s := fun ω => mem_union_left _ (mem_image_of_mem f (mem_univ ω))
  have hg : ∀ ω', g ω' ∈ s := fun ω' => mem_union_right _ (mem_image_of_mem g (mem_univ ω'))
  have e1 := card_comp_eq_sum f φ w s hf
  have e2 := card_comp_eq_sum g φ w s hg
  have hv : ∀ v, #{ω | f ω = v} * Fintype.card Ω' = #{ω' | g ω' = v} * Fintype.card Ω := fun v => by
    convert h v
  calc #{ω | φ (f ω) = w} * Fintype.card Ω'
      = ∑ v ∈ s.filter (fun v => φ v = w), #{ω | f ω = v} * Fintype.card Ω' := by
        rw [← Finset.sum_mul]; congr 1
    _ = ∑ v ∈ s.filter (fun v => φ v = w), #{ω' | g ω' = v} * Fintype.card Ω :=
        Finset.sum_congr rfl fun v _ => hv v
    _ = #{ω' | φ (g ω') = w} * Fintype.card Ω := by
        rw [← Finset.sum_mul]; congr 1; convert e2.symm using 3

theorem SameDist.symm {Ω Ω' V : Type*} [Fintype Ω] [Fintype Ω'] {f : Ω → V} {g : Ω' → V}
    (h : SameDist f g) : SameDist g f := fun v => (h v).symm

theorem SameDist.of_eq {Ω Ω' V : Type*} [Fintype Ω] [Fintype Ω'] {f f' : Ω → V} {g g' : Ω' → V}
    (h : SameDist f g) (hf : ∀ ω, f ω = f' ω) (hg : ∀ ω', g ω' = g' ω') : SameDist f' g' := by
  have ef : f = f' := funext hf
  have eg : g = g' := funext hg
  subst ef eg
  exact h

theorem card_eq_const_mul {Ω V : Type*} [Fintype Ω] [Fintype V] [DecidableEq V] (k : Ω → V) (c : ℕ)
    (hk : ∀ v, #{ω | k ω = v} = c) : Fintype.card Ω = c * Fintype.card V := by
  rw [← Finset.card_univ, Finset.card_eq_sum_card_fiberwise (f := k) (t := univ) (fun _ _ => mem_univ _)]
  rw [Finset.sum_congr rfl (fun v _ => by simpa only [Finset.filter_filter, mem_univ, true_and] using hk v),
    Finset.sum_const, Finset.card_univ]
  simp [mul_comm]

theorem SameDist.of_const {Ω Ω' V : Type*} [Fintype Ω] [Fintype Ω'] [Fintype V] [DecidableEq V]
    {f : Ω → V} {g : Ω' → V} {c c' : ℕ} (hf : ∀ v, #{ω | f ω = v} = c) (hg : ∀ v, #{ω' | g ω' = v} = c') :
    SameDist f g := by
  classical
  intro v
  rw [card_eq_const_mul f c hf, card_eq_const_mul g c' hg]
  convert (show c * (c' * Fintype.card V) = c' * (c * Fintype.card V) by ring) using 2
  · convert hf v
  · convert hg v

theorem SameDist.of_card_fiber_eq {Ω V : Type*} [Fintype Ω] [DecidableEq V] {f g : Ω → V}
    (h : ∀ v, #{ω | f ω = v} = #{ω | g ω = v}) : SameDist f g := by
  intro v
  congr 1
  convert h v

end FlockSoundness.ZK
