module

public import Definitions.Slowdown.Assumptions.Cover
public import Definitions.Slowdown.Cover
import Mathlib.Algebra.Order.AbsoluteValue.Basic
import Mathlib.Algebra.Order.Field.Basic

/-! The cover theorem: paths into a block. -/

@[expose] public section

namespace Slowdown.Cover.SecurityProofs

theorem mem_inputs {C : Circuit} {I : Finset (Fin C.N)} {g : Fin C.N} :
    g ∈ C.inputs I ↔ g ∈ I ∧ (C.pred g = ∅ ∨ ¬ C.pred g ⊆ I) :=
  Finset.mem_filter

theorem mem_computed {C : Circuit} {I : Finset (Fin C.N)} {g : Fin C.N} :
    g ∈ C.computed I ↔ g ∈ I ∧ g ∉ C.inputs I :=
  Finset.mem_sdiff

theorem mem_outputs {C : Circuit} {I : Finset (Fin C.N)} {g : Fin C.N} :
    g ∈ C.outputs I ↔ g ∈ I ∧ ∀ h ∈ I, g ∉ C.pred h :=
  Finset.mem_filter

theorem computed_subset (C : Circuit) (I : Finset (Fin C.N)) : C.computed I ⊆ I :=
  Finset.sdiff_subset

theorem pred_subset_of_mem_computed {C : Circuit} {I : Finset (Fin C.N)} {g : Fin C.N} (hg : g ∈ C.computed I) :
    C.pred g ⊆ I := by
  obtain ⟨hgI, hgn⟩ := mem_computed.1 hg
  by_contra h
  exact hgn (mem_inputs.2 ⟨hgI, Or.inr h⟩)

theorem chain_entry {C : Circuit} {I : Finset (Fin C.N)} {l : List (Fin C.N)}
    (hl : l.IsChain fun u v => u ∈ C.pred v) {y : Fin C.N} (hy : l.getLast? = some y) (hyI : y ∈ I) :
    (∃ g ∈ l, g ∈ C.inputs I) ∨ ∀ g ∈ l, g ∈ C.computed I := by
  induction hl with
  | nil => simp at hy
  | singleton a =>
    simp only [List.getLast?_singleton, Option.some.injEq] at hy
    subst hy
    by_cases h : a ∈ C.inputs I
    · exact Or.inl ⟨a, List.mem_singleton_self a, h⟩
    · refine Or.inr fun g hg => ?_
      rw [List.mem_singleton.1 hg]
      exact mem_computed.2 ⟨hyI, h⟩
  | @cons_cons a b l hr _ ih =>
    rw [List.getLast?_cons_cons] at hy
    rcases ih hy with ⟨g, hg, hgI⟩ | hall
    · exact Or.inl ⟨g, List.mem_cons_of_mem a hg, hgI⟩
    · have haI : a ∈ I := pred_subset_of_mem_computed (hall b List.mem_cons_self) hr
      by_cases ha : a ∈ C.inputs I
      · exact Or.inl ⟨a, List.mem_cons_self, ha⟩
      · refine Or.inr fun g hg => ?_
        rcases List.mem_cons.1 hg with rfl | hg
        · exact mem_computed.2 ⟨haI, ha⟩
        · exact hall g hg

theorem entry (C : Circuit) (I : Finset (Fin C.N)) (π : List (Fin C.N)) (x y : Fin C.N) (hπ' : C.IsPath π x y)
    (hyI : y ∈ I) (hxc : x ∉ C.computed I) : ∃ g ∈ π, g ∈ C.inputs I := by
  obtain ⟨hx, hy, hπ⟩ := hπ'
  rcases chain_entry hπ hy hyI with h | hall
  · exact h
  · exact absurd (hall x (List.mem_of_mem_head? hx)) hxc

theorem backwardClosure (C : Circuit) (I A : Finset (Fin C.N)) (a x : Fin C.N) (haI : a ∈ I)
    (hA : C.inputs I ∩ A = ∅) (hreach : C.ReachesIn A x a) : x ∈ C.computed I := by
  obtain ⟨π, ⟨hx, hy, hπ⟩, hπA⟩ := hreach
  rcases chain_entry hπ hy haI with ⟨g, hg, hgI⟩ | hall
  · have : g ∈ C.inputs I ∩ A := Finset.mem_inter.2 ⟨hgI, hπA g hg⟩
    rw [hA] at this
    exact absurd this (Finset.notMem_empty g)
  · exact hall x (List.mem_of_mem_head? hx)

theorem isPath_cons {C : Circuit} {π : List (Fin C.N)} {x h y : Fin C.N} (hπ : C.IsPath π h y) (hx : x ∈ C.pred h) :
    C.IsPath (x :: π) x y := by
  obtain ⟨hh, hy, hc⟩ := hπ
  match π, hh, hy, hc with
  | h' :: π', hh, hy, hc =>
    simp only [List.head?_cons, Option.some.injEq] at hh
    subst hh
    exact ⟨rfl, by rw [List.getLast?_cons_cons]; exact hy, List.isChain_cons_cons.2 ⟨hx, hc⟩⟩

theorem reachesIn_output (C : Circuit) (I : Finset (Fin C.N)) (o : Fin C.N) (ho : C.outputs I = {o}) :
    ∀ x ∈ I, C.ReachesIn I x o := by
  suffices h : ∀ n, ∀ x : Fin C.N, C.N - x.val = n → x ∈ I → C.ReachesIn I x o from fun x hx => h _ x rfl hx
  intro n
  induction n using Nat.strong_induction_on with
  | _ n ih =>
    intro x hn hx
    by_cases hout : x ∈ C.outputs I
    · rw [ho, Finset.mem_singleton] at hout
      subst hout
      refine ⟨[x], ⟨rfl, rfl, List.IsChain.singleton x⟩, fun v hv => ?_⟩
      rw [List.mem_singleton.1 hv]
      exact hx
    · obtain ⟨h, hhI, hxh⟩ : ∃ h ∈ I, x ∈ C.pred h := by
        by_contra hne
        push Not at hne
        exact hout (mem_outputs.2 ⟨hx, hne⟩)
      have hlt : x.val < h.val := C.pred_lt h x hxh
      have hh : h.val < C.N := h.isLt
      obtain ⟨π, hπ, hπI⟩ := ih (C.N - h.val) (by omega) h rfl hhI
      refine ⟨x :: π, isPath_cons hπ hxh, fun v hv => ?_⟩
      rcases List.mem_cons.1 hv with rfl | hv
      · exact hx
      · exact hπI v hv

end Slowdown.Cover.SecurityProofs
