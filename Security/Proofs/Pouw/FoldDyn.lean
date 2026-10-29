import Mathlib

/-!
# FOLD/DYNAMIC's counting step

A fold member θ, drawn by live coins, is computed by some program of a finite set `P` only when `f θ` lies in the image
of the programs' semantics, so the members some program computes number at most `|P|` times the largest fibre of the
member map `f`. With `P` the straight-line programs of `L` instructions (`Prog`), described by each instruction's form
and operands and by the result that is the output, that is the counting theorem `fold_dynamic.py` evaluates (in
`benchmarks/pouw/theory/`): over `2^(bT)` members whose fibres hold at most `2^((b - b')T)`, at most a
`|Prog| · 2^(-b'T)` share has a program of `L` instructions, whatever each instruction computes.
-/

namespace Pouw.FoldDyn

open Finset

/-- The members some program computes are at most `|P|` times the largest fibre of the member map. -/
theorem card_computed_le {Θ P G : Type*} [Fintype Θ] [Fintype P] [DecidableEq G]
    (eval : P → G) (f : Θ → G) (M : ℕ)
    (hM : ∀ g, (univ.filter fun θ => f θ = g).card ≤ M) :
    (univ.filter fun θ => ∃ p, eval p = f θ).card ≤ Fintype.card P * M := by
  classical
  calc (univ.filter fun θ => ∃ p, eval p = f θ).card
      ≤ (univ.biUnion fun p => univ.filter fun θ => f θ = eval p).card := by
        apply card_le_card
        intro θ hθ
        simp only [mem_filter, mem_univ, true_and] at hθ
        obtain ⟨p, hp⟩ := hθ
        simp only [mem_biUnion, mem_univ, mem_filter, true_and]
        exact ⟨p, hp.symm⟩
    _ ≤ ∑ p, (univ.filter fun θ => f θ = eval p).card := card_biUnion_le
    _ ≤ ∑ _p : P, M := sum_le_sum fun p _ => hM (eval p)
    _ = Fintype.card P * M := by rw [sum_const, card_univ, smul_eq_mul]

/-- One instruction's description among `m` inputs and earlier results: its form (opcode, modifiers, immediate shift
or LUT, predicate guard), which of its 3 operands is a variable, that operand, and the other two operands (each a
variable or a 32-bit constant). -/
abbrev Instr (F m : ℕ) := Fin F × Fin 3 × Fin m × Fin (m + 2 ^ 32) × Fin (m + 2 ^ 32)

/-- A straight-line program of `L` instructions over `n` inputs, each result able to be two words: the result that is
its output, and its instructions. -/
abbrev Prog (F n L : ℕ) := Fin (n + 2 * L) × (Fin L → Instr F (n + 2 * L))

theorem card_prog (F n L : ℕ) :
    Fintype.card (Prog F n L) =
      (n + 2 * L) * (F * (3 * ((n + 2 * L) * ((n + 2 * L + 2 ^ 32) * (n + 2 * L + 2 ^ 32))))) ^ L := by
  simp [Prog, Instr, Fintype.card_prod, Fintype.card_fin]

/-- The counting theorem: when every fibre of the member map holds at most `M` members, the members some program of
`L` instructions computes, under any semantics `eval`, number at most `m · (F · 3m · (m + 2^32)^2)^L · M`, with
`m = n + 2L`. -/
theorem card_computed_by_short_le {Θ G : Type*} [Fintype Θ] [DecidableEq G] (F n L : ℕ)
    (eval : Prog F n L → G) (f : Θ → G) (M : ℕ)
    (hM : ∀ g, (univ.filter fun θ => f θ = g).card ≤ M) :
    (univ.filter fun θ => ∃ p, eval p = f θ).card ≤
      (n + 2 * L) * (F * (3 * ((n + 2 * L) * ((n + 2 * L + 2 ^ 32) * (n + 2 * L + 2 ^ 32))))) ^ L * M := by
  rw [← card_prog]
  exact card_computed_le eval f M hM

end Pouw.FoldDyn
