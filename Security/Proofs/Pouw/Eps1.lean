import Models.Pouw.Guarantees.Eps1

/-!
# Dither's CUDA-core lower bounds in M3: proofs (stubs)

Each guarantee of `Models.Pouw.Guarantees.Eps1` as `Pouw.SecurityProofs.Eps1.G`, and the lemmas QSTAR's proofs need,
each a `sorry` until a prover takes it (the gap list: note:20261017T0801Z-draft-lean-stmt).
-/

namespace Pouw.SecurityProofs.Eps1

open Pouw.Eps1

/-- Every M3 program has a run on every input (instructions read only earlier words). -/
theorem run_exists {ι : Type} {G : ℕ} (P : M3 ι G) (x : ι → Word) : ∃ v, P.IsRun x v := by
  sorry

/-- The run is unique. -/
theorem run_unique {ι : Type} {G : ℕ} (P : M3 ι G) (x : ι → Word) (v v' : Fin G → Word) (h : P.IsRun x v)
    (h' : P.IsRun x v') : v = v' := by
  sorry

open Classical in
/-- Theorem B's counting: a block whose live part has `a` instructions, each but the writer an argument of another
live one, takes at most `2a + 1` of its input slots from outside it (at least `a − 1` are edges inside it). -/
theorem block_outside_slots {Rd Cd : Type} {m n Lseg : ℕ} {FR FU : Type} {G : ℕ}
    (P : Prog Rd Cd m n Lseg FR FU G) (blk : Fin G → Tau m n Lseg) (hW : P.WordLocal blk) (τ : Tau m n Lseg)
    (live : Finset (Fin G)) (hlive : ∀ t ∈ live, blk t = τ) (hout : P.out τ ∈ live)
    (hused : ∀ t ∈ live, t ≠ P.out τ → ∃ t' ∈ live, ∃ j s, P.prog.arg t' j = .inr s ∧ M3.earlier t' s = t) :
    ((Finset.univ : Finset (Fin 3)).product live |>.filter fun tj =>
      ∀ s, P.prog.arg tj.2 tj.1 = .inr s → M3.earlier tj.2 s ∉ live).card ≤ 2 * live.card + 1 := by
  sorry

/-- The zero instance's fiber: `r ↦ r·u` determines a column `u` with entries in `±1..±64`, so the column-side
words a block reads carry at least 224 bits, at least 7 words. -/
theorem fiber_needs_seven {n : ℕ} (U U' : Cols n) (j : Fin n) (hU : ColsValid U) (hU' : ColsValid U')
    (h : ∀ r : Fin 32 → Bool, (∑ p, if r p then U j p else 0) = ∑ p, if r p then U' j p else 0) :
    ∀ p, U j p = U' j p := by
  sorry

/-- The information count: six words can't tell apart the `128^32 = 2^224` columns with entries in `±1..±64`; any
map from such columns to `k` words that is injective on them has `k ≥ 7` (`k ≥ 8` for the `2^256` int8 segments). -/
theorem words_needed (k : ℕ) (f : (Fin 32 → ℤ) → Fin k → Word)
    (hf : ∀ u u', (∀ p, 1 ≤ |u p| ∧ |u p| ≤ 64) → (∀ p, 1 ≤ |u' p| ∧ |u' p| ≤ 64) → f u = f u' → u = u') :
    7 ≤ k := by
  sorry

/-- Theorem B, zero instance. -/
theorem WordLocalZero : Pouw.Guarantees.Eps1.WordLocalZero := by
  sorry

/-- Theorem B, general instance. -/
theorem WordLocalGeneral : Pouw.Guarantees.Eps1.WordLocalGeneral := by
  sorry

/-- Theorem C. -/
theorem ClassPPacked : Pouw.Guarantees.Eps1.ClassPPacked := by
  sorry

end Pouw.SecurityProofs.Eps1
