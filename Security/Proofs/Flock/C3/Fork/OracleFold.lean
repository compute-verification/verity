import Proofs.Flock.C3.Fork.Sparse

/-!
# Piece (a), the oracle count: the hidden part's fold is the sparse phase's

A rep's lincheck runs on the public statement `cls.pubStatement regs` and takes the hidden part's fold `h` from the
prover (`holoLincheck`, L4′). At a registered table `R`, the hidden statement `R.toStatement τ regs` has the public
part plus, on each unit slot `u` of slot `σ`, type `τ σ`'s template. Its lincheck comb, folded by the column
challenges, is the public comb's fold plus `sparseFold` at `R`'s cells (`holoFold_eq`): the comb is linear in the
matrices (characteristic 2 casts `ZMod 2` sums to sums), a template's entries at row `i` and column `c` of unit `u` are
weighted `eq(ρ_in, i)·eq(ts, c)`, which `Φ` restates on the entries' planes, and the unit's part of the weights is the
slot's layout factor `U_σ` (`holoPub`), unit by unit, since every unit is inside the block (`hunits`).

So where the prover sends `h = hiddenFold`, `holoLincheck` checks what `lincheck` checks on the hidden statement.
-/

namespace FlockC3.Fork

open FlockSoundness FlockSoundness.Discharge.Recursion
open Model

variable {F K : Type} [Field F] [Fintype F] [DecidableEq F] [Field K] [Fintype K] [DecidableEq K]

/-- **The hidden part's fold** at `R`'s cells, at a rep's zerocheck output and its lincheck's `α` and column challenges
`ts`: the `h` an honest prover sends, and the one the sparse phase checks `h` against (`holoPub` reads no more of the
lincheck's output). -/
noncomputable def hiddenFold (A : Arith F K) (cls : HoloClass) (regs : List (Region cls.kLog (cls.m - cls.kLog)))
    (τ : Fin (2 ^ cls.sLog) → Fin (2 ^ cls.tLog)) (R : RegTable cls) (zc : ZcOut F cls.m) (α : F)
    (ts : Fin (cls.kLog - 6) → F) : Fin 64 → F :=
  sparseFold A cls.shape (holoPub (cls.pubStatement regs) cls.shape cls.units zc ⟨⟨0, 0, 0⟩, α, ts, 0⟩)
    (fun σ => (τ σ).val) R.bits

/-- **The hidden statement's folded comb is the public one's plus the hidden fold.** -/
theorem holoFold_eq (A : Arith F K) (hA : A.Correct) (cls : HoloClass)
    (regs : List (Region cls.kLog (cls.m - cls.kLog))) (τ : Fin (2 ^ cls.sLog) → Fin (2 ^ cls.tLog))
    (R : RegTable cls) (hunits : ∀ σ, ∀ u ∈ cls.units σ, u < 2 ^ (cls.kLog - cls.slotLog))
    (zc : ZcOut F cls.m) (α β : F) (ts : Fin (cls.kLog - 6) → F) (s : Fin 64) :
    foldAll (List.ofFn ts) (2 ^ cls.kLog) (comb A (R.toStatement τ regs) zc.z α β
        fun j : Fin (cls.kLog - 6) => zc.ρ ⟨j.val, by have := cls.hk; omega⟩) s.val =
      foldAll (List.ofFn ts) (2 ^ cls.kLog) (comb A (cls.pubStatement regs) zc.z α β
        fun j : Fin (cls.kLog - 6) => zc.ρ ⟨j.val, by have := cls.hk; omega⟩) s.val +
      hiddenFold A cls regs τ R zc α ts s := by
  sorry

end FlockC3.Fork
