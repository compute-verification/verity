import Proofs.Flock.Recursive.HoloExec

/-!
# Piece (a), the sparse count: one rep's sparse phase is sound

`Model.sparsePhase` (H1–H4) checks a rep's hidden fold `h` against a registered table `z` at the rep's public values
`pub` and the slots' types `τ`, and hands on a registered claim on `z`. `sparse_sound`: where `h` is not `z`'s fold
(`sparseFold`), the phase hands on a claim that holds on `z` with probability at most `sparseError`, each coin its degree
over `qF` (Schwartz–Zippel): H1's 6, H2's 6 rounds of degree 2, H3's `ν` rounds of degree `2ℓ + 1` and `sLog` of degree
`2ℓ + 2`, H4's `pLog` and the type point's `tLog`.

`sparseFold` is the phase's own algebra: row `s` of the skip, `Σ_r lagS(z_skip, r) · Σ_σ U_σ · Σ_k Φ(x(r), y(s), q(τ_σ, k))`
with `q(t, k)` the planes of type `t` at entry `k` (`planeAt`, the bit `k + 2^ν·(p + 2^pLog·t)` of `z`). That it is the
hidden part of the lincheck's fold at `z = R.bits`, for a table `R` whose unit slots are in range, is the assembly's
(`holo_runs_sound`, `C3/Fork/Runs.lean`), not this lemma's.
-/

namespace FlockC3.Fork

open FlockSoundness FlockSoundness.Discharge.Recursion
open Game Model Accounting
open scoped ENNReal

variable {F K : Type} [Field F] [Fintype F] [DecidableEq F] [Field K] [Fintype K] [DecidableEq K]

/-- The skip's row `r` as a point of `{0, 1}^6`. -/
def skipBits (r : Fin 64) : Fin 6 → F := fun j => if r.val.testBit j.val then 1 else 0

/-- Plane `p` of type `t` at entry `k` of the registered table `z`: bit `k + 2^ν·(p + 2^pLog·t)`, zero past the table. -/
def planeAt (sh : SparseShape) (z : Witness sh.mReg) (t p k : ℕ) : F :=
  if h : k + 2 ^ sh.nu * (p + 2 ^ sh.pLog * t) < 2 ^ sh.mReg then ((z ⟨_, h⟩).val : F) else 0

/-- **The hidden fold the sparse phase checks** at the registered table `z`: row `s` of the skip. -/
noncomputable def sparseFold (A : Arith F K) (sh : SparseShape) (pub : SparsePub F sh) (τ : Fin (2 ^ sh.sLog) → ℕ)
    (z : Witness sh.mReg) (s : Fin 64) : F :=
  ∑ r : Fin 64, A.lagS pub.z r * ∑ σ : Fin (2 ^ sh.sLog), pub.U σ * ∑ k : Fin (2 ^ sh.nu),
    sparsePhi sh.ell pub.alpha (cat6 (skipBits r) pub.xRest) (cat6 (skipBits s) pub.yRest)
      (fun p => planeAt sh z (τ σ) p k.val)

/-- **The sparse phase is sound**: at slots' types in range, a hidden fold `h` other than the registered table's
(`sparseFold`) yields a registered claim that holds on the table with probability at most `sparseError`. -/
theorem sparse_sound (A : Arith F K) (hA : A.Correct) (sh : SparseShape) (pub : SparsePub F sh)
    (τ : Fin (2 ^ sh.sLog) → ℕ) (hτ : ∀ σ, τ σ < 2 ^ sh.tLog) (z : Witness sh.mReg) (h : Fin 64 → F)
    (hne : h ≠ sparseFold A sh pub τ z) :
    value (fun o : Option (RegClaim F sh) => ∃ c, o = some c ∧ c.toClaim.Holds z) (sparsePhase A sh pub τ h) ≤
      ENNReal.ofReal (sparseError sh) := by
  sorry

end FlockC3.Fork
