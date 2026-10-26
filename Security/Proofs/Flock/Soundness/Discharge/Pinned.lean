import Definitions.Flock.Statement

/-!
# Pinned outputs (META `pinned_outputs`): the rows a pinned bit has, and what they force

A pinned output port has no row. The C-Flock prover's Δ copies a unit slot's forced-zero row onto each of its bits,
`(i, i)` and `(i, zero)` in `A` and in `B`, at a bit `i` whose net row is linear (`B` the constant alone, which its
`check` requires of a pinned bit). So the block rows at `i` are `A = a + eᵢ + e_zero` and `B = e_pin + eᵢ + e_zero`
(`PinnedRow`), and a satisfying witness has `zᵢ = (a·z + zᵢ)(1 + zᵢ)`: `zᵢ = 0` and `a·z = 0` (`PinnedRow.block_eq`).
The bit's own linear form, the value its unit computes, is zero.

A writer unit reads a shared row `S`, selects its window `j` of `w` words, and pins `d = S[j] XOR y`: with `d` zero, `y`
is that window of `S` (`window_eq`). Neither reads a row size, a `G`, a window width or a producer family.

The Lean verifier refuses the form until its reader of `pinned_outputs` lands (`HmRow.parse` requires `out_row_ports`
to cover every output port).
-/

namespace FlockSoundness.Discharge.Pinned

variable {S : Model.Statement}

/-- **A pinned bit's block rows**: its net's linear form `a` in `A` and the constant in `B`, each with Δ's copy of the
forced-zero row `zero` (`(r, r)` and `(r, zero)`) added. -/
def PinnedRow (S : Model.Statement) (r zero : Fin (2 ^ S.kLog)) (a : Fin (2 ^ S.kLog) → ZMod 2) : Prop :=
  (∀ j, S.A₀ r j = a j + (if j = r then 1 else 0) + (if j = zero then 1 else 0)) ∧
    ∀ j, S.B₀ r j = (if j = S.pin then 1 else 0) + (if j = r then 1 else 0) + (if j = zero then 1 else 0)

theorem mulVec_two (M : Matrix (Fin (2 ^ S.kLog)) (Fin (2 ^ S.kLog)) (ZMod 2)) {r p q : Fin (2 ^ S.kLog)}
    {a : Fin (2 ^ S.kLog) → ZMod 2} (h : ∀ j, M r j = a j + (if j = p then 1 else 0) + (if j = q then 1 else 0))
    (v : Fin (2 ^ S.kLog) → ZMod 2) : M.mulVec v r = dotProduct a v + v p + v q := by
  simp only [Matrix.mulVec, dotProduct, h, add_mul, ite_mul, one_mul, zero_mul, Finset.sum_add_distrib,
    Finset.sum_ite_eq', Finset.mem_univ, ite_true]

/-- Over GF(2), `x = (α + x)(1 + x)` holds only at `x = 0` and `α = 0`. -/
theorem pinned_rule {x α : ZMod 2} (h : (α + x) * (1 + x) = x) : x = 0 ∧ α = 0 := by
  have key : ∀ x α : ZMod 2, (α + x) * (1 + x) = x → x = 0 ∧ α = 0 := by decide
  exact key x α h

/-- **A pinned bit carries 0, and so does its linear form**, in every block of a satisfying witness whose forced-zero
row carries 0. -/
theorem PinnedRow.block_eq {r zero : Fin (2 ^ S.kLog)} {a : Fin (2 ^ S.kLog) → ZMod 2} (h : PinnedRow S r zero a)
    {z : Model.Witness S.m} (hz : S.Satisfies z) (o : Fin (2 ^ S.nbl)) (hzero : S.block z o zero = 0) :
    S.block z o r = 0 ∧ dotProduct a (S.block z o) = 0 := by
  have hr := hz.1 o r
  have hp : dotProduct (fun j => if j = S.pin then (1 : ZMod 2) else 0) (S.block z o) = 1 := by
    simpa [dotProduct, ite_mul] using hz.2.1 o
  rw [mulVec_two S.A₀ h.1, mulVec_two (a := fun j => if j = S.pin then (1 : ZMod 2) else 0) S.B₀ h.2, hp, hzero]
    at hr
  simp only [add_zero] at hr
  exact pinned_rule hr

/-- **A pinned difference makes the output its window**: if `sel` reads window `j` of a shared row `S` (`w` words of 16
bits), `d = sel + y` bit for bit and `d` is pinned to zero, then bit `t` of `y` is bit `16 w j + t` of `S`. -/
theorem window_eq {w j : ℕ} {row sel d y : ℕ → ZMod 2} (hsel : ∀ t < 16 * w, sel t = row (16 * w * j + t))
    (hxor : ∀ t < 16 * w, d t = sel t + y t) (hpin : ∀ t < 16 * w, d t = 0) :
    ∀ t < 16 * w, y t = row (16 * w * j + t) := by
  intro t ht
  have h := hxor t ht
  rw [hpin t ht, hsel t ht] at h
  have key : ∀ s x : ZMod 2, 0 = s + x → x = s := by decide
  exact key _ _ h

end FlockSoundness.Discharge.Pinned
