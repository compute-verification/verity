import Properties.Flock.Level3.Seam

/-!
The novel basis (§13.1): `s_0 = X`, `s_{j+1} = s_j² + s_j(β_j)·s_j` over `β_j = x^j`, and `Ŵ_i = s_i · s_i(β_i)⁻¹` (the
soundness model's `What`). That the `X̂_c` are polynomials of degree exactly `c` (`Arith.Correct.xhat_poly`) is
`Proofs.Flock.Level3.NovelBasis`.
-/

open Polynomial Flock

set_option maxRecDepth 20000

namespace FlockLevel3

noncomputable def beta (j : ℕ) : GF128 := GF128.of (F128.monomial j)

noncomputable def sPoly : ℕ → GF128[X]
  | 0 => X
  | j + 1 => sPoly j ^ 2 + C ((sPoly j).eval (beta j)) * sPoly j

/-- `Ŵ_i(ω) = s_i(ω) · s_i(β_i)⁻¹` (the soundness model's `What`). -/
noncomputable def What (i : ℕ) (ω : GF128) : GF128 := (sPoly i).eval ω * ((sPoly i).eval (beta i))⁻¹

end FlockLevel3
