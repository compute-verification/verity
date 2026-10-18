import Definitions.Accounting.Gates.Model

/-!
# Red team of G: the property's caps are whole units per certified byte

Scratch, not built by any `lean_lib`. `Gates.Caps`'s fields are `ℕ` units per certified byte, so a circuit with one
rounding unit in a window needs `caps.rounding ≥ 1`: at least `1/ρ_read ≈ 196` times the honest peak's
`5.1·10^-3` roundings a certified byte. G's `C` grows with `caps.total`, so the cap the property can state at all
sets `C` at about 200 times what the honest rate would.
-/

namespace GRedTeam

open Accounting.Gates

theorem rounding_cap_pos {ι : Type} (k : Caps) (A : Audit) (W : Dag ι) (hP : Property k A W) (w : ℕ)
    (hw : w < A.windows) (hpos : 0 < (census W fun t => A.win t = w).rounding) : 1 ≤ k.rounding := by
  have h := (hP.2.1 w hw).1
  rcases Nat.eq_zero_or_pos k.rounding with h0 | h0
  · rw [h0, Nat.zero_mul] at h; omega
  · exact h0

theorem caps_total_pos {ι : Type} (k : Caps) (A : Audit) (W : Dag ι) (hP : Property k A W) (w : ℕ)
    (hw : w < A.windows) (hpos : 0 < (census W fun t => A.win t = w).rounding) : 1 ≤ k.total := by
  have := rounding_cap_pos k A W hP w hw hpos
  unfold Caps.total; omega

end GRedTeam

#print axioms GRedTeam.rounding_cap_pos
#print axioms GRedTeam.caps_total_pos
