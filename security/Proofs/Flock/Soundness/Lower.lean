module

public import Flock.Lower
public import Proofs.Flock.Soundness.Assumptions.Lower

/-!
# A lowered template unrolls

`Flock.Lower.lower` refuses what fails the reader's checks (`Flock.Lower.lower_checked`), and those are the checks
`Flock.TmplUnroll.unroll_ok` takes, so every template it writes has a committed net.
-/

@[expose] public section

namespace FlockSoundness.Lower

open Flock

/-- **A template `lower` writes unrolls**: `Tmpl.unroll` builds its net. -/
theorem lower_unrolls {P : Program.Prog} {d : Nat} {inP outP : Array Nat} {t : Tmpl.Tmpl}
    (hl : Lower.lower P d inP outP = .ok t) : ∃ n, Tmpl.unroll t = .ok n :=
  let ⟨hw, ha, ho, hi, hg⟩ := Lower.lower_checked hl
  TmplUnroll.unroll_ok hw ha ho hi hg

end FlockSoundness.Lower
