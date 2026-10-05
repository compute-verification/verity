import FlockSoundness.Protocol.LigeritoMCA

/-!
# Ligerito's levels meet the Johnson-range MCA bound (PROTOCOL.md §13, §15)

Each level's folds happen over `K`, in the level's Reed–Solomon code embedded from `F`
(`codeK`: `2^logCols` coefficients on the `2^logLen` points `ω_q`). For every level of a
`fast100` schedule (`Level.Ok`), `prCoin_mca_le` applies at the level's radius, and its
bound is the accounting's `Level.mcaA / qK` (`prCoin_level_mca_le`):
* `1 < 2^logCols`, since `logCols ≥ 1`;
* the radius `1 − √ρ − η` is positive, since `ρ ≤ 1/2`;
* the radius is in the Johnson range `< 1 − √((k − 1)/n)`, since `(k − 1)/n < ρ`;
* the bound's numerator, at `n = 2^logLen`, `k = 2^logCols` and `δ` the radius, is `mcaA`.
-/

namespace FlockSoundness

open Game Model Accounting
open CodingTheory CoreDefinitions
open scoped ENNReal NNReal

variable {F K : Type} [Field F] [Fintype F] [DecidableEq F] [Field K] [Fintype K] [DecidableEq K]

/-- A level's evaluation points `ω_q`, in `K`. -/
def domainK (A : Arith F K) (hA : A.Correct) (d : ℕ) (hd : d ≤ 64) : Fin (2 ^ d) ↪ K :=
  (domainEmb A hA d hd).trans ⟨A.embed, A.embed.injective⟩

/-- A level's Reed–Solomon code over `K`: the code its folds live in. -/
noncomputable def codeK (A : Arith F K) (hA : A.Correct) (l : Level) (hf : l.Fits) :
    Submodule K (Fin (2 ^ l.logLen) → K) :=
  ReedSolomon.code (domainK A hA l.logLen hf.2) (2 ^ l.logCols)

end FlockSoundness
