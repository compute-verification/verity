import Mathlib
import Definitions.Pouw.RelativeSecurity

/-!
# Neekon's security definition with the two-stage game, and Dither's ideal protocol (trusted)

Neekon Vafa's definition of a secure proof of useful work, with the deviations agreed for it
(note:20261017T0013Z-report-epsilon-program): the two-stage game, an edited useful output, and a long proof in the ideal
protocol. It reuses relative security's machine model (`Machine`: algorithms, what they compute, and an abstract
`time`, worst case over inputs and coins, at least the program's size) and its uniform probability (`prob`).

* `PoUW`: instances, the verifier's live coins σ, proofs, useful outputs, `Verify` and `Solve`.
* `Adversary`: `M = (M₁, M₂)`. `M₁` fixes the instance before σ is drawn; `M₂` maps σ to a proof, and `time(M)` is
  `M₂`'s time.
* `NeekonSecure ε δ η`: every adversary accepted with probability at least `δ` gives an `M'` that, for every
  instance, outputs the useful product with an accepted proof except with probability `η`, with
  `time(M') ≤ (1 + ε) time(M)`.
* `dither`: Dither's ideal protocol. σ draws `R ∈ {0,1}^{m×k}` and `U ∈ {±1, …, ±64}^{n×32}`, `S = ΠUᵀ`
  (`S_{tj} = U_{j, t mod 32}`); the proof is the transcript `T`, every 32-step prefix sum of `(A + R)(B + S)` over
  `Z/2³²`; `Verify` recomputes `T`; the useful output is `(A + R)B`.

Main's `Secure` (`Proofs.Pouw.RelativeSecurity`, Neekon's `∃ M'` form as RS-DEF froze it) is the one-stage form of the
same shape: its adversary is one algorithm that commits first, timed whole, its `M'` may use coins and an additive
slack `α`, and its acceptance doesn't tie the proof to the instance. Here `M₁` is free, so only `M₂` is timed, and
`Verify` reads the instance `M₁` fixed. `RelativelySecure` adds completeness and efficiency to `Secure`'s
infimum form; `Complete` here is its completeness, and efficiency is `Solve`'s time, which Theorem B's `ε` reads.
-/

namespace Pouw.Neekon

open Pouw.RelativeSecurity Finset

section Definition

/-- **A proof of useful work** in a machine `Mc`, in Neekon's form: instances `Inst` (what `M₁` fixes, for matmul
`(A, B)`), the verifier's live coins σ (uniform on the finite `Seed`), proofs, useful outputs, the verifier, the
problem and the honest prover. -/
structure PoUW (Mc : Machine) where
  /-- The instances. -/
  Inst : Type
  /-- The verifier's live coins σ, drawn after the instance is fixed. -/
  Seed : Type
  [instSeed : Fintype Seed]
  /-- A proof π. -/
  Proof : Type
  /-- The useful output. -/
  Out : Type
  /-- `Verify(σ, x, π)`, deterministic: π proves instance `x` against σ. In the ideal protocol π carries `x` in the
  clear, and the game reads it as the instance `M₁` fixed. -/
  verify : Seed → Inst → Proof → Bool
  /-- `C` is instance `x`'s useful product against σ (for Dither, the edited product `(A + R)B`). -/
  useful : Seed → Inst → Out → Prop
  /-- `Solve(σ, x)`: a proof and the useful product. -/
  solve : Mc.Alg (Seed × Inst) (Proof × Out)

attribute [instance] PoUW.instSeed

variable {Mc : Machine}

/-- **A two-stage adversary** `M = (M₁, M₂)` with coins on `Ω`: `M₁` runs before σ is drawn and fixes the instance
`inst` and a state; `M₂` maps σ, its coins and the state to a proof. `M₁` is any function and is not timed, so it is
its outputs: the instance, and `M₂` with the state built into its program. Its state is then charged as program size
(`Machine.size_le_time`, T4). As an input it would be free, and `M₁` could tabulate every σ's proof. A randomized
`M₁` is a mixture of these, and one of its outputs is accepted at least as often as its average. -/
structure Adversary (P : PoUW Mc) (Ω : Type) where
  /-- The instance `M₁` fixes. -/
  inst : P.Inst
  /-- `M₂(·, state)`: σ and its coins to a proof. -/
  M₂ : Mc.Alg (P.Seed × Ω) P.Proof

namespace Adversary

variable {P : PoUW Mc} {Ω : Type}

/-- `time(M)`: the adversary's time is `M₂`'s. -/
def time (M : Adversary P Ω) : ℝ := Mc.time M.M₂

/-- `Pr_{σ, ω}[Verify(σ, x, M₂(σ; ω)) = 1]` for the instance `x` that `M₁` fixed. -/
noncomputable def acceptance [Fintype Ω] (M : Adversary P Ω) : ℝ :=
  prob fun ω : P.Seed × Ω => P.verify ω.1 M.inst (Mc.eval M.M₂ ω) = true

end Adversary

/-- `M'` solves every instance: for every instance `x`, except with probability at most `η` over σ, it outputs `x`'s
useful product with a proof of `x` that verifies. -/
def SolvesAll (P : PoUW Mc) (M' : Mc.Alg (P.Seed × P.Inst) (P.Proof × P.Out)) (η : ℝ) : Prop :=
  ∀ x : P.Inst, 1 - η ≤ prob fun σ : P.Seed =>
    P.useful σ x (Mc.eval M' (σ, x)).2 ∧ P.verify σ x (Mc.eval M' (σ, x)).1 = true

/-- **Completeness**: `Solve` solves every instance except with probability `η`. -/
def Complete (P : PoUW Mc) (η : ℝ) : Prop :=
  SolvesAll P P.solve η

/-- **Neekon's security with the two-stage game**: every two-stage adversary `M` accepted with probability at least
`δ` gives an `M'` that solves every instance except with probability `η`, with `time(M') ≤ (1 + ε) time(M)`. -/
def NeekonSecure (P : PoUW Mc) (ε δ η : ℝ) : Prop :=
  ∀ (Ω : Type) [Fintype Ω] (M : Adversary P Ω), δ ≤ M.acceptance →
    ∃ M' : Mc.Alg (P.Seed × P.Inst) (P.Proof × P.Out), Mc.time M' ≤ (1 + ε) * M.time ∧ SolvesAll P M' η

end Definition

section Dither

/-- `Z/2³²`: int8 products accumulated exactly in s32. -/
abbrev Z32 : Type := ZMod (2 ^ 32)

/-- Dither's instances `(A, B)`: `A` is `m × k` and `B` is `k × n` over `Z/2³²`, the contraction index `k = 32K`
read as (segment `ℓ < K`, offset `o < 32`). -/
abbrev DInst (m K n : ℕ) : Type :=
  Matrix (Fin m) (Fin K × Fin 32) Z32 × Matrix (Fin K × Fin 32) (Fin n) Z32

/-- Dither's draw σ, as the re-encoded design's bits (`DBits n 32 7 K m`, uniform on its subsets): the bit
`b(ℓ, o, i)` is `R`'s entry `(i, (ℓ, o))`, and `β_p(j, o)` is plane `p` of `U`'s entry `(j, o)`. -/
abbrev DSeed (m K n : ℕ) : Type := Finset (DBits n 32 7 K m)

/-- A transcript: the prefix word `T(i, j, ℓ)` of each output cell `(i, j)` through each segment `ℓ`. -/
abbrev DTrans (m K n : ℕ) : Type := Fin m → Fin n → Fin K → Z32

variable {m K n : ℕ}

/-- `R ∈ {0,1}^{m×k}`, fresh and uniform. -/
def noiseR (σ : DSeed m K n) : Matrix (Fin m) (Fin K × Fin 32) Z32 :=
  Matrix.of fun i t => bitInd σ (.inr (t.1, t.2, i))

/-- `U ∈ {±1, …, ±64}^{n×32}`, fresh and uniform: `U = −64 + Σ_p w_p β_p` at the weights `wRe = (1, 2, 4, 8, 16, 32,
65)`, which map 7 uniform bits onto `{±1, …, ±64}` one to one. -/
def noiseU (σ : DSeed m K n) : Matrix (Fin n) (Fin 32) Z32 :=
  Matrix.of fun j o => -64 + ∑ p, wRe p * bitInd σ (.inl (j, o, p))

/-- `S = ΠUᵀ`, rank 32: `S_{tj} = U_{j, t mod 32}`, so `U`'s column `o` serves every segment. -/
def noiseS (σ : DSeed m K n) : Matrix (Fin K × Fin 32) (Fin n) Z32 :=
  Matrix.of fun t j => noiseU σ j t.2

/-- `Ã = A + R`. -/
def editA (σ : DSeed m K n) (x : DInst m K n) : Matrix (Fin m) (Fin K × Fin 32) Z32 := x.1 + noiseR σ

/-- `B̃ = B + S`. -/
def editB (σ : DSeed m K n) (x : DInst m K n) : Matrix (Fin K × Fin 32) (Fin n) Z32 := x.2 + noiseS σ

/-- **Dither's transcript**: every 32-step prefix sum of `ÃB̃` over `Z/2³²`, `T(i, j, ℓ) = Σ_{ℓ' ≤ ℓ} Σ_o
Ã(i, (ℓ', o)) B̃((ℓ', o), j)`. -/
def transcript (σ : DSeed m K n) (x : DInst m K n) : DTrans m K n :=
  fun i j ℓ => ∑ ℓ' : Fin K, if ℓ' ≤ ℓ then ∑ o : Fin 32, editA σ x i (ℓ', o) * editB σ x (ℓ', o) j else 0

/-- **Dither's ideal protocol**, served by the honest prover `solve`: the proof is the transcript (the ideal proof is
`(A, B, T)`, and `(A, B)` is the instance `M₁` fixed), `Verify` recomputes it, and the useful output is the edited
product `(A + R)B = ÃB̃ − (ÃΠ)Uᵀ`. -/
def dither (Mc : Machine) (m K n : ℕ)
    (solve : Mc.Alg (DSeed m K n × DInst m K n) (DTrans m K n × Matrix (Fin m) (Fin n) Z32)) : PoUW Mc where
  Inst := DInst m K n
  Seed := DSeed m K n
  Proof := DTrans m K n
  Out := Matrix (Fin m) (Fin n) Z32
  verify σ x T := decide (T = transcript σ x)
  useful σ x C := C = editA σ x * x.2
  solve := solve

end Dither

end Pouw.Neekon
