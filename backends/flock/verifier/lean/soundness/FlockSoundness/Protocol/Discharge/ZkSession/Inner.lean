import ArkLib.Data.CodingTheory.ProximityGap.BCIKS20.AffineLines.UniqueDecoding
import VCVio.OracleComp.Constructions.SampleableType.NativeMeasure
import FlockSoundness.Protocol.ZerocheckPhase
import Mathlib.Analysis.MeanInequalities

/-!
# The `--zk` inner proof is sound (live/PROTOCOL.md §5)

Under `--zk` the prover commits to its pads `h` before the first masked value: the pads commitment is the Reed–Solomon
code over `F` of `(h ‖ pad_h)` and of `(mu ‖ pad_mu)`, `L = s + nP` coefficients on `N` points (§3), and its root is
absorbed before `r_skip`. Every check the verifier would make on the masked transcript becomes an affine form in `h`
(§4's `Lin`), and the inner proof shows they all vanish:

1. `ε` ← V; P sends `ρ, σ` (the sacrifice's two openings);
2. `γ` ← V (one coin per row); both sides form the batch `(c, t)`; P sends `τ = ⟨c, mu⟩` (committed under M2; binding,
   which `cr/sha-512` gives as for every other leaf, makes the commitment a message sent here);
3. `β` ← V; P sends `y = (h ‖ pad_h) + β·(mu ‖ pad_mu)` (§5's `Y`);
4. `q` query words ← V, each read as a position `pos w`; V accepts iff `⟨c, y[..s]⟩ = t + βτ` and
   `C(y)_j = C_h[j] + β·C_mu[j]` at every queried `j`.

`innerGame` is that interaction against fixed committed words `Ch, Cμ` and fixed rows (the transcript before the inner
proof fixes them). `inner_sound`: unless some message `m` within relative distance `δ` of `Ch` has every row vanish at
its pads and `h_ab = h_fa·h_fb` (`InnerGood`), the verifier accepts with probability at most
`(N + 3)/|F| + (1 − δ)^q`, for `δ` up to the unique-decoding radius:

* if `Ch, Cμ` are jointly `δ`-close to the code, the decoded pads fail some row; the sacrifice (`ε`), the batch (`γ`) and
  `β` each miss with probability at most `1/|F|`, and then any `y` but the decoded one disagrees with
  `Ch + β·Cμ` on more than a `δ` fraction of positions;
* if not, BCIKS20's correlated agreement on lines in the unique-decoding regime
  (ArkLib's `RS_correlatedAgreement_affineLines_uniqueDecodingRegime`) makes `Ch + β·Cμ` `δ`-far from the code except
  with probability `N/|F|`, and then every `y` fails a query with probability at least `δ`.

The query positions are any map `pos` with `Pr[pos w ∈ D] ≤ |D|/N` (`hpos`); the protocol's `lo(w) mod N` with
`N` a power of two is one (`Arith.Correct.lo_balanced`).
-/

namespace FlockSoundness.Discharge.ZkSession

open Game
open scoped ENNReal NNReal

variable {F : Type} [Field F]

/-! ## The rows and the batch -/

/-- An affine form `Σ_i coef_i·h_i + const` in the pads (§4's `Lin`). -/
structure Lin (F : Type) (s : ℕ) where
  coef : Fin s → F
  const : F

/-- The pad indices the sacrifice reads: the product `h_ab = h_fa·h_fb` it binds and the sacrificed triple
`x, y, w = x·y` (§2's `fa, fb, hab, tx, ty, tw`). -/
structure Triple (s : ℕ) where
  fa : Fin s
  fb : Fin s
  ab : Fin s
  tx : Fin s
  ty : Fin s
  tw : Fin s

/-- §4's three triple rows at the coin `ε` and the prover's `ρ, σ`:
`ε h_fa + h_tx + ρ`, `h_fb + h_ty + σ`, `ε h_ab + h_tw + σ h_tx + ρ h_ty + ρσ`. -/
def tripleRows {s : ℕ} (tr : Triple s) (ε ρ σ : F) : Fin 3 → Lin F s :=
  ![⟨Pi.single tr.fa ε + Pi.single tr.tx 1, ρ⟩,
    ⟨Pi.single tr.fb 1 + Pi.single tr.ty 1, σ⟩,
    ⟨Pi.single tr.ab ε + Pi.single tr.tw 1 + Pi.single tr.tx σ + Pi.single tr.ty ρ, ρ * σ⟩]

/-- The batch's coefficient vector `c = Σ_k γ_k·coef_k`. -/
def batchCoef {s n : ℕ} (γ : Fin n → F) (R : Fin n → Lin F s) : Fin s → F :=
  fun i => ∑ k, γ k * (R k).coef i

/-- The batch's target `t = −Σ_k γ_k·const_k`, so that `⟨c, h⟩ = t` iff `Σ_k γ_k·R_k(h) = 0`. -/
def batchT {s n : ℕ} (γ : Fin n → F) (R : Fin n → Lin F s) : F := -∑ k, γ k * (R k).const

/-! ## The pads code -/

/-- The Reed–Solomon encoding of a message read as the coefficients of `Σ_i m_i X^i`, at the points `dom` (§3). -/
def enc {N L : ℕ} (dom : Fin N ↪ F) (m : Fin L → F) : Fin N → F :=
  fun j => ∑ i, m i * dom j ^ (i : ℕ)

/-- The first `s` coefficients of a pads message: the pads `h` (or `mu`), without the code's padding. -/
def head {s nP : ℕ} (m : Fin (s + nP) → F) : Fin s → F := fun i => m (Fin.castAdd nP i)

/-! ## The inner proof -/

/-- The verifier's final check (§5, step 5): the batched constraint on `y`'s pads, and `C(y)_j = C_h[j] + β·C_mu[j]` at
every queried position. -/
def innerCheck {N : ℕ} (dom : Fin N ↪ F) {s nP q : ℕ} (pos : F → Fin N) {nA : ℕ} (rows : Fin nA → Lin F s)
    (tr : Triple s) (Ch Cμ : Fin N → F) (ε : F) (ρσ : F × F) (γ : Fin (nA + 3) → F) (τ β : F)
    (y : Fin (s + nP) → F) (w : Fin q → F) : Prop :=
  let R := Fin.append rows (tripleRows tr ε ρσ.1 ρσ.2)
  (∑ i, batchCoef γ R i * head y i = batchT γ R + β * τ) ∧
    ∀ j, enc dom y (pos (w j)) = Ch (pos (w j)) + β * Cμ (pos (w j))

instance [DecidableEq F] {N : ℕ} (dom : Fin N ↪ F) {s nP q : ℕ} (pos : F → Fin N) {nA : ℕ}
    (rows : Fin nA → Lin F s) (tr : Triple s) (Ch Cμ : Fin N → F) (ε : F) (ρσ : F × F) (γ : Fin (nA + 3) → F)
    (τ β : F) (y : Fin (s + nP) → F) (w : Fin q → F) :
    Decidable (innerCheck dom pos rows tr Ch Cμ ε ρσ γ τ β y w) := by
  unfold innerCheck
  infer_instance

/-- The inner proof's soundness error: BCIKS20's `N/|F|` (or the sacrifice, the batch and `β`, `1/|F|` each) plus
`q` queries at distance `δ`. -/
noncomputable def εInner (N : ℕ) (F : Type) [Fintype F] (q : ℕ) (δ : ℝ≥0) : ℝ≥0∞ :=
  (N + 3 : ℝ≥0∞) / Fintype.card F + ((1 - δ : ℝ≥0) : ℝ≥0∞) ^ q

/-! ## Rows -/

/-! ## The code -/

section Code

open Polynomial

variable {N L : ℕ} (dom : Fin N ↪ F)

end Code

/-! ## Distance, queries and coins -/

section Sound

variable [Fintype F] [DecidableEq F]

open Code ProximityGap

/-! ## Soundness -/

end Sound

end FlockSoundness.Discharge.ZkSession
