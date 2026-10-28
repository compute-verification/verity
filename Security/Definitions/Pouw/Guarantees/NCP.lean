import Definitions.Pouw.NCP.Pinned
import Definitions.Pouw.NCP.Protocol
import Definitions.Pouw.NCP.Relation
import Definitions.Pouw.NCP.RouteU
import Definitions.Pouw.Assumptions.NCP.RouteUGame
import Definitions.Pouw.Assumptions.NCP.RouteUWord

/-!
# NCP's guarantees (trusted)

Each guarantee is a `def G : Prop`, stated over `Definitions.Pouw`'s definitions with the named assumptions it rests on as
hypotheses (`Pouw.Assumptions`). Its proof is the theorem `Pouw.SecurityProofs.NCP.G : Pouw.Guarantees.NCP.G`. The
lead's algebra (`Definitions.Pouw.NCP.Pinned`), the shift by 24 under the rejection rule (`Definitions.Pouw.NCP.Protocol`),
the rejection rule's witness (`Definitions.Pouw.NCP.Relation`), route U's checked values (`Definitions.Pouw.NCP.RouteU`), its
game instance (`Pouw.Assumptions.NCP.RouteUGame`) and its word layer (`Pouw.Assumptions.NCP.RouteUWord`). Each keeps
the text, the docstring and the `open`s of the statement it was in its source module. `ShiftBy24Admissible` is the type
of the theorem `Pouw.NCP.GameProofs.shiftBy24_admissible`, its binders written as a `∀`, in that theorem's `open`s but
`Pouw.Proofs`, which only proof modules declare.

`Pouw.NCP.RouteU.GammaFromTTNCP_U`, which `GammaFromTTNCP_U_v1` and `GammaFromTTNCP_U_C2` instantiate, keeps its name
and namespace and comes just before them, as in its source: `GammaFromTTNCP_U_C2` reuses the auxiliary proof Lean makes
for its `3 : ℝ` (`Pouw.NCP.RouteU.GammaFromTTNCP_U._proof_1`), and the audit reads that proof by name.
-/

namespace Pouw.Guarantees.NCP

/-! ## The lead's algebra (`Definitions.Pouw.NCP.Pinned`) -/

section

open Pouw.NCP Pouw.NCP.Pinned

/-- **Cancellation**: for all integer matrices and every `R`,
`(A + E₁)(B + F₁) + (A + E₂)(B + F₂) − (A + E₃)(B + F₃) = A·B`. -/
def Cancel : Prop :=
  ∀ (k : ℕ) (A B E1 F1 R : Mat),
    mm k (A + E1) (B + F1) + mm k (A + E2 k E1 R) (B + F2 k R F1) - mm k (A + E3 k E1 R) (B + F3 k R F1) = mm k A B

/-- **One product**: the same identity as one product of inner dimension `3k` on the stacked operands. -/
def Stacked : Prop :=
  ∀ (k : ℕ) (A B E1 F1 R : Mat),
    mm (3 * k) (stackX k A E1 R) (stackY k B R F1)
        = mm k (A + E1) (B + F1) + mm k (A + E2 k E1 R) (B + F2 k R F1) - mm k (A + E3 k E1 R) (B + F3 k R F1) ∧
      mm (3 * k) (stackX k A E1 R) (stackY k B R F1) = mm k A B

/-- **The int8 range**: for `R` a signed permutation, `E₁`, `F₁` in `[−31, 31]` and int7 `A`, `B`, every entry of both
stacked operands lies in `[−126, 126]`. -/
def Range : Prop :=
  ∀ (k : ℕ) (A B E1 F1 R : Mat), SignedPerm k R →
    (∀ i, ∀ l < k, -31 ≤ E1 i l ∧ E1 i l ≤ 31) → (∀ l < k, ∀ j, -31 ≤ F1 l j ∧ F1 l j ≤ 31) →
    (∀ i, ∀ l < k, InInt7 (A i l)) → (∀ l < k, ∀ j, InInt7 (B l j)) →
    (∀ i l, l < 3 * k → -126 ≤ stackX k A E1 R i l ∧ stackX k A E1 R i l ≤ 126) ∧
      (∀ l j, l < 3 * k → -126 ≤ stackY k B R F1 l j ∧ stackY k B R F1 l j ≤ 126)

end

/-! ## The shift by 24 under the rejection rule (`Definitions.Pouw.NCP.Protocol`) -/

section

open Pouw.NCP Pouw.NCP.GameProofs Finset QTree

/-- **The shift by `24` passes the rejection rule at `d = 16`**, for every `k ≥ 1024` with `16 ∣ k`: step `t`'s image
`[16t + 24, 16t + 40) mod k` is disjoint from it, columns `16t + 7` and `16t + 8` land in different steps, and `R²` (the
shift by `48`) has `0` at `(0, 0)`. -/
def ShiftBy24Admissible : Prop := ∀ {k : ℕ}, 1024 ≤ k → 16 ∣ k → Admissible k 16 (shiftBy 24 k)

end

/-! ## The rejection rule's witness (`Definitions.Pouw.NCP.Relation`) -/

section

open Pouw.NCP Pouw.NCP.Relation Finset

/-- **An admissible `R` with independent checkpoints exists** (the witness for the rule): for every `k ≥ 1024` with
`16 ∣ k`. -/
def IndependentWitness : Prop := ∀ k, 1024 ≤ k → 16 ∣ k → ∃ R, Admissible k 16 R ∧ CheckpointIndependent k 16 R

end

/-! ## Route U's checked values (`Definitions.Pouw.NCP.RouteU`) -/

section

open Pouw.NCP Pouw.NCP.RouteU Finset

/-- **The biases that keep `X + β` in u8**: for permutation matrices `R` exactly `β ∈ {127, 128, 129}`, and for
signed permutations exactly `β ∈ {127, 128}`. On the int7 domain `X` ranges over `[−127, 126]` for a permutation
matrix and over `[−127, 127]` for a signed permutation (at `k = 2`, `X₃ = −(A + E₁ + E₂)` reaches `127` with
`E₂ = −E₁[π l]`), and both ends are attained. -/
def BiasU8Iff : Prop :=
  ∀ β : ℤ, (BiasU8 β Unsigned ↔ 127 ≤ β ∧ β ≤ 129) ∧ (BiasU8 β SignedPerm ↔ 127 ≤ β ∧ β ≤ 128)

/-- **The shift identity** (over ℤ, every `β`): for `16t ≤ 3k`, route U's checked value is the unbiased running sum
minus `β` times the suffix column sum of `Y` from `16t`: `C^U_t = C_t − β·s_t`. -/
def CheckedUEq : Prop :=
  ∀ (β : ℤ) (k : ℕ) (A B E1 F1 R : Mat) (t i j : ℕ), 16 * t ≤ 3 * k →
    checkedU β k A B E1 F1 R t i j =
      mm (16 * t) (stackX k A E1 R) (stackY k B R F1) i j - β * ∑ l ∈ Ico (16 * t) (3 * k), stackY k B R F1 l j

/-- **The final checked value** (every `β`): at `t = T = 3k/16` (`16 ∣ k`) route U's checked value is `A·B`. -/
def CheckedUFinal : Prop :=
  ∀ (β : ℤ) (k : ℕ) (A B E1 F1 R : Mat) (i j : ℕ), 16 ∣ k → checkedU β k A B E1 F1 R (3 * k / 16) i j = mm k A B i j

/-- **Route U's operand ranges**: for `β ∈ {127, 128, 129}`, a permutation matrix `R`, `E₁`, `F₁` in `[−31, 32]`
and int7 `A`, `B`, every entry of `X_b` is u8 (`[0, 255]`) and every entry of `Y` is s8 (`[−128, 127]`). For a
signed permutation the same holds at `β ∈ {127, 128}` (`BiasU8Iff`; `Y` is s8 for every signed permutation). -/
def RangeU : Prop :=
  ∀ (β : ℤ) (k : ℕ) (A B E1 F1 R : Mat), 127 ≤ β → β ≤ 129 → Unsigned k R →
    (∀ i, ∀ l < k, InNoise6 (E1 i l)) → (∀ l < k, ∀ j, InNoise6 (F1 l j)) →
    (∀ i, ∀ l < k, InInt7 (A i l)) → (∀ l < k, ∀ j, InInt7 (B l j)) →
    (∀ i l, l < 3 * k → 0 ≤ stackXU β k A E1 R i l ∧ stackXU β k A E1 R i l ≤ 255) ∧
      (∀ l j, l < 3 * k → InInt8 (stackY k B R F1 l j))

/-- **int32 exactness of the last checkpoint** (as `WrapInt7`, every `β`): for int7 `A`, `B`, `16 ∣ k` and
`k ≤ 2^19 − 1`, with any noise and `R`, route U's int32 accumulator at the last checkpoint, `16·(3k/16) = 3k`, is
`A·B`, and so is the signed reading of the last checked value mod `2^32`. -/
def WrapU : Prop :=
  ∀ (β : ℤ) (k : ℕ) (A B E1 F1 R : Mat) (i j : ℕ), 16 ∣ k → k ≤ 2 ^ 19 - 1 →
    (∀ l < k, InInt7 (A i l)) → (∀ l < k, InInt7 (B l j)) →
    mmWU β k A B E1 F1 R (16 * (3 * k / 16)) i j = mm k A B i j ∧
      wrap32 (checkedU β k A B E1 F1 R (3 * k / 16) i j) = mm k A B i j

/-- **The wrap is the identity at every depth**: route U's int32 accumulator is its running value over ℤ at every
`τ ≤ 3k`, for `β ∈ {127, 128, 129}`, `E₁`, `F₁` in `[−31, 32]` and int7 `A`, `B`, when `R` is a signed
permutation and `k ≤ 2^15`, or a permutation matrix and `k ≤ 2^16`. -/
def MmWUExact : Prop :=
  ∀ (β : ℤ) (k : ℕ) (A B E1 F1 R : Mat) (τ i j : ℕ), 127 ≤ β → β ≤ 129 →
    (SignedPerm k R ∧ k ≤ 2 ^ 15 ∨ Unsigned k R ∧ k ≤ 2 ^ 16) →
    (∀ i, ∀ l < k, InNoise6 (E1 i l)) → (∀ l < k, ∀ j, InNoise6 (F1 l j)) →
    (∀ i, ∀ l < k, InInt7 (A i l)) → (∀ l < k, ∀ j, InInt7 (B l j)) → τ ≤ 3 * k →
    mmWU β k A B E1 F1 R τ i j = runU β k A B E1 F1 R τ i j

/-- **Route U's `Y` is the lead's depths**: at `d = 16` and `16 ∣ k`, block `t < 3k/16` of `YU β` is route U's int32
reading of `checkedU β` at `t + 1`. -/
def YUChecked : Prop :=
  ∀ {Q R S : Type} (P : Params Q R S) (β : ℤ) (H : Q → R) (s : S) (U : Workload) (u : ℕ) (A : Mat) (t i j : ℕ),
    P.d = 16 → 16 ∣ (U.layout.shape u).k → t < 3 * (U.layout.shape u).k / 16 →
    P.YU β H s U u A t i j = wrap32 (checkedU β (U.layout.shape u).k A (U.weight (U.layout.wid u))
      (P.e1 H s U u A) (P.wtNoise H s U u).1 (P.wtNoise H s U u).2 (t + 1) i j)

end

/-! ## Route U's game instance (`Pouw.Assumptions.NCP.RouteUGame`) -/

section

open Pouw.NCP Pouw.NCP.RouteU Finset

/-- **`TT_NCP_U` is `TT_NCP` of route U's protocol**, by definition. -/
def TTNCPUIff : Prop :=
  ∀ (Q R S : Type) [Fintype Q] [DecidableEq Q] [Fintype R] [Fintype S]
    (CM : CostModel Q R S) (P : Params Q R S) (β : ℤ) (D : UnitLayout → Prop) (γ₀ : ℝ) (ε : ℕ → ℕ → ℝ),
    Assumptions.TTNCP_U β CM P D γ₀ ε ↔ Assumptions.TTNCP CM (P.toProtocolU β) P.d D γ₀ ε

end

end Pouw.Guarantees.NCP

namespace Pouw.NCP.RouteU

open Finset

/-- **`G_γ` from TT_NCP_U** (`GamePinned.GammaFromTTNCP` on route U): for every bias `β`, count `f` of forming
units per position, cost model, NCP instance whose `W_ref` is route U's, `wrefU f`, and domain on which every unit
has at least `n₀ > 0` columns and `γ₀ ≥ P.d/(3k)`, TT_NCP_U(γ₀) at `β` with error `ε` gives `G_γ` for route U's protocol at `β` with
`Ω_max = 1 + f/(3n₀)`, so `γ = 1 − (1 − γ₀)/(1 + f/(3n₀))`, and `η = ε`. -/
def GammaFromTTNCP_U : Prop :=
  ∀ (Q R S : Type) [Fintype Q] [DecidableEq Q] [Fintype R] [Fintype S]
    (CM : CostModel Q R S) (P : Params Q R S) (β : ℤ) (f : ℕ) (D : UnitLayout → Prop) (γ₀ : ℝ) (n₀ : ℕ)
    (ε : ℕ → ℕ → ℝ),
    γ₀ < 1 → 0 < n₀ → (∀ (L : UnitLayout) (u : ℕ), P.Wref L u = wrefU f L u) →
    (∀ U : Workload, U.InDomain D → ∀ u < U.layout.N, n₀ ≤ (U.layout.shape u).n) →
    (∀ U : Workload, U.InDomain D → ∀ u < U.layout.N, (P.d : ℝ) ≤ γ₀ * (3 * (U.layout.shape u).k)) →
    Assumptions.TTNCP_U β CM P D γ₀ ε →
    Gγ CM (P.toProtocolU β) D (reductionGamma γ₀ (1 + f / (3 * n₀))) ε

end Pouw.NCP.RouteU

namespace Pouw.Guarantees.NCP

section

open Pouw.NCP Pouw.NCP.RouteU Finset

/-- **`G_γ` from TT_NCP_U at ncp-v1** (`GammaFromTTNCP_U` at `β = 128`, `f = 24`): `Ω_max = 1 + 8/n₀`, so
`γ = 1 − (1 − γ₀)/(1 + 8/n₀)`. -/
def GammaFromTTNCP_U_v1 : Prop :=
  ∀ (Q R S : Type) [Fintype Q] [DecidableEq Q] [Fintype R] [Fintype S]
    (CM : CostModel Q R S) (P : Params Q R S) (D : UnitLayout → Prop) (γ₀ : ℝ) (n₀ : ℕ) (ε : ℕ → ℕ → ℝ),
    γ₀ < 1 → 0 < n₀ → (∀ (L : UnitLayout) (u : ℕ), P.Wref L u = wrefU 24 L u) →
    (∀ U : Workload, U.InDomain D → ∀ u < U.layout.N, n₀ ≤ (U.layout.shape u).n) →
    (∀ U : Workload, U.InDomain D → ∀ u < U.layout.N, (P.d : ℝ) ≤ γ₀ * (3 * (U.layout.shape u).k)) →
    Assumptions.TTNCP_U 128 CM P D γ₀ ε →
    Gγ CM (P.toProtocolU 128) D (reductionGamma γ₀ (1 + 8 / n₀)) ε

/-- **`G_γ` from TT_NCP_U at C2** (`GammaFromTTNCP_U` at `β = 129`, `f = 20`): `Ω_max = 1 + 20/(3n₀)`, so
`γ = 1 − (1 − γ₀)/(1 + 20/(3n₀))`. -/
def GammaFromTTNCP_U_C2 : Prop :=
  ∀ (Q R S : Type) [Fintype Q] [DecidableEq Q] [Fintype R] [Fintype S]
    (CM : CostModel Q R S) (P : Params Q R S) (D : UnitLayout → Prop) (γ₀ : ℝ) (n₀ : ℕ) (ε : ℕ → ℕ → ℝ),
    γ₀ < 1 → 0 < n₀ → (∀ (L : UnitLayout) (u : ℕ), P.Wref L u = wrefU 20 L u) →
    (∀ U : Workload, U.InDomain D → ∀ u < U.layout.N, n₀ ≤ (U.layout.shape u).n) →
    (∀ U : Workload, U.InDomain D → ∀ u < U.layout.N, (P.d : ℝ) ≤ γ₀ * (3 * (U.layout.shape u).k)) →
    Assumptions.TTNCP_U 129 CM P D γ₀ ε →
    Gγ CM (P.toProtocolU 129) D (reductionGamma γ₀ (1 + 20 / (3 * n₀))) ε

/-- **Satisfiability witness for TT_NCP_U** at every `γ₀ ∈ [0, 1)` (`GamePinned.TTNCPWitness` on route U): for every
bias `β` and NCP instance with at most `c` queries per unit and `W_ref ≥ 3·W_mm`, in the output-priced model at
`3·m·k·n` TT_NCP_U(γ₀) holds with error 0, and one program runs route U's `H_ref` at cost `≤ W_ref` with `N·c`
oracle calls. -/
def TTNCPUWitness : Prop :=
  ∀ (Q R : Type) [Fintype Q] [DecidableEq Q] [Fintype R] (P : Params Q R Salt) (β : ℤ) (c : ℕ) (γ₀ : ℝ),
    0 ≤ γ₀ → γ₀ < 1 → P.QueryBound c →
    (∀ U : Workload, U.InDomain leadDomain → ∀ u < U.layout.N, 3 * (U.layout.shape u).Wmm ≤ P.Wref U.layout u) →
    Assumptions.TTNCP_U β (outputPriced3 Q R Salt) P leadDomain γ₀ (fun _ _ => 0) ∧
    AdmitsRef (outputPriced3 Q R Salt) (P.toProtocolU β) leadDomain (fun U => U.layout.N * c)

end

/-! ## Route U's word layer (`Pouw.Assumptions.NCP.RouteUWord`) -/

section

open Pouw.NCP Pouw.NCP.RouteU Finset Pouw.NCP.Relation

/-- **The floor for route U under independence** (`Relation.NCPWordFloorInd` for `C^U`, with NzNCP of `C^U`):
`N·2^−(m n·(γ₀·3k/16 − 1))`. -/
def NCPWordFloorIndU : Prop :=
  ∀ (β : ℤ) (k m n N : ℕ) (R : Mat) (A B : ℕ → Mat) (adv : Word.Adversary) (γ₀ : ℝ), 16 ∣ k →
    CheckpointIndependent k 16 R → 0 ≤ γ₀ → γ₀ < 1 → Word.Valid (fun _ => positions k m n) adv →
    Word.NzNCP (fun _ => positions k m n) (checkedUPos β k m n N R A B) →
    pr (fun ω => (Word.cost (fun _ => positions k m n) N adv : ℝ) / (1 - γ₀) <
        16 * ∑ _u ∈ Word.correctSet (fun _ => positions k m n) (checkedUPos β k m n N R A B) N adv ω,
          (positions k m n : ℝ)) ≤
      N * (2 : ℝ) ^ (-((m * n : ℕ) * (γ₀ * (3 * k / 16 : ℕ) - 1)))

/-- **The int32 range of route U's checked values** (`Chain.Int7Int32` on route U): for `β ∈ {127, 128, 129}` and
int7 `A u`, `B u`, every checked value of unit `u` is int32 when `R` is a signed permutation and `k ≤ 2^15`, or a
permutation matrix and `k ≤ 2^16` (`RouteU.Int7Int32U`, `RouteU.Int7Int32UUnsigned`). -/
def Int7Int32UPos : Prop :=
  ∀ (β : ℤ) (k m n N : ℕ) (R : Mat) (A B : ℕ → Mat) (ω : Noise N m n k) (u p : ℕ), 127 ≤ β → β ≤ 129 →
    (SignedPerm k R ∧ k ≤ 2 ^ 15 ∨ Unsigned k R ∧ k ≤ 2 ^ 16) →
    (∀ i, ∀ l < k, InInt7 (A u i l)) → (∀ l < k, ∀ j, InInt7 (B u l j)) → p < positions k m n →
    InInt32 (checkedUPos β k m n N R A B ω u p)

/-- **FinalOnly for route U's protocol checked values** (`Chain.WrapFinalOnlyInd` on route U), under independence,
in the int32-exact range. -/
def WrapFinalOnlyIndU : Prop :=
  ∀ (β : ℤ) (k m n N : ℕ) (R : Mat) (A B : ℕ → Mat) (adv : Word.Adversary), 16 ∣ k →
    CheckpointIndependent k 16 R → Word.Valid (fun _ => positions k m n) adv →
    (∀ (ω : Noise N m n k) (u p : ℕ), u < N → p < positions k m n → InInt32 (checkedUPos β k m n N R A B ω u p)) →
    ∀ u < N, Word.identCount (fun _ => positions k m n) (wrappedU β k m n N R A B) adv u ≤ m * n

/-- **The floor for route U's protocol checked values** (`Chain.WrapWordFloorInd` on route U, with NzNCP of the
wrapped route-U values), under independence, in the int32-exact range. -/
def WrapWordFloorIndU : Prop :=
  ∀ (β : ℤ) (k m n N : ℕ) (R : Mat) (A B : ℕ → Mat) (adv : Word.Adversary) (γ₀ : ℝ), 16 ∣ k →
    CheckpointIndependent k 16 R → 0 ≤ γ₀ → γ₀ < 1 → Word.Valid (fun _ => positions k m n) adv →
    (∀ (ω : Noise N m n k) (u p : ℕ), u < N → p < positions k m n → InInt32 (checkedUPos β k m n N R A B ω u p)) →
    Word.NzNCP (fun _ => positions k m n) (wrappedU β k m n N R A B) →
    pr (fun ω => (Word.cost (fun _ => positions k m n) N adv : ℝ) / (1 - γ₀) <
        16 * ∑ _u ∈ Word.correctSet (fun _ => positions k m n) (wrappedU β k m n N R A B) N adv ω,
          (positions k m n : ℝ)) ≤
      N * (2 : ℝ) ^ (-((m * n : ℕ) * (γ₀ * (3 * k / 16 : ℕ) - 1)))

end

end Pouw.Guarantees.NCP
