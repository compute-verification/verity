import Definitions.Pouw.D1
import Definitions.Pouw.NCP.Defs

/-!
# NCP as a protocol (trusted)

Unit `u` of shape `(m, k, n)` with activations `A` and weight `B = B_{w(u)}`:
* noise: `E₁` (`m × k`) per unit, an expansion `expAct` of `H`'s answers to `qAct s u A w(u)`; `F₁` (`k × n`) and
  `R` (`k × k`) per (epoch, weight), an expansion `expWt` of `H`'s answers to `qWt s w(u)` (the salt is the epoch,
  and the queries do not name the unit);
* `E₂ = E₁·R`, `F₂ = −R·F₁`, `E₃ = E₁ + E₂`, `F₃ = F₁ + F₂`, and the stacked operands, block-major:
  `X = [A + E₁ | A + E₂ | −(A + E₃)]` (`m × 3k`) and `Y = [B + F₁; B + F₂; B + F₃]` (`3k × n`);
* checked values: the running sums of `X·Y` at inner depths `d, 2d, …`, the last capped at `3k`, mod `2^32` (the
  int32 accumulator's two's-complement value). The last one is `A·B` for int7 operands (`WrapInt7`), so
  the useful output is the final checked value;
* `H_ref` computes them with an int32 accumulator and outputs the accumulator at `3k`;
* `W_ref`, from the layout only (generic).

The lead's parameters (`IsLead`): `d = 16`, `E₁` and `F₁` in `[−31, 32]` (6 bits), and every `R` a signed permutation
that passes the rejection rule (`Admissible`). The domain `leadDomain` is §3.2's default shape domain with at
most `2^40` rows per weight.

`GameProofs.shiftBy c k` is the cyclic shift by `c`; at `c = 24` it passes the rule for every `k ≥ 1024` with `16 ∣ k`
(`Pouw.Guarantees.NCP.ShiftBy24Admissible`).
-/

namespace Pouw.NCP

open Finset

/-- The symmetric range `[−31, 31]` (`ThreeBlock`'s). -/
def InSmall (x : ℤ) : Prop := -31 ≤ x ∧ x ≤ 31

/-- The noise range of `E₁` and `F₁`: `[−31, 32]`, 6 bits of `H` output per entry (not `Pouw.InNoise`, the problem
statement's `[−63, 63]`). -/
def InNoise6 (x : ℤ) : Prop := -31 ≤ x ∧ x ≤ 32

/-- The rejection rule for a `k × k` matrix `R` at depth `d`: `R` is a signed permutation
`x ↦ σ·e_{π x}`; for every step `S = [t·d, (t + 1)·d)` of a block (`t < k/d`), `π(S) ∩ S = ∅` (no column of `S` has
its nonzero entry in a row of `S`) and `π(S)` lies inside no step `S′` (so, having `d` elements, it equals no step);
and `R² ≠ I`, `R² ≠ −I` on `{0, …, k − 1}`. The first clause excludes N1's structures; the second excludes step-level
involutions such as a step reversal, which a designer-chosen `R` could have (NA1); the third excludes `R² = ±I`, which
makes the zero-input checked value at depth `2k`, `E₁·(I − R²)·F₁`, zero or twice the one at depth `k`. -/
def Admissible (k d : ℕ) (R : Mat) : Prop :=
  SignedPerm k R ∧
  (∀ t < k / d,
    (∀ j, t * d ≤ j → j < (t + 1) * d → ∀ i, t * d ≤ i → i < (t + 1) * d → R i j = 0) ∧
    ∀ t' < k / d, ¬ ∀ j, t * d ≤ j → j < (t + 1) * d → ∃ i, t' * d ≤ i ∧ i < (t' + 1) * d ∧ R i j ≠ 0) ∧
  (¬ ∀ i < k, ∀ j < k, mm k R R i j = if i = j then 1 else 0) ∧
  ¬ ∀ i < k, ∀ j < k, mm k R R i j = if i = j then -1 else 0

namespace GameProofs

/-- The cyclic shift by `c`: column `j < k` has its `1` in row `j + c mod k`. -/
def shiftBy (c k : ℕ) : Mat := fun i j => if j < k ∧ i = (if j + c < k then j + c else j + c - k) then 1 else 0

end GameProofs

/-- NCP's parameters. -/
structure Params (Q R S : Type) where
  d : ℕ
  qAct : S → ℕ → Mat → ℕ → List Q
  qWt : S → ℕ → List Q
  expAct : Shape → List R → Mat
  expWt : Shape → List R → Mat × Mat
  Wref : Layout → ℕ → ℕ

namespace Params

variable {Q R S : Type} (P : Params Q R S)

/-- Unit `u`'s `E₁`. -/
def e1 (H : Q → R) (s : S) (U : Workload) (u : ℕ) (A : Mat) : Mat :=
  P.expAct (U.layout.shape u) ((P.qAct s u A (U.layout.wid u)).map H)

/-- Unit `u`'s weight noise `(F₁, R)`, shared by every unit on the same (epoch, weight). -/
def wtNoise (H : Q → R) (s : S) (U : Workload) (u : ℕ) : Mat × Mat :=
  P.expWt (U.layout.shape u) ((P.qWt s (U.layout.wid u)).map H)

/-- The stacked activations `X`. -/
def opX (H : Q → R) (s : S) (U : Workload) (u : ℕ) (A : Mat) : Mat :=
  stackX (U.layout.shape u).k A (P.e1 H s U u A) (P.wtNoise H s U u).2

/-- The stacked weights `Y`. -/
def opY (H : Q → R) (s : S) (U : Workload) (u : ℕ) : Mat :=
  stackY (U.layout.shape u).k (U.weight (U.layout.wid u)) (P.wtNoise H s U u).2 (P.wtNoise H s U u).1

/-- The checked values: block `t` is the running sum of `X·Y` up to `blockEnd d (3k) t`, mod `2^32` (its int32
two's-complement value, `mmW`), so no int32 range bound is needed. -/
def Y (H : Q → R) (s : S) (U : Workload) (u : ℕ) (A : Mat) : Checked :=
  fun t => mmW (D1.blockEnd P.d (3 * (U.layout.shape u).k) t) (P.opX H s U u A) (P.opY H s U u)

/-- The compared entries: blocks `t < ⌈3k/d⌉`, rows `i < m`, columns `j < n`. -/
def checkIdx (sh : Shape) : Finset (ℕ × ℕ × ℕ) :=
  range (D1.nBlocks P.d (3 * sh.k)) ×ˢ range sh.m ×ˢ range sh.n

/-- `H_ref`'s checked values: the int32 accumulator of `X·Y` read at each block end. -/
def refChecked (H : Q → R) (s : S) (U : Workload) (u : ℕ) (A : Mat) : Checked :=
  fun t => mmW (D1.blockEnd P.d (3 * (U.layout.shape u).k) t) (P.opX H s U u A) (P.opY H s U u)

/-- `H_ref`'s useful output: the int32 accumulator at `3k`. -/
def refOut (H : Q → R) (s : S) (U : Workload) (u : ℕ) (A : Mat) : Mat :=
  mmW (3 * (U.layout.shape u).k) (P.opX H s U u A) (P.opY H s U u)

/-- NCP as a protocol. -/
def toProtocol : Protocol Q R S where
  Y := P.Y
  checkIdx := P.checkIdx
  refChecked := P.refChecked
  refOut := P.refOut
  Wref := P.Wref

/-- `E₁` and `F₁` lie in `[−31, 32]` inside the shape, for every oracle answer. -/
def NoiseRange : Prop :=
  (∀ (sh : Shape) (a : List R), ∀ i < sh.m, ∀ l < sh.k, InNoise6 (P.expAct sh a i l)) ∧
  (∀ (sh : Shape) (b : List R), ∀ l < sh.k, ∀ j < sh.n, InNoise6 ((P.expWt sh b).1 l j))

/-- Every `R` of a default-domain shape is a signed permutation. -/
def RSigned : Prop := ∀ (sh : Shape) (b : List R), Shape.InDefaultDomain sh → SignedPerm sh.k (P.expWt sh b).2

/-- Every `R` of a default-domain shape passes the rejection rule. -/
def RAdmissible : Prop :=
  ∀ (sh : Shape) (b : List R), Shape.InDefaultDomain sh → Admissible sh.k P.d (P.expWt sh b).2

/-- At most `c` oracle queries per unit for its noise. -/
def QueryBound (c : ℕ) : Prop := ∀ (s : S) (u : ℕ) (A : Mat) (w : ℕ), (P.qAct s u A w).length + (P.qWt s w).length ≤ c

/-- The lead's parameters: `d = 16`, noise in `[−31, 32]`, and the rejection rule. `W_ref` is not fixed here. -/
def IsLead : Prop := P.d = 16 ∧ P.NoiseRange ∧ P.RAdmissible

/-- The noise of the stacked product at zero activations and weights, as a noise source of `Pouw.Barrier.gemmModel`:
source `0` is `X` at `A = 0`, source `1` is `Y` at `B = 0`. -/
def nz (H : Q → R) (s : S) (U : Workload) (u : ℕ) (A : Mat) (i : ℕ) : Mat :=
  if i = 0 then stackX (U.layout.shape u).k 0 (P.e1 H s U u A) (P.wtNoise H s U u).2
  else if i = 1 then stackY (U.layout.shape u).k 0 (P.wtNoise H s U u).2 (P.wtNoise H s U u).1 else 0

end Params

/-- The lead's domain: §3.2's default shapes, and at most `2^40` rows on any weight. -/
def leadDomain (L : Layout) : Prop := L.ShapesIn Shape.InDefaultDomain ∧ ∀ w, L.rowsOf w ≤ 2 ^ 40

end Pouw.NCP
